# coding=utf-8
"""Custom attention backend for Maple model comparison runs.

Both extend and decode use the same flash_attn kernels as the HF reference model:
  - Extend: flash_attn_varlen_func via _flash_attention_forward (identical path to HF prefill)
  - Decode: flash_attn_func via _flash_attention_forward (identical path to HF decode),
            with K/V gathered from the pool per-request
Edge cases (MLA, cross-attention, prefix cache) fall through to FlashAttentionBackend.
"""

import torch

from sglang.srt.layers.attention.attention_registry import register_attention_backend
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

from .fa3_utils import _flash_attention_forward


class MapleFABackend(FlashAttentionBackend):
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        q_rope=None,
        k_rope=None,
        sinks=None,
    ):
        # ── KV pool write (identical to parent; done once here) ──────────────
        if k is not None and save_kv_cache and not self.use_mla:
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        # ── Detect cases we can't handle: fall back to parent (skip its KV write) ──
        metadata = self.forward_metadata
        cu_seqlens_q = metadata.cu_seqlens_q
        max_seqlen_q = metadata.max_seq_len_q
        cache_seqlens = metadata.cache_seqlens_int32

        total_new = int(cu_seqlens_q[-1])
        has_prefix = int(cache_seqlens.sum()) > total_new

        if self.use_mla or layer.is_cross_attention or q_rope is not None or has_prefix:
            return super().forward_extend(
                q, k, v, layer, forward_batch,
                save_kv_cache=False,  # pool already written above
                q_rope=q_rope, k_rope=k_rope, sinks=sinks,
            )

        # ── Attention via _flash_attention_forward (identical path to HF reference) ─
        # _flash_attention_forward expects [B, S, H, D]; add a batch dim then remove it.
        num_tokens = int(cu_seqlens_q[-1])
        q3 = q.contiguous().view(1, num_tokens, layer.tp_q_head_num, layer.head_dim)
        k3 = k.view(1, num_tokens, layer.tp_k_head_num, layer.head_dim)
        v3 = v.view(1, num_tokens, layer.tp_v_head_num, layer.v_head_dim)

        is_swa = layer.sliding_window_size is not None and layer.sliding_window_size > -1
        sliding_window = layer.sliding_window_size if is_swa else None

        # A non-None position_ids triggers the varlen branch; cu_seq_lens_q being
        # non-None means the pre-computed boundaries are used directly (position_ids
        # itself is ignored inside that branch).
        dummy_pos = torch.zeros(1, dtype=torch.long, device=q.device)

        out = _flash_attention_forward(
            q3, k3, v3,
            attention_mask=None,
            query_length=num_tokens,
            is_causal=True,
            softmax_scale=layer.scaling,
            sliding_window=sliding_window,
            position_ids=dummy_pos,
            cu_seq_lens_q=cu_seqlens_q,
            cu_seq_lens_k=cu_seqlens_q,  # no prefix: kv layout == q layout
            max_length_q=max_seqlen_q,
            max_length_k=max_seqlen_q,
        )
        # out: [1, num_tokens, num_heads, head_dim] → [num_tokens, num_heads * head_dim]
        return out.reshape(num_tokens, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        q_rope=None,
        k_rope=None,
        sinks=None,
    ):
        # ── Write new token's K/V to pool ────────────────────────────────────
        if k is not None and save_kv_cache and not self.use_mla:
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        if self.use_mla or layer.is_cross_attention or q_rope is not None:
            return super().forward_decode(
                q, k, v, layer, forward_batch,
                save_kv_cache=False,
                q_rope=q_rope, k_rope=k_rope, sinks=sinks,
            )

        batch_size = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        is_swa = layer.sliding_window_size is not None and layer.sliding_window_size > -1
        sliding_window = layer.sliding_window_size if is_swa else None

        # ── Per-request attention via flash_attn_func (identical to HF decode path) ─
        # _flash_attention_forward with no mask/position_ids falls through to
        # flash_attn_func — the exact branch HF takes for single-token decode steps.
        outputs = []
        for i in range(batch_size):
            kv_len = int(seq_lens[i])
            tokens = req_to_token[int(req_pool_indices[i]), :kv_len]

            q_i = q[i].view(1, 1, layer.tp_q_head_num, layer.head_dim)   # [1, 1, H, D]
            k_i = k_cache[tokens].unsqueeze(0)                             # [1, kv_len, H_k, D]
            v_i = v_cache[tokens].unsqueeze(0)                             # [1, kv_len, H_v, D]

            out_i = _flash_attention_forward(
                q_i, k_i, v_i,
                attention_mask=None,
                query_length=1,
                is_causal=True,
                softmax_scale=layer.scaling,
                sliding_window=sliding_window,
            )
            outputs.append(out_i.view(layer.tp_q_head_num * layer.v_head_dim))

        return torch.stack(outputs, dim=0)  # [batch_size, num_heads * head_dim]


@register_attention_backend("maple_fa")
def _create_maple_fa_backend(runner):
    return MapleFABackend(runner)
