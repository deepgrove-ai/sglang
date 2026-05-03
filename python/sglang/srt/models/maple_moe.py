# coding=utf-8
"""Inference-only Maple model — 1-1 port of modeling_maple.py for SGLang.

SGLang compatibility additions only:
  - VocabParallelEmbedding for word_embeddings
  - ParallelLMHead + LogitsProcessor for lm_head / output
  - RadixAttention replaces HF Cache + flash_attention_forward (same math)
  - load_weights() for weight-file loading
All computation (norms, MoE routing, MLP, RoPE, QKV math) is identical to HF.
"""

import logging
import math
import os
import atexit as _atexit
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.maple_fa.fa3_utils import flash_attention_forward as fa3_flash_attention_forward

logger = logging.getLogger(__name__)

# ── debug tensor dumps ────────────────────────────────────────────────────────
_LOGIT_DEBUG = os.environ.get("LOGIT_DEBUG", "").lower() == "true"
_SGL_DBG  = {}
_SGL_STEP = [0]
_SGL_LAYER = [0]

def _sgl_dbg(name, t):
    if not _LOGIT_DEBUG:
        return
    key = f"s{_SGL_STEP[0]:03d}_l{_SGL_LAYER[0]:02d}_{name}"
    _SGL_DBG[key] = t.reshape(-1, t.shape[-1])[-1].detach().cpu().to(torch.float32).clone()

def _sgl_save():
    torch.save(_SGL_DBG, "/tmp/sgl_debug.pt")
    print(f"[SGL debug] saved {len(_SGL_DBG)} tensors → /tmp/sgl_debug.pt")

if _LOGIT_DEBUG:
    _atexit.register(_sgl_save)
# ─────────────────────────────────────────────────────────────────────────────


# ── identical to modeling_maple.py ───────────────────────────────────────────

class MapleRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else: 
            self.rope_type = "default"

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = getattr(config, "rope_theta")
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        freqs = torch.cat([freqs, freqs], dim=-1)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype), freqs.float()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class MapleMLP(nn.Module):
    def __init__(self, config, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate_weight, up_weight, down_weight = self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight
        return F.linear(
            self.act_fn(F.linear(x, gate_weight)) * F.linear(x, up_weight),
            down_weight,
        )


class MapleRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        weight_float = self.weight.float()
        res = weight_float * hidden_states
        return res.to(input_dtype)


class MapleGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        routing_weights = F.softmax(logits, dim=1, dtype=torch.float)
        scores, topk_idx = torch.topk(routing_weights, self.top_k, dim=-1)
        scores = scores.type_as(logits)
        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight, logits


class MapleSparseMoeBlock(nn.Module):
    """Unfused MoE block — identical to HF; always uses moe_infer path at inference."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([
            MapleMLP(config=config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.gate = MapleGate(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # SGLang: hidden_states is (num_tokens, hidden_size); add fake batch dim to match HF.
        is_2d = hidden_states.dim() == 2
        if is_2d:
            hidden_states = hidden_states.unsqueeze(0)

        bsz, seq_len, h = hidden_states.shape
        topk_idx, topk_weight, _router_logits = self.gate(hidden_states)
        _sgl_dbg("router_logits", _router_logits)
        _sgl_dbg("router_probs", topk_weight)
        hidden_states = hidden_states.view(-1, h)
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(bsz, seq_len, h)
        _sgl_dbg("ffn_out", y.reshape(-1, y.shape[-1]))

        if is_2d:
            y = y.squeeze(0)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        _pfx = f"s{_SGL_STEP[0]:03d}_l{_SGL_LAYER[0]:02d}" if _LOGIT_DEBUG else None
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            if _LOGIT_DEBUG:
                _SGL_DBG[f"{_pfx}_expert{i:03d}_in"] = tokens_for_this_expert.detach().cpu().float().clone()
            expert_out = expert(tokens_for_this_expert)
            if _LOGIT_DEBUG:
                _SGL_DBG[f"{_pfx}_expert{i:03d}_out"] = expert_out.detach().cpu().float().clone()
            outputs.append(expert_out.to(x.device))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        weighted = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
        )
        if _LOGIT_DEBUG:
            _SGL_DBG[f"{_pfx}_expert_weighted"] = weighted.detach().cpu().float().clone()
        final_out = weighted.sum(dim=1).type(new_x.dtype)
        return final_out


# ── SGLang-adapted attention (same math, RadixAttention for KV cache) ─────────

class MapleAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5

        partial_rotary_factor = 0.5
        self.rope_dim = int(self.head_dim * partial_rotary_factor)

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True

        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        self.q_norm = MapleRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MapleRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)

        # SGLang KV cache manager (replaces HF DynamicCache + flash_attention_forward).
        # Sliding-window layers: pass the window size; NoPe (full-attention) layers: -1.
        sliding_window_size = self.sliding_window if self.sliding_window is not None else -1
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_key_value_heads,
            layer_id=layer_idx,
            sliding_window_size=sliding_window_size,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        num_tokens, _ = hidden_states.shape

        # ── QKV (identical fused computation as HF) ───────────────────────────
        qkv_weight = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        out_qkv = F.linear(hidden_states, qkv_weight)
        cos, sin, _ = position_embeddings
        qkv = out_qkv.view(num_tokens, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        q, k, v = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        # print(q.shape, k.shape, v.shape)

        # ── raw projection outputs (before norm / RoPE) ───────────────────────
        _sgl_dbg("q_proj", q.reshape(num_tokens, -1))
        _sgl_dbg("k_proj", k.reshape(num_tokens, -1))
        _sgl_dbg("v_proj", v.reshape(num_tokens, -1))

        # ── Q/K norm (per-head, identical to HF) ──────────────────────────────
        q = self.q_norm(q)
        k = self.k_norm(k)
        _sgl_dbg("q_post_norm", q.reshape(num_tokens, -1))
        _sgl_dbg("k_post_norm", k.reshape(num_tokens, -1))

        # ── RoPE — only for sliding_attention layers (identical to HF) ────────
        if self.sliding_window is not None and position_embeddings is not None:
            cos = cos.squeeze(0)   # (num_tokens, head_dim)
            sin = sin.squeeze(0)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            _sgl_dbg("q_post_rope", q.reshape(num_tokens, -1))
            _sgl_dbg("k_post_rope", k.reshape(num_tokens, -1))

        # ── Debug dumps ────────────────────────────────────────────────────────
        _sgl_dbg("q", q.reshape(num_tokens, -1))
        _sgl_dbg("k", k.reshape(num_tokens, -1))
        _sgl_dbg("v", v.reshape(num_tokens, -1))

        # ── Attention — bypass RadixAttention, call flash_attention_forward identically to HF ──
        # q/k/v are [num_tokens, n_heads, head_dim]; write K/V to pool then compute attention.

        # Write K/V to pool (pool stores [pool_size, n_kv_heads, head_dim])
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn, forward_batch.out_cache_loc,
            k.contiguous(), v.contiguous(),
            self.attn.k_scale, self.attn.v_scale,
        )

        sliding_window = self.sliding_window  # None for full-attention layers

        if forward_batch.forward_mode.is_decode():
            # One new query token per sequence. Gather full K/V history from pool per sequence.
            # q[i]: [n_heads, D] → unsqueeze → [1, n_heads, 1, D] = [B, H, S, D] for flash_attention_forward
            # k_pool[tokens]: [kv_len, H_k, D] → permute → [1, H_k, kv_len, D]
            k_pool = forward_batch.token_to_kv_pool.get_key_buffer(self.layer_idx)
            v_pool = forward_batch.token_to_kv_pool.get_value_buffer(self.layer_idx)
            req_to_token = forward_batch.req_to_token_pool.req_to_token
            req_pool_indices = forward_batch.req_pool_indices
            seq_lens = forward_batch.seq_lens

            outputs = []
            for i in range(num_tokens):  # num_tokens == batch_size during decode
                kv_len = int(seq_lens[i])
                tokens = req_to_token[int(req_pool_indices[i]), :kv_len]
                q_i = q[i].unsqueeze(0).unsqueeze(2)                  # [1, n_heads, 1, D]
                k_i = k_pool[tokens].unsqueeze(0).permute(0, 2, 1, 3) # [1, H_k, kv_len, D]
                v_i = v_pool[tokens].unsqueeze(0).permute(0, 2, 1, 3) # [1, H_k, kv_len, D]
                out_i, _ = fa3_flash_attention_forward(
                    self, q_i, k_i, v_i,
                    attention_mask=None,
                    scaling=self.scaling,
                    sliding_window=sliding_window,
                )
                outputs.append(out_i.view(self.num_heads * self.head_dim))
            attn_output = torch.stack(outputs)  # [batch_size, num_heads * head_dim]

        else:
            # Extend (prefill): flat [num_tokens, H, D] → [1, H, num_tokens, D] = [B, H, S, D].
            # Pass cu_seqlens via kwargs to trigger the varlen branch inside _flash_attention_forward,
            # identical to HF's path after _upad_input strips padding.
            extend_seq_lens = forward_batch.extend_seq_lens
            cu_seqlens = F.pad(extend_seq_lens.cumsum(0, dtype=torch.int32), (1, 0))
            max_seqlen = int(extend_seq_lens.max())
            dummy_pos = torch.zeros(1, dtype=torch.long, device=q.device)

            q3 = q.view(1, num_tokens, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            k3 = k.view(1, num_tokens, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()
            v3 = v.view(1, num_tokens, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()

            out, _ = fa3_flash_attention_forward(
                self, q3, k3, v3,
                attention_mask=None,
                scaling=self.scaling,
                sliding_window=sliding_window,
                position_ids=dummy_pos,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
            )
            # out: [1, num_tokens, n_heads, head_dim]
            attn_output = out.reshape(num_tokens, self.num_heads * self.head_dim)
        _sgl_dbg("attn_pre_o", attn_output)

        # ── Output projection (identical to HF) ───────────────────────────────
        attn_output = F.linear(attn_output, self.o_proj.weight)
        _sgl_dbg("o_proj", attn_output)

        return attn_output


# ── Decoder layer — identical structure to HF, simplified forward ─────────────

class MapleDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = MapleAttention(config=config, layer_idx=layer_idx)
        self.mlp = MapleSparseMoeBlock(config)

        self.input_layernorm = MapleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MapleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        _SGL_LAYER[0] = self.layer_idx
        initial_dtype = hidden_states.dtype
        _sgl_dbg("layer_in", hidden_states)

        # ── Attention block (mirrors HF exactly) ──────────────────────────────
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        _sgl_dbg("post_input_norm", hidden_states)

        hidden_states = self.self_attn(positions, hidden_states, forward_batch, position_embeddings)
        _sgl_dbg("post_attn", hidden_states)
        # hidden_states = residual + hidden_states.to(residual.device)
        hidden_states = (residual.to(torch.float32) + hidden_states.to(torch.float32)).to(initial_dtype)
        _sgl_dbg("post_attn_residual", hidden_states)

        # ── MLP block (mirrors HF exactly) ────────────────────────────────────
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        _sgl_dbg("post_post_attn_norm", hidden_states)

        hidden_states = self.mlp(hidden_states)
        _sgl_dbg("post_mlp", hidden_states)

        # hidden_states = residual + hidden_states.to(residual.device)
        hidden_states = (residual.to(torch.float32) + hidden_states.to(torch.float32)).to(initial_dtype)
        _sgl_dbg("post_ffn_residual", hidden_states)
        _sgl_dbg("layer_out", hidden_states)

        return hidden_states


# ── Model backbone ────────────────────────────────────────────────────────────

class MapleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # VocabParallelEmbedding is weight-compatible with nn.Embedding; same weight name.
        self.word_embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            MapleDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        self.norm = MapleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MapleRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.word_embeddings(input_ids)
        else:
            hidden_states = input_embeds

        _SGL_LAYER[0] = 0
        _sgl_dbg("embed", hidden_states)

        # Compute RoPE cos/sin once; sliding-window layers apply it, NoPe layers ignore it.
        # positions is (num_tokens,); rotary_emb expects (batch, seq) → unsqueeze to (1, T).
        position_embeddings = self.rotary_emb(hidden_states, positions.unsqueeze(0))
        _sgl_dbg("rope_cos", position_embeddings[0].reshape(-1, position_embeddings[0].shape[-1]))
        _sgl_dbg("rope_sin", position_embeddings[1].reshape(-1, position_embeddings[1].shape[-1]))

        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_batch, position_embeddings)

        hidden_states = self.norm(hidden_states)
        _SGL_LAYER[0] = 99
        _sgl_dbg("final_norm", hidden_states)

        return hidden_states


# ── Causal LM head ────────────────────────────────────────────────────────────

class MapleForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.model = MapleModel(config)
        self.vocab_size = config.vocab_size

        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.word_embeddings

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        if _LOGIT_DEBUG:
            _step = _SGL_STEP[0]
            _lm_out = torch.nn.functional.linear(hidden_states, self.lm_head.weight).float()
            _lm_2d = _lm_out.reshape(-1, _lm_out.shape[-1]).detach().cpu()
            _SGL_DBG[f"s{_step:03d}_output_logits"] = _lm_2d.clone()
            _SGL_DBG[f"s{_step:03d}_output_probs"] = torch.softmax(_lm_2d[-1], dim=-1).clone()
            _SGL_STEP[0] += 1
            torch.save(_SGL_DBG, "/tmp/sgl_debug.pt")
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name not in params_dict:
                logger.debug(f"Skipping {name} — not in model")
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = MapleForCausalLM
