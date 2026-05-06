"""
simple_varlen_flash_attention.py

Triton-based variable-length flash attention — drop-in replacement for fa3.py.
Exports flash_attention_forward() with the identical signature.

All arithmetic is in float32 for reproducibility. No flash_attn dependency.
"""
from typing import Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Padding / unpadding helpers (no flash_attn.bert_padding dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _get_unpad_data(attention_mask: torch.Tensor):
    seqlens    = attention_mask.sum(-1, dtype=torch.int32)
    indices    = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = int(seqlens.max())
    cu_seqlens = F.pad(torch.cumsum(seqlens, 0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen


def _upad_input(q, k, v, attention_mask, query_length):
    indices_k, cu_seqlens_k, max_k = _get_unpad_data(attention_mask)
    B, kv_len, n_kv, d = k.shape

    k = k.reshape(B * kv_len, n_kv, d)[indices_k]
    v = v.reshape(B * kv_len, n_kv, d)[indices_k]

    if query_length == kv_len:
        q          = q.reshape(B * kv_len, -1, d)[indices_k]
        cu_seqlens_q = cu_seqlens_k
        max_q        = max_k
        indices_q    = indices_k
    elif query_length == 1:
        max_q        = 1
        cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device=q.device)
        indices_q    = cu_seqlens_q[:-1]
        q            = q.squeeze(1)
    else:
        attention_mask = attention_mask[:, -query_length:]
        seqlens_q      = attention_mask.sum(-1, dtype=torch.int32)
        indices_q      = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_q          = int(seqlens_q.max())
        cu_seqlens_q   = F.pad(torch.cumsum(seqlens_q, 0, dtype=torch.int32), (1, 0))
        q = q.reshape(B * query_length, -1, d)[indices_q]

    return q, k, v, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_q, max_k)


def _pad_input(x: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    out = x.new_zeros(batch * seqlen, *x.shape[1:])
    out[indices] = x
    return out.view(batch, seqlen, *x.shape[1:])


def _prepare_from_position_ids(q, k, v, position_ids):
    q = q.view(-1, q.size(-2), q.size(-1))
    k = k.contiguous().view(-1, k.size(-2), k.size(-1))
    v = v.contiguous().view(-1, v.size(-2), v.size(-1))
    position_ids = position_ids.flatten()
    indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)
    cu_seq_lens = torch.cat([
        indices_q[position_ids == 0],
        torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
    ])
    max_length = int(position_ids.max()) + 1
    return q, k, v, indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _flash_fwd_kernel(
    Q, K, V, Out,
    cu_seqlens_q, cu_seqlens_k,
    softmax_scale,
    stride_qm, stride_qh, stride_qd,
    stride_km, stride_kh, stride_kd,
    stride_vm, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    GQA_RATIO: tl.constexpr,   # n_heads_q // n_heads_kv
    BLOCK_M:   tl.constexpr,   # Q tile rows
    BLOCK_N:   tl.constexpr,   # KV tile rows
    HEAD_DIM:  tl.constexpr,   # must be power-of-2
    CAUSAL:    tl.constexpr,   # bool
    WINDOW:    tl.constexpr,   # left sliding-window size; 0 = full attention
):
    """
    Grid: (n_heads_q, ceil(max_seqlen_q / BLOCK_M), batch)
    Q/K/V layout: [total_tokens, n_heads, head_dim]  (varlen packed, no padding)
    """
    head_h  = tl.program_id(0)
    m_blk   = tl.program_id(1)
    batch   = tl.program_id(2)
    kv_head = head_h // GQA_RATIO

    # ── Sequence boundaries ──────────────────────────────────────────────────
    q_start  = tl.load(cu_seqlens_q + batch    ).to(tl.int32)
    q_end    = tl.load(cu_seqlens_q + batch + 1).to(tl.int32)
    k_start  = tl.load(cu_seqlens_k + batch    ).to(tl.int32)
    k_end    = tl.load(cu_seqlens_k + batch + 1).to(tl.int32)
    seqlen_q = q_end - q_start
    seqlen_k = k_end - k_start

    q_blk_off = m_blk * BLOCK_M
    if q_blk_off >= seqlen_q:
        return  # empty tile for this (batch, m_blk)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # ── Load Q tile [BLOCK_M, HEAD_DIM] ─────────────────────────────────────
    q_valid = offs_m < (seqlen_q - q_blk_off)
    q_toks  = q_start + q_blk_off + offs_m          # global token indices

    q_ptrs = (Q
              + q_toks[:, None] * stride_qm
              + head_h          * stride_qh
              + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0).to(tl.float32) * softmax_scale

    # ── Online-softmax state ─────────────────────────────────────────────────
    # Use -1e9 (not -inf) as initial max to avoid nan from (−inf − −inf) = nan.
    m_i = tl.full((BLOCK_M,), -1e9, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Right-aligned causal offset — identical to flash_attn_varlen_func behaviour:
    # when seqlen_q < seqlen_k (e.g. single-token decode), the Q is treated as
    # sitting at the END of the K sequence so it can attend to all past tokens.
    # For prefill seqlen_q == seqlen_k so q_off = 0 and behaviour is unchanged.
    q_off = seqlen_k - seqlen_q   # 0 for prefill; kv_len - 1 for single-token decode

    # ── KV tile range ────────────────────────────────────────────────────────
    if CAUSAL:
        # Highest aligned Q position (exclusive) in this block
        kv_hi = tl.minimum(tl.cdiv(q_off + q_blk_off + BLOCK_M, BLOCK_N),
                           tl.cdiv(seqlen_k, BLOCK_N))
    else:
        kv_hi = tl.cdiv(seqlen_k, BLOCK_N)

    # ── Main loop over KV tiles ──────────────────────────────────────────────
    for n_blk in range(0, kv_hi):
        k_blk_off = n_blk * BLOCK_N
        k_valid   = offs_n < (seqlen_k - k_blk_off)
        k_toks    = k_start + k_blk_off + offs_n

        k_ptrs = (K
                  + k_toks[:, None] * stride_km
                  + kv_head         * stride_kh
                  + offs_d[None, :] * stride_kd)
        v_ptrs = (V
                  + k_toks[:, None] * stride_vm
                  + kv_head         * stride_vh
                  + offs_d[None, :] * stride_vd)

        k = tl.load(k_ptrs, mask=k_valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0).to(tl.float32)

        # QK^T — (BLOCK_M, BLOCK_N); allow_tf32=False forces full IEEE fp32
        # (TF32 default on Ampere/Hopper reduces mantissa to 10 bits)
        qk = tl.dot(q, tl.trans(k), allow_tf32=False)

        # Sequence-boundary mask (padding / out-of-range KV tokens)
        qk = tl.where(k_valid[None, :], qk, float("-inf"))

        # Right-aligned causal and sliding-window masks
        q_pos = q_off + q_blk_off + offs_m   # aligned position in K-coordinate space
        k_pos = k_blk_off + offs_n

        if CAUSAL:
            qk = tl.where(q_pos[:, None] >= k_pos[None, :], qk, float("-inf"))

        if CAUSAL and WINDOW > 0:
            qk = tl.where(q_pos[:, None] - k_pos[None, :] < WINDOW, qk, float("-inf"))

        # Online softmax update (standard two-pass-free formulation)
        m_ij  = tl.max(qk, axis=1)                  # (BLOCK_M,)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i  - m_new)                 # rescale factor for acc
        p     = tl.exp(qk   - m_new[:, None])        # (BLOCK_M, BLOCK_N)
        p     = tl.where(k_valid[None, :], p, 0.0)   # zero masked positions

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=False)
        m_i = m_new

    # ── Normalise and write output ───────────────────────────────────────────
    acc = acc / (l_i[:, None] + 1e-9)

    out_ptrs = (Out
                + q_toks[:, None] * stride_om
                + head_h          * stride_oh
                + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=q_valid[:, None])


# ─────────────────────────────────────────────────────────────────────────────
# Python launch wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _triton_varlen_fwd(
    q:            torch.Tensor,   # (total_q, n_heads_q,  head_dim)
    k:            torch.Tensor,   # (total_k, n_heads_kv, head_dim)
    v:            torch.Tensor,   # (total_k, n_heads_kv, head_dim)
    cu_seqlens_q: torch.Tensor,   # (batch+1,) int32
    cu_seqlens_k: torch.Tensor,   # (batch+1,) int32
    max_seqlen_q: int,
    softmax_scale: float,
    causal:       bool,
    window:       int,            # 0 = full attention, >0 = left-window size
) -> torch.Tensor:
    total_q, n_heads_q, head_dim = q.shape
    n_heads_kv = k.shape[1]
    batch = cu_seqlens_q.shape[0] - 1

    assert n_heads_q % n_heads_kv == 0, "n_heads_q must be divisible by n_heads_kv"
    assert head_dim in (32, 64, 128, 256), f"Unsupported head_dim={head_dim}"

    out = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64
    grid    = (n_heads_q, triton.cdiv(max_seqlen_q, BLOCK_M), batch)

    _flash_fwd_kernel[grid](
        q, k, v, out,
        cu_seqlens_q.to(torch.int32).contiguous(),
        cu_seqlens_k.to(torch.int32).contiguous(),
        softmax_scale,
        q.stride(0),   q.stride(1),   q.stride(2),
        k.stride(0),   k.stride(1),   k.stride(2),
        v.stride(0),   v.stride(1),   v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        GQA_RATIO = n_heads_q // n_heads_kv,
        BLOCK_M   = BLOCK_M,
        BLOCK_N   = BLOCK_N,
        HEAD_DIM  = head_dim,
        CAUSAL    = causal,
        WINDOW    = window,
    )
    return out


def _varlen_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                softmax_scale=None, causal=False, window_size=None, **_):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    window = int(window_size[0]) if (window_size is not None and window_size[0] > 0) else 0
    return _triton_varlen_fwd(
        q.contiguous(), k.contiguous(), v.contiguous(),
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, softmax_scale, causal, window,
    )


def _dense_fwd(q, k, v, softmax_scale=None, causal=False, window_size=None, **_):
    """Convert dense (B, T, H, D) layout to varlen and run the Triton kernel."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    window = int(window_size[0]) if (window_size is not None and window_size[0] > 0) else 0
    B, q_len, nq, d = q.shape
    kv_len, nkv     = k.shape[1], k.shape[2]

    q_flat = q.contiguous().view(B * q_len,  nq,  d)
    k_flat = k.contiguous().view(B * kv_len, nkv, d)
    v_flat = v.contiguous().view(B * kv_len, nkv, d)

    cu_q = torch.arange(0, (B + 1) * q_len,  q_len,  dtype=torch.int32, device=q.device)
    cu_k = torch.arange(0, (B + 1) * kv_len, kv_len, dtype=torch.int32, device=k.device)

    out_flat = _triton_varlen_fwd(q_flat, k_flat, v_flat, cu_q, cu_k,
                                  q_len, softmax_scale, causal, window)
    return out_flat.view(B, q_len, nq, d)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch — mirrors fa3.py's _flash_attention_forward exactly
# ─────────────────────────────────────────────────────────────────────────────

def _flash_attention_forward(
    query_states:  torch.Tensor,
    key_states:    torch.Tensor,
    value_states:  torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length:  int,
    is_causal:     bool,
    dropout:       float = 0.0,
    position_ids:  Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap:       Optional[float] = None,
    deterministic: Optional[bool] = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q:  Optional[int] = None,
    max_length_k:  Optional[int] = None,
    target_dtype:  Optional[torch.dtype] = None,
    **kwargs,
):
    causal = is_causal if not use_top_left_mask else (is_causal and query_length != 1)
    window_size = (sliding_window, 0) if sliding_window is not None else None

    if attention_mask is not None:
        batch_size = query_states.shape[0]
        q_u, k_u, v_u, idx_q, (cu_q, cu_k), (max_q, max_k) = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )
        out_u = _varlen_fwd(q_u, k_u, v_u, cu_q, cu_k, max_q, max_k,
                            softmax_scale=softmax_scale, causal=causal,
                            window_size=window_size)
        return _pad_input(out_u, idx_q, batch_size, query_length)

    elif position_ids is not None and max_length_q is not None:
        batch_size = query_states.size(0)
        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            q_u, k_u, v_u, _, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = \
                _prepare_from_position_ids(query_states, key_states, value_states, position_ids)
        else:
            q_u = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
            k_u = key_states.reshape(  -1, key_states.size(-2),   key_states.size(-1))
            v_u = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))
        out = _varlen_fwd(q_u, k_u, v_u, cu_seq_lens_q, cu_seq_lens_k,
                          max_length_q, max_length_k,
                          softmax_scale=softmax_scale, causal=causal,
                          window_size=window_size)
        return out.view(batch_size, -1, out.size(-2), out.size(-1))

    else:
        return _dense_fwd(query_states, key_states, value_states,
                          softmax_scale=softmax_scale, causal=causal,
                          window_size=window_size)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — identical signature to fa3.flash_attention_forward
# ─────────────────────────────────────────────────────────────────────────────

class FlashAttentionKwargs(TypedDict, total=False):
    cu_seq_lens_q: Optional[torch.LongTensor]
    cu_seq_lens_k: Optional[torch.LongTensor]
    max_length_q:  Optional[int]
    max_length_k:  Optional[int]


def flash_attention_forward(
    module:         torch.nn.Module,
    query:          torch.Tensor,
    key:            torch.Tensor,
    value:          torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout:        float = 0.0,
    scaling:        Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap:        Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # fa3.py convention: inputs arrive as (B, N, T, E), transpose to (B, T, N, E)
    query = query.transpose(1, 2)
    key   = key.transpose(1, 2)
    value = value.transpose(1, 2)
    seq_len = query.shape[1]

    kwargs.pop("is_causal", None)

    attn_output = _flash_attention_forward(
        query, key, value, attention_mask,
        query_length   = seq_len,
        is_causal      = module.is_causal,
        dropout        = dropout,
        softmax_scale  = scaling,
        sliding_window = sliding_window,
        softcap        = softcap,
        **kwargs,
    )
    return attn_output, None
