# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def qk_norm_and_half_rope_forward_kernel(qkv_ptr,
                                         q_norm_weight_ptr, k_norm_weight_ptr,
                                         freqs_ptr,
                                         qo_ptr, ko_ptr, vo_ptr,
                                         B,
                                         stride,
                                         eps,
                                         H: tl.constexpr,
                                         h: tl.constexpr,
                                         D: tl.constexpr,
                                         d: tl.constexpr,
                                         INTERLEAVED: tl.constexpr,
                                         TRANSPOSED: tl.constexpr,
                                         SILU: tl.constexpr):
    pid = tl.program_id(0)
    L = tl.num_programs(0)
    DD = D * 2

    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D)).to(tl.float32)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    signs = tl.arange(0, 2).to(tl.float32) * 2 - 1

    q_weight_0 = tl.load(q_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    q_weight_1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)
    q_ptr = qkv_ptr
    w = H // h

    # [len, bs, q_head, head_dim] -> [bs, len, q_head, head_dim]
    if INTERLEAVED:
        row_offs = tl.arange(0, H) + tl.arange(0, H) // w * 2
    else:
        row_offs = tl.arange(0, H)

    for i in range(B):
        if TRANSPOSED:
            q0 = tl.load(
                q_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
            q1 = tl.load(
                q_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
        else:
            q0 = tl.load(
                q_ptr + i * L * stride + pid * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
            q1 = tl.load(
                q_ptr + i * L * stride + pid * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
        if SILU:
            q0 = q0 * tl.sigmoid(q0)
            q1 = q1 * tl.sigmoid(q1)
        rms = tl.rsqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)
        q1 *= rms[:, None]
        q1 *= q_weight_1
        tl.store(
            qo_ptr + pid * H * DD + i * L * H * DD + D + DD * tl.arange(0, H)[:,
                                                              None] + tl.arange(
                0, D)[None, :], q1)

        q0 *= rms[:, None]
        q0 *= q_weight_0
        qr = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(q0, (H, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (H, D))
        q0 = q0 * cos + qr * sin
        tl.store(
            qo_ptr + pid * H * DD + i * L * H * DD + DD * tl.arange(0, H)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :], q0)

    k_weight_0 = tl.load(k_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    k_weight_1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)
    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        k_ptr = qkv_ptr + DD * w
    else:
        row_offs = tl.arange(0, h)
        k_ptr = qkv_ptr + DD * H
    for i in range(B):
        if TRANSPOSED:
            k0 = tl.load(
                k_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
            k1 = tl.load(
                k_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
        else:
            k0 = tl.load(
                k_ptr + i * L * stride + pid * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
            k1 = tl.load(
                k_ptr + i * L * stride + pid * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
        if SILU:
            k0 = k0 * tl.sigmoid(k0)
            k1 = k1 * tl.sigmoid(k1)
        rms = tl.rsqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)
        k1 *= rms[:, None]
        k1 *= k_weight_1
        tl.store(
            ko_ptr + pid * h * DD + i * L * h * DD + D + DD * tl.arange(0, h)[:,
                                                              None] + tl.arange(
                0, D)[None, :], k1)

        k0 *= rms[:, None]
        k0 *= k_weight_0
        kr = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(k0, (h, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (h, D))
        k0 = k0 * cos + kr * sin
        tl.store(
            ko_ptr + pid * h * DD + i * L * h * DD + DD * tl.arange(0, h)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :], k0)

    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        v_ptr = qkv_ptr + DD * w + DD
    else:
        row_offs = tl.arange(0, h)
        v_ptr = qkv_ptr + DD * H + DD * h
    for i in range(B):
        if TRANSPOSED:
            v0 = tl.load(
                v_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
            v1 = tl.load(
                v_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
        else:
            v0 = tl.load(
                v_ptr + i * L * stride + pid * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
            v1 = tl.load(
                v_ptr + i * L * stride + pid * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
        if SILU:
            v0 = v0 * tl.sigmoid(v0)
            v1 = v1 * tl.sigmoid(v1)

        tl.store(
            vo_ptr + pid * h * DD + i * L * h * DD + DD * tl.arange(0, h)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :], v0)
        tl.store(
            vo_ptr + pid * h * DD + i * L * h * DD + D + DD * tl.arange(0, h)[:,
                                                              None] + tl.arange(
                0, D)[None, :], v1)


@triton.jit
def compatible_qk_norm_and_half_rop_forward_kernel(qkv_ptr,
                                                   q_norm_weight_ptr,
                                                   k_norm_weight_ptr,
                                                   freqs_ptr,
                                                   qo_ptr, ko_ptr, vo_ptr,
                                                   B,
                                                   stride,
                                                   eps,
                                                   H: tl.constexpr,
                                                   h: tl.constexpr,
                                                   H_p: tl.constexpr,
                                                   h_p: tl.constexpr,
                                                   D: tl.constexpr,
                                                   d: tl.constexpr,
                                                   INTERLEAVED: tl.constexpr,
                                                   TRANSPOSED: tl.constexpr,
                                                   SILU: tl.constexpr):
    pid = tl.program_id(0)
    L = tl.num_programs(0)
    DD = D * 2

    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D)).to(tl.float32)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    signs = tl.arange(0, 2).to(tl.float32) * 2 - 1

    q_weight_0 = tl.load(q_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    q_weight_1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)
    q_ptr = qkv_ptr
    w = H // h  # H =8 h = 2 w=4

    # [len, bs, q_head, head_dim] -> [bs, len, q_head, head_dim]
    if INTERLEAVED:
        # row_offs = tl.arange(0, H) + tl.arange(0, H) // w * 2
        row_offs = tl.arange(0, H_p) + tl.arange(0, H_p) // w * 2
        row_mask = row_offs[:, None] < (H + 2 * h)
    else:
        # row_offs = tl.arange(0, H)
        row_offs = tl.arange(0, H_p)
        row_mask = row_offs[:, None] < H

    for i in range(B):
        if TRANSPOSED:
            q0 = tl.load(
                q_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
            q1 = tl.load(
                q_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
        else:
            q0 = tl.load(
                q_ptr + i * L * stride + pid * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
            q1 = tl.load(
                q_ptr + i * L * stride + pid * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
        if SILU:
            q0 = q0 * tl.sigmoid(q0.to(tl.float32))
            q1 = q1 * tl.sigmoid(q1.to(tl.float32))
        rms = tl.rsqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)
        q1 *= rms[:, None]
        q1 *= q_weight_1
        q_mask = tl.arange(0, H_p)[:, None] < H
        tl.store(
            qo_ptr + pid * H * DD + i * L * H * DD + D + DD * tl.arange(0, H_p)[
                                                              :,
                                                              None] + tl.arange(
                0, D)[None, :], q1, mask=q_mask)

        q0 *= rms[:, None]
        q0 *= q_weight_0
        qr = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(q0, (H_p, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (H_p, D))
        q0 = q0 * cos + qr * sin
        tl.store(
            qo_ptr + pid * H * DD + i * L * H * DD + DD * tl.arange(0, H_p)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :], q0,
            mask=q_mask)

    k_weight_0 = tl.load(k_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    k_weight_1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)
    if INTERLEAVED:
        # row_offs = tl.arange(0, h) * (w + 2)
        row_offs = tl.arange(0, h_p) * (w + 2)
        row_mask = row_offs[:, None] < (h * (w + 2))
        k_ptr = qkv_ptr + DD * w
    else:
        # row_offs = tl.arange(0, h)
        row_offs = tl.arange(0, h_p)
        row_mask = tl.arange(0, h_p)[:, None] < h
        k_ptr = qkv_ptr + DD * H

    for i in range(B):
        if TRANSPOSED:
            k0 = tl.load(
                k_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
            k1 = tl.load(
                k_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
        else:
            k0 = tl.load(
                k_ptr + i * L * stride + pid * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
            k1 = tl.load(
                k_ptr + i * L * stride + pid * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)

        if SILU:
            k0 = k0 * tl.sigmoid(k0)
            k1 = k1 * tl.sigmoid(k1)
        rms = tl.rsqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)
        k1 *= rms[:, None]
        k1 *= k_weight_1
        k_mask = tl.arange(0, h_p)[:, None] < h
        tl.store(
            ko_ptr + pid * h * DD + i * L * h * DD + D + DD * tl.arange(0, h_p)[
                                                              :,
                                                              None] + tl.arange(
                0, D)[None, :], k1,
            mask=k_mask
        )

        k0 *= rms[:, None]
        k0 *= k_weight_0
        kr = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(k0, (h_p, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (h_p, D))
        k0 = k0 * cos + kr * sin
        tl.store(
            ko_ptr + pid * h * DD + i * L * h * DD + DD * tl.arange(0, h_p)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :], k0,
            mask=k_mask
        )

    if INTERLEAVED:
        # row_offs = tl.arange(0, h) * (w + 2)
        row_offs = tl.arange(0, h_p) * (w + 2)
        row_mask = row_offs[:, None] < (h * (w + 2))
        v_ptr = qkv_ptr + DD * w + DD
    else:
        # row_offs = tl.arange(0, h)
        row_offs = tl.arange(0, h_p)
        row_mask = tl.arange(0, h_p)[:, None] < h
        v_ptr = qkv_ptr + DD * H + DD * h

    for i in range(B):
        if TRANSPOSED:
            v0 = tl.load(
                v_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
            v1 = tl.load(
                v_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
        else:
            v0 = tl.load(
                v_ptr + i * L * stride + pid * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
            v1 = tl.load(
                v_ptr + i * L * stride + pid * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
        if SILU:
            v0 = v0 * tl.sigmoid(v0)
            v1 = v1 * tl.sigmoid(v1)

        v_mask = tl.arange(0, h_p)[:, None] < h
        tl.store(
            vo_ptr + pid * h * DD + i * L * h * DD + DD * tl.arange(0, h_p)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :], v0,
            mask=v_mask)
        tl.store(
            vo_ptr + pid * h * DD + i * L * h * DD + D + DD * tl.arange(0, h_p)[
                                                              :,
                                                              None] + tl.arange(
                0, D)[None, :], v1, mask=v_mask)


def triton_qk_norm_and_half_rope_forward(qkv, q_norm_weight, k_norm_weight,
                                         freqs, H=16, h=4, eps=1e-6,
                                         interleaved=False, transposed=False,
                                         silu=False
                                         ):
    """
    split qkv to q/k/v, apply qk norm and half rope to q/k,
        transpose q/k/v to flash-attention layout
    Args:
        qkv: QKV tensor with size of [S, B, dim], heads are interleaved
        q_norm_weight: rms norm weight for query
        k_norm_weight: rms norm weight for key
        freqs: Freqs tensor based on half dim.
        H: Number of attention heads.
        h: Number of key/value heads.
        eps: epsilon value for L2 normalization.
        interleaved: whether head of qkv is interleaved,
            interleaved: [q...qkvq...qkv]
            non-interleaved: [q...qk...kv...v]
        transposed: whether qkv is tranposed
            transposed: [S, B, dim]
            non-transposed: [B, S, dim]
        silu: apply silu on qkv before qk norm and rope
    Returns:
        - qo: shape [B, S, H, head_dim]
        - ko: shape [B, S, h, head_dim]
        - vo: shape [B, S, h, head_dim]
    """
    assert qkv.is_contiguous() and freqs.is_contiguous()
    assert k_norm_weight.is_contiguous() and q_norm_weight.is_contiguous()
    if transposed:
        L, B, Dim = qkv.shape
    else:
        B, L, Dim = qkv.shape
    stride = qkv.stride(1)  # qkv may be a slice of a tensor
    D = k_norm_weight.size(0)
    # tp = (H + 2 * h) * D // Dim
    # if tp > 1:
    #     H = H // tp
    #     h = h // tp
    # D = Dim // (H + 2 * h)  # error with tp
    # assert freqs.size(0) == L and freqs.size(
    #     -1) == D // 2, f'{freqs.shape=} {L=} {D=}'
    # Cat freqs to itself to get last dimension as D//2
    # Ensures freqs' last dimension doubles to match q/k half-dim expectations
    dtype = qkv.dtype
    device = qkv.device
    qo = torch.empty((B, L, H, D), dtype=dtype, device=device)
    ko = torch.empty((B, L, h, D), dtype=dtype, device=device)
    vo = torch.empty((B, L, h, D), dtype=dtype, device=device)

    num_stages = 5
    num_warps = 2
    grid = (L,)

    H_p = triton.next_power_of_2(H)
    h_p = triton.next_power_of_2(h)

    if H_p == H and h_p == h:
        qk_norm_and_half_rope_forward_kernel[grid](
            qkv,
            q_norm_weight, k_norm_weight,
            freqs,
            qo, ko, vo,
            B,
            stride,
            eps,
            H,
            h,
            D // 2,
            D // 4,
            interleaved,
            transposed,
            silu,
            num_stages=num_stages,
            num_warps=num_warps
        )
    else:
        compatible_qk_norm_and_half_rop_forward_kernel[grid](
            qkv,
            q_norm_weight, k_norm_weight,
            freqs,
            qo, ko, vo,
            B,
            stride,
            eps,
            H,
            h,
            H_p,
            h_p,
            D // 2,
            D // 4,
            interleaved,
            transposed,
            silu,
            num_stages=num_stages,
            num_warps=num_warps
        )
    return qo, ko, vo


@triton.jit
def qk_norm_and_half_rope_backward_kernel(gq_ptr, gk_ptr, gv_ptr,
                                          qkv_ptr,
                                          q_norm_weight_ptr,
                                          k_norm_weight_ptr,
                                          freqs_ptr,
                                          dqkv_ptr,
                                          dqw_ptr,
                                          dkw_ptr,
                                          B,
                                          stride,
                                          grad_stride,
                                          eps,
                                          H: tl.constexpr,
                                          h: tl.constexpr,
                                          D: tl.constexpr,
                                          d: tl.constexpr,
                                          INTERLEAVED: tl.constexpr,
                                          TRANSPOSED: tl.constexpr,
                                          SILU: tl.constexpr
                                          ):
    pid = tl.program_id(0)
    L = tl.num_programs(0)
    DD = 2 * D
    w = H // h

    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D)).to(tl.float32)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    signs = -tl.arange(0, 2).to(tl.float32) * 2 + 1

    q_w0 = tl.load(q_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    q_w1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)

    dqw_0 = tl.zeros((D,), dtype=tl.float32)
    dqw_1 = tl.zeros((D,), dtype=tl.float32)
    q_ptr = qkv_ptr
    dq_ptr = dqkv_ptr
    # [bs, len, q_head, head_dim] -> [len, bs, q_head, head_dim]
    if INTERLEAVED:
        row_offs = tl.arange(0, H) + tl.arange(0, H) // w * 2
    else:
        row_offs = tl.arange(0, H)

    for i in range(B):
        gq_0 = tl.load(
            gq_ptr + i * L * H * DD + pid * H * DD + DD * tl.arange(0, H)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :]).to(
            tl.float32)
        gq_1 = tl.load(
            gq_ptr + i * L * H * DD + pid * H * DD + D + DD * tl.arange(0, H)[:,
                                                              None] + tl.arange(
                0, D)[None, :]).to(tl.float32)

        gq_r = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(gq_0, (H, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (H, D))
        gq_0 = gq_0 * cos + gq_r * sin

        if TRANSPOSED:
            q0 = tl.load(
                q_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
            q1 = tl.load(
                q_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
        else:
            q0 = tl.load(
                q_ptr + pid * stride + i * L * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
            q1 = tl.load(
                q_ptr + pid * stride + i * L * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)

        if SILU:
            s0 = tl.sigmoid(q0)
            s1 = tl.sigmoid(q1)
            q_0 = q0 * s0
            q_1 = q1 * s1

            r = tl.rsqrt(
                (tl.sum(q_0 * q_0, 1) + tl.sum(q_1 * q_1, 1)) / DD + eps)[:,
                None]

            dqw_0 += tl.sum(q_0 * gq_0 * r, 0)
            dqw_1 += tl.sum(q_1 * gq_1 * r, 0)

            s = tl.sum(q_0 * gq_0 * q_w0, 1) + tl.sum(q_1 * gq_1 * q_w1, 1)

            dq_0 = r * gq_0 * q_w0 - r * r * r / DD * q_0 * s[:, None]
            dq_1 = r * gq_1 * q_w1 - r * r * r / DD * q_1 * s[:, None]

            dq_0 = dq_0 * s0 * (1 + q0 * (1 - s0))
            dq_1 = dq_1 * s1 * (1 + q1 * (1 - s1))

        else:
            r = tl.rsqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)[
                :, None]

            dqw_0 += tl.sum(q0 * gq_0 * r, 0)
            dqw_1 += tl.sum(q1 * gq_1 * r, 0)

            s = tl.sum(q0 * gq_0 * q_w0, 1) + tl.sum(q1 * gq_1 * q_w1, 1)

            dq_0 = r * gq_0 * q_w0 - r * r * r / DD * q0 * s[:, None]
            dq_1 = r * gq_1 * q_w1 - r * r * r / DD * q1 * s[:, None]

        if TRANSPOSED:
            tl.store(
                dq_ptr + pid * B * grad_stride + i * grad_stride + DD * row_offs[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :], dq_0)
            tl.store(
                dq_ptr + pid * B * grad_stride + i * grad_stride + D + DD * row_offs[
                                                                            :,
                                                                            None] + tl.arange(
                    0, D)[None, :], dq_1)
        else:
            tl.store(
                dq_ptr + pid * grad_stride + i * L * grad_stride + DD * row_offs[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :], dq_0)
            tl.store(
                dq_ptr + pid * grad_stride + i * L * grad_stride + D + DD * row_offs[
                                                                            :,
                                                                            None] + tl.arange(
                    0, D)[None, :], dq_1)

    tl.store(dqw_ptr + pid * D * 2 + tl.arange(0, D), dqw_0)
    tl.store(dqw_ptr + pid * D * 2 + D + tl.arange(0, D), dqw_1)

    k_w0 = tl.load(k_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    k_w1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)

    dkw_0 = tl.zeros((D,), dtype=tl.float32)
    dkw_1 = tl.zeros((D,), dtype=tl.float32)
    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        k_ptr = qkv_ptr + DD * w
        dk_ptr = dqkv_ptr + DD * w
    else:
        row_offs = tl.arange(0, h)
        k_ptr = qkv_ptr + DD * H
        dk_ptr = dqkv_ptr + DD * H
    # [bs, len, k_head, head_dim] -> [len, bs, k_head, head_dim]
    for i in range(B):
        gk_0 = tl.load(
            gk_ptr + i * L * h * DD + pid * h * DD + DD * tl.arange(0, h)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :]).to(
            tl.float32)
        gk_1 = tl.load(
            gk_ptr + i * L * h * DD + pid * h * DD + D + DD * tl.arange(0, h)[:,
                                                              None] + tl.arange(
                0, D)[None, :]).to(tl.float32)

        gk_r = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(gk_0, (h, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (h, D))
        gk_0 = gk_0 * cos + gk_r * sin

        if TRANSPOSED:
            k0 = tl.load(
                k_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
            k1 = tl.load(
                k_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
        else:
            k0 = tl.load(
                k_ptr + pid * stride + i * L * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)
            k1 = tl.load(
                k_ptr + pid * stride + i * L * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :]).to(tl.float32)

        if SILU:

            s0 = tl.sigmoid(k0)
            s1 = tl.sigmoid(k1)
            k_0 = k0 * s0
            k_1 = k1 * s1

            r = tl.rsqrt(
                (tl.sum(k_0 * k_0, 1) + tl.sum(k_1 * k_1, 1)) / DD + eps)[:,
                None]

            dkw_0 += tl.sum(k_0 * gk_0 * r, 0)
            dkw_1 += tl.sum(k_1 * gk_1 * r, 0)

            s = tl.sum(k_0 * gk_0 * k_w0, 1) + tl.sum(k_1 * gk_1 * k_w1, 1)

            dk_0 = r * gk_0 * k_w0 - r * r * r / DD * k_0 * s[:, None]
            dk_1 = r * gk_1 * k_w1 - r * r * r / DD * k_1 * s[:, None]

            dk_0 = dk_0 * s0 * (1 + k0 * (1 - s0))
            dk_1 = dk_1 * s1 * (1 + k1 * (1 - s1))

        else:
            r = tl.rsqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)[
                :, None]

            dkw_0 += tl.sum(k0 * gk_0 * r, 0)
            dkw_1 += tl.sum(k1 * gk_1 * r, 0)

            s = tl.sum(k0 * gk_0 * k_w0, 1) + tl.sum(k1 * gk_1 * k_w1, 1)

            dk_0 = r * gk_0 * k_w0 - r * r * r / DD * k0 * s[:, None]
            dk_1 = r * gk_1 * k_w1 - r * r * r / DD * k1 * s[:, None]

        if TRANSPOSED:
            tl.store(
                dk_ptr + pid * B * grad_stride + i * grad_stride + DD * row_offs[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :], dk_0)
            tl.store(
                dk_ptr + pid * B * grad_stride + i * grad_stride + D + DD * row_offs[
                                                                            :,
                                                                            None] + tl.arange(
                    0, D)[None, :], dk_1)
        else:
            tl.store(
                dk_ptr + pid * grad_stride + i * L * grad_stride + DD * row_offs[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :], dk_0)
            tl.store(
                dk_ptr + pid * grad_stride + i * L * grad_stride + D + DD * row_offs[
                                                                            :,
                                                                            None] + tl.arange(
                    0, D)[None, :], dk_1)
    tl.store(dkw_ptr + pid * D * 2 + tl.arange(0, D), dkw_0)
    tl.store(dkw_ptr + pid * D * 2 + D + tl.arange(0, D), dkw_1)

    # [bs, len, k_head, head_dim] -> [len, bs, k_head + 2 * kv_head, head_dim]
    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        v_ptr = qkv_ptr + DD * w + DD
        dv_ptr = dqkv_ptr + DD * w + DD
    else:
        row_offs = tl.arange(0, h)
        v_ptr = qkv_ptr + DD * H + DD * h
        dv_ptr = dqkv_ptr + DD * H + DD * h
    for i in range(B):

        gv_0 = tl.load(
            gv_ptr + i * L * h * DD + pid * h * DD + DD * tl.arange(0, h)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :]).to(
            tl.float32)
        gv_1 = tl.load(
            gv_ptr + i * L * h * DD + pid * h * DD + D + DD * tl.arange(0, h)[:,
                                                              None] + tl.arange(
                0, D)[None, :]).to(tl.float32)

        if SILU:
            if TRANSPOSED:
                v0 = tl.load(
                    v_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                                 None] + tl.arange(
                        0, D)[None, :]).to(tl.float32)
                v1 = tl.load(
                    v_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                     None] + tl.arange(
                        0, D)[None, :]).to(tl.float32)
            else:
                v0 = tl.load(
                    v_ptr + i * L * stride + pid * stride + DD * row_offs[:,
                                                                 None] + tl.arange(
                        0, D)[None, :]).to(tl.float32)
                v1 = tl.load(
                    v_ptr + i * L * stride + pid * stride + D + DD * row_offs[:,
                                                                     None] + tl.arange(
                        0, D)[None, :]).to(tl.float32)

            s0 = tl.sigmoid(v0)
            s1 = tl.sigmoid(v1)
            dv_0 = gv_0 * s0 * (1 + v0 * (1 - s0))
            dv_1 = gv_1 * s1 * (1 + v1 * (1 - s1))
        else:
            dv_0 = gv_0
            dv_1 = gv_1

        if TRANSPOSED:
            tl.store(
                dv_ptr + pid * B * grad_stride + i * grad_stride + DD * row_offs[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :], dv_0)
            tl.store(
                dv_ptr + pid * B * grad_stride + i * grad_stride + D + DD * row_offs[
                                                                            :,
                                                                            None] + tl.arange(
                    0, D)[None, :], dv_1)
        else:
            tl.store(
                dv_ptr + pid * grad_stride + i * L * grad_stride + DD * row_offs[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :], dv_0)
            tl.store(
                dv_ptr + pid * grad_stride + i * L * grad_stride + D + DD * row_offs[
                                                                            :,
                                                                            None] + tl.arange(
                    0, D)[None, :], dv_1)


@triton.jit
def compatible_qk_norm_and_half_rope_backward_kernel(gq_ptr, gk_ptr, gv_ptr,
                                                     qkv_ptr,
                                                     q_norm_weight_ptr,
                                                     k_norm_weight_ptr,
                                                     freqs_ptr,
                                                     dqkv_ptr,
                                                     dqw_ptr, dkw_ptr,
                                                     B,
                                                     stride,
                                                     grad_stride,
                                                     eps,
                                                     H: tl.constexpr,
                                                     h: tl.constexpr,
                                                     H_p: tl.constexpr,
                                                     h_p: tl.constexpr,
                                                     D: tl.constexpr,
                                                     d: tl.constexpr,
                                                     INTERLEAVED: tl.constexpr,
                                                     TRANSPOSED: tl.constexpr,
                                                     SILU: tl.constexpr
                                                     ):
    pid = tl.program_id(0)
    L = tl.num_programs(0)
    DD = 2 * D
    w = H // h

    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D)).to(tl.float32)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    signs = -tl.arange(0, 2).to(tl.float32) * 2 + 1

    q_w0 = tl.load(q_norm_weight_ptr + tl.arange(0, D))
    q_w1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D))

    dqw_0 = tl.zeros((D,), dtype=tl.float32)
    dqw_1 = tl.zeros((D,), dtype=tl.float32)
    q_ptr = qkv_ptr
    dq_ptr = dqkv_ptr
    # [bs, len, q_head, head_dim] -> [len, bs, q_head, head_dim]
    if INTERLEAVED:
        # row_offs = tl.arange(0, H) + tl.arange(0, H) // w * 2
        row_offs = tl.arange(0, H_p) + tl.arange(0, H_p) // w * 2
        row_mask = row_offs[:, None] < (H + 2 * h)
    else:
        # row_offs = tl.arange(0, H)
        row_offs = tl.arange(0, H_p)
        row_mask = row_offs[:, None] < H

    for i in range(B):
        gq_0 = tl.load(
            gq_ptr + i * L * H * DD + pid * H * DD + DD * tl.arange(0, H_p)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :]
            , mask=tl.arange(0, H_p)[:, None] < H
        ).to(tl.float32)
        gq_1 = tl.load(
            gq_ptr + i * L * H * DD + pid * H * DD + D + DD * tl.arange(0, H_p)[
                                                              :,
                                                              None] + tl.arange(
                0, D)[None, :]
            , mask=tl.arange(0, H_p)[:, None] < H
        ).to(tl.float32)

        gq_r = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(gq_0, (H_p, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (H_p, D))
        gq_0 = gq_0 * cos + gq_r * sin

        if TRANSPOSED:
            # q0 = tl.load(q_ptr + pid * B * stride + i * stride + DD * row_offs[:,None] + tl.arange(0, D)[None, :])
            # q1 = tl.load(q_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,None] + tl.arange(0, D)[None, :])
            q0 = tl.load(
                q_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
            q1 = tl.load(
                q_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)

        else:
            # q0 = tl.load(q_ptr + pid * stride + i * L * stride + DD * row_offs[:,None] + tl.arange(0, D)[None, :])
            # q1 = tl.load(q_ptr + pid * stride + i * L * stride + D + DD * row_offs[:,None] + tl.arange(0, D)[None, :])
            q0 = tl.load(
                q_ptr + pid * stride + i * L * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
            q1 = tl.load(
                q_ptr + pid * stride + i * L * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)

        if SILU:
            s0 = tl.sigmoid(q0)
            s1 = tl.sigmoid(q1)
            q_0 = q0 * s0
            q_1 = q1 * s1

            r = tl.rsqrt(
                (tl.sum(q_0 * q_0, 1) + tl.sum(q_1 * q_1, 1)) / DD + eps)[:,
                None]

            dqw_0 += tl.sum(q_0 * gq_0 * r, 0)
            dqw_1 += tl.sum(q_1 * gq_1 * r, 0)

            s = tl.sum(q_0 * gq_0 * q_w0, 1) + tl.sum(q_1 * gq_1 * q_w1, 1)

            dq_0 = r * gq_0 * q_w0 - r * r * r / DD * q_0 * s[:, None]
            dq_1 = r * gq_1 * q_w1 - r * r * r / DD * q_1 * s[:, None]

            dq_0 = dq_0 * s0 * (1 + q0 * (1 - s0))
            dq_1 = dq_1 * s1 * (1 + q1 * (1 - s1))

        else:
            r = tl.rsqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)[
                :, None]

            dqw_0 += tl.sum(q0 * gq_0 * r, 0)
            dqw_1 += tl.sum(q1 * gq_1 * r, 0)

            s = tl.sum(q0 * gq_0 * q_w0, 1) + tl.sum(q1 * gq_1 * q_w1, 1)

            dq_0 = r * gq_0 * q_w0 - r * r * r / DD * q0 * s[:, None]
            dq_1 = r * gq_1 * q_w1 - r * r * r / DD * q1 * s[:, None]

        if TRANSPOSED:
            # tl.store(dq_ptr + pid * B * grad_stride + i * grad_stride + DD * row_offs[:,None] + tl.arange(0, D)[None, :], dq_0)
            # tl.store(dq_ptr + pid * B * grad_stride + i * grad_stride + D + DD * row_offs[:,None] + tl.arange(0, D)[None, :], dq_1)
            tl.store(
                dq_ptr + pid * B * grad_stride + i * grad_stride + DD * row_offs[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :], dq_0, mask=row_mask)
            tl.store(
                dq_ptr + pid * B * grad_stride + i * grad_stride + D + DD * row_offs[
                                                                            :,
                                                                            None] + tl.arange(
                    0, D)[None, :], dq_1, mask=row_mask)

        else:
            # tl.store(dq_ptr + pid * grad_stride + i * L * grad_stride + DD * row_offs[:,None] + tl.arange(0, D)[None, :], dq_0)
            # tl.store(dq_ptr + pid * grad_stride + i * L * grad_stride + D + DD * row_offs[:,None] + tl.arange(0, D)[None, :], dq_1)
            tl.store(
                dq_ptr + pid * grad_stride + i * L * grad_stride + DD * row_offs[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :], dq_0, mask=row_mask)
            tl.store(
                dq_ptr + pid * grad_stride + i * L * grad_stride + D + DD * row_offs[
                                                                            :,
                                                                            None] + tl.arange(
                    0, D)[None, :], dq_1, mask=row_mask)

    tl.store(dqw_ptr + pid * D * 2 + tl.arange(0, D), dqw_0)
    tl.store(dqw_ptr + pid * D * 2 + D + tl.arange(0, D), dqw_1)

    k_w0 = tl.load(k_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    k_w1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)

    dkw_0 = tl.zeros((D,), dtype=tl.float32)
    dkw_1 = tl.zeros((D,), dtype=tl.float32)
    if INTERLEAVED:
        # row_offs = tl.arange(0, h) * (w + 2)
        row_offs = tl.arange(0, h_p) * (w + 2)
        row_mask = row_offs[:, None] < (h * (w + 2))
        k_ptr = qkv_ptr + DD * w
        dk_ptr = dqkv_ptr + DD * w
    else:
        # row_offs = tl.arange(0, h)
        row_offs = tl.arange(0, h_p)
        row_mask = row_offs[:, None] < h
        k_ptr = qkv_ptr + DD * H
        dk_ptr = dqkv_ptr + DD * H
    # [bs, len, k_head, head_dim] -> [len, bs, k_head, head_dim]
    for i in range(B):
        gk_0 = tl.load(
            gk_ptr + i * L * h * DD + pid * h * DD + DD * tl.arange(0, h_p)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :],
            mask=tl.arange(0, h_p)[:, None] < h
        ).to(tl.float32)
        gk_1 = tl.load(
            gk_ptr + i * L * h * DD + pid * h * DD + D + DD * tl.arange(0, h_p)[
                                                              :,
                                                              None] + tl.arange(
                0, D)[None, :],
            mask=tl.arange(0, h_p)[:, None] < h
        ).to(tl.float32)

        gk_r = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(gk_0, (h_p, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (h_p, D))
        gk_0 = gk_0 * cos + gk_r * sin

        if TRANSPOSED:
            k0 = tl.load(
                k_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
            k1 = tl.load(
                k_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
        else:
            k0 = tl.load(
                k_ptr + pid * stride + i * L * stride + DD * row_offs[:,
                                                             None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)
            k1 = tl.load(
                k_ptr + pid * stride + i * L * stride + D + DD * row_offs[:,
                                                                 None] + tl.arange(
                    0, D)[None, :], mask=row_mask).to(tl.float32)

        if SILU:

            s0 = tl.sigmoid(k0)
            s1 = tl.sigmoid(k1)
            k_0 = k0 * s0
            k_1 = k1 * s1

            r = tl.rsqrt(
                (tl.sum(k_0 * k_0, 1) + tl.sum(k_1 * k_1, 1)) / DD + eps)[:,
                None]

            dkw_0 += tl.sum(k_0 * gk_0 * r, 0)
            dkw_1 += tl.sum(k_1 * gk_1 * r, 0)

            s = tl.sum(k_0 * gk_0 * k_w0, 1) + tl.sum(k_1 * gk_1 * k_w1, 1)

            dk_0 = r * gk_0 * k_w0 - r * r * r / DD * k_0 * s[:, None]
            dk_1 = r * gk_1 * k_w1 - r * r * r / DD * k_1 * s[:, None]

            dk_0 = dk_0 * s0 * (1 + k0 * (1 - s0))
            dk_1 = dk_1 * s1 * (1 + k1 * (1 - s1))

        else:
            r = tl.rsqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)[
                :, None]

            dkw_0 += tl.sum(k0 * gk_0 * r, 0)
            dkw_1 += tl.sum(k1 * gk_1 * r, 0)

            s = tl.sum(k0 * gk_0 * k_w0, 1) + tl.sum(k1 * gk_1 * k_w1, 1)

            dk_0 = r * gk_0 * k_w0 - r * r * r / DD * k0 * s[:, None]
            dk_1 = r * gk_1 * k_w1 - r * r * r / DD * k1 * s[:, None]

        if TRANSPOSED:
            tl.store(
                dk_ptr + pid * B * grad_stride + i * grad_stride + DD * row_offs[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :], dk_0, mask=row_mask)
            tl.store(
                dk_ptr + pid * B * grad_stride + i * grad_stride + D + DD * row_offs[
                                                                            :,
                                                                            None] + tl.arange(
                    0, D)[None, :], dk_1, mask=row_mask)
        else:
            tl.store(
                dk_ptr + pid * grad_stride + i * L * grad_stride + DD * row_offs[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :], dk_0, mask=row_mask)
            tl.store(
                dk_ptr + pid * grad_stride + i * L * grad_stride + D + DD * row_offs[
                                                                            :,
                                                                            None] + tl.arange(
                    0, D)[None, :], dk_1, mask=row_mask)

    tl.store(dkw_ptr + pid * D * 2 + tl.arange(0, D), dkw_0)
    tl.store(dkw_ptr + pid * D * 2 + D + tl.arange(0, D), dkw_1)

    # [bs, len, k_head, head_dim] -> [len, bs, k_head + 2 * kv_head, head_dim]
    if INTERLEAVED:
        # row_offs = tl.arange(0, h) * (w + 2)
        row_offs = tl.arange(0, h_p) * (w + 2)
        row_mask = row_offs[:, None] < (h * (w + 2))
        v_ptr = qkv_ptr + DD * w + DD
        dv_ptr = dqkv_ptr + DD * w + DD
    else:
        # row_offs = tl.arange(0, h)
        row_offs = tl.arange(0, h_p)
        row_mask = row_offs[:, None] < h
        v_ptr = qkv_ptr + DD * H + DD * h
        dv_ptr = dqkv_ptr + DD * H + DD * h
    for i in range(B):

        gv_0 = tl.load(
            gv_ptr + i * L * h * DD + pid * h * DD + DD * tl.arange(0, h_p)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :],
            mask=tl.arange(0, h_p)[:, None] < h).to(tl.float32)
        gv_1 = tl.load(
            gv_ptr + i * L * h * DD + pid * h * DD + D + DD * tl.arange(0, h_p)[
                                                              :,
                                                              None] + tl.arange(
                0, D)[None, :], mask=tl.arange(0, h_p)[:, None] < h).to(
            tl.float32)

        if SILU:
            if TRANSPOSED:
                v0 = tl.load(
                    v_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                                 None] + tl.arange(
                        0, D)[None, :], mask=row_mask).to(tl.float32)
                v1 = tl.load(
                    v_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                     None] + tl.arange(
                        0, D)[None, :], mask=row_mask).to(tl.float32)
            else:
                v0 = tl.load(
                    v_ptr + i * L * stride + pid * stride + DD * row_offs[:,
                                                                 None] + tl.arange(
                        0, D)[None, :], mask=row_mask).to(tl.float32)
                v1 = tl.load(
                    v_ptr + i * L * stride + pid * stride + D + DD * row_offs[:,
                                                                     None] + tl.arange(
                        0, D)[None, :], mask=row_mask).to(tl.float32)

            s0 = tl.sigmoid(v0)
            s1 = tl.sigmoid(v1)
            dv_0 = gv_0 * s0 * (1 + v0 * (1 - s0))
            dv_1 = gv_1 * s1 * (1 + v1 * (1 - s1))
        else:
            dv_0 = gv_0
            dv_1 = gv_1

        if TRANSPOSED:
            tl.store(
                dv_ptr + pid * B * grad_stride + i * grad_stride + DD * row_offs[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :], dv_0, mask=row_mask)
            tl.store(
                dv_ptr + pid * B * grad_stride + i * grad_stride + D + DD * row_offs[
                                                                            :,
                                                                            None] + tl.arange(
                    0, D)[None, :], dv_1, mask=row_mask)
        else:
            tl.store(
                dv_ptr + pid * grad_stride + i * L * grad_stride + DD * row_offs[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :], dv_0, mask=row_mask)
            tl.store(
                dv_ptr + pid * grad_stride + i * L * grad_stride + D + DD * row_offs[
                                                                            :,
                                                                            None] + tl.arange(
                    0, D)[None, :], dv_1, mask=row_mask)


def triton_qk_norm_and_half_rope_backward(gq, gk, gv, qkv, q_norm_weight,
                                          k_norm_weight, freqs, eps=1e-6,
                                          interleaved=True, transposed=True,
                                          silu=False):
    """
    backward kernel of triton_qk_norm_and_half_rope_forward
    Args:
        gq: gradient of qo, [len, bs, q_head, head_dim]
        gk: gradient of ko, [len, bs, q_head, head_dim]
        gv: gradient of vo, [len, bs, q_head, head_dim]
        qkv: input qkv
        q_norm_weight: rms norm weight for query
        k_norm_weight: rms norm weight for key
        freqs: Freqs tensor based on half dim.
        eps: epsilon value for L2 normalization.
        interleaved: whether head of qkv is interleaved,
            interleaved: [q...qkvq...qkv]
            non-interleaved: [q...qk...kv...v]
        transposed: whether qkv is tranposed
            transposed: [S, B, dim]
            non-transposed: [B, S, dim]
        silu: whether silu is applied to qkv

    Returns:
        - dqkv: gradient of qkv
        - dqw: gradient of q_norm_weight
        - dkw: gradient of k_norm_weight
    """
    assert gq.is_contiguous() and gk.is_contiguous() and gv.is_contiguous()
    B, L, H, D = gq.shape
    h = gk.shape[2]
    stride = qkv.stride(1)

    dtype = gq.dtype
    device = gq.device
    if transposed:
        dqkv = torch.empty((L, B, (H + 2 * h) * D), dtype=dtype, device=device)
    else:
        dqkv = torch.empty((B, L, (H + 2 * h) * D), dtype=dtype, device=device)
    grad_stride = dqkv.stride(1)  # for potential fused kernel

    tmp_dqw = torch.empty((L, D), dtype=torch.float32, device=device)
    tmp_dkw = torch.empty((L, D), dtype=torch.float32, device=device)

    H_p = triton.next_power_of_2(H)
    h_p = triton.next_power_of_2(h)

    num_stages = 5
    num_warps = 1
    grid = (L,)
    if H == H_p and h == h_p:
        qk_norm_and_half_rope_backward_kernel[grid](
            gq,
            gk,
            gv,
            qkv,
            q_norm_weight,
            k_norm_weight,
            freqs,
            dqkv,
            tmp_dqw,
            tmp_dkw,
            B,
            stride,
            grad_stride,
            eps,
            H,
            h,
            D // 2,
            D // 4,
            interleaved,
            transposed,
            silu,
            num_stages=num_stages,
            num_warps=num_warps
        )

    else:
        compatible_qk_norm_and_half_rope_backward_kernel[grid](
            gq, gk, gv,
            qkv,
            q_norm_weight, k_norm_weight,
            freqs,
            dqkv,
            tmp_dqw, tmp_dkw,
            B,
            stride,
            grad_stride,
            eps,
            H,
            h,
            H_p,
            h_p,
            D // 2,
            D // 4,
            interleaved,
            transposed,
            silu,
            num_stages=num_stages,
            num_warps=num_warps
        )
    dqw = tmp_dqw.sum(0)
    dkw = tmp_dkw.sum(0)
    return dqkv, dqw, dkw
