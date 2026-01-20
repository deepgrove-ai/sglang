"""
Triton-based Ternary MoE Kernel

This kernel extends the existing fused_moe_kernel to support ternary (2-bit) quantized weights.
Key advantages over CUDA implementation:
1. Dimension-agnostic (works for any model size)
2. Leverages Triton's autotuning
3. Integrates cleanly with SGLang infrastructure

Ternary weight encoding:
- Values: {-1, 0, +1}
- Encoding: {0b00, 0b01, 0b10} (2 bits per weight)
- Packing: 4 weights per byte (K dimension packed by factor of 4)

Weight layout: [num_experts, N, K//4] as uint8
Scale layout: [num_experts, K] as float32 (per-expert, per-K scale)
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def unpack_ternary_4(packed_byte: tl.tensor, offset: tl.constexpr) -> tl.tensor:
    """Unpack 4 ternary values from a packed byte.
    
    Each byte contains 4 x 2-bit values: [v0, v1, v2, v3]
    Encoded as: 0b_v3v3_v2v2_v1v1_v0v0
    
    Returns values in {-1, 0, +1} as int8.
    """
    shift = offset * 2
    val = (packed_byte >> shift) & 0x3  # Extract 2 bits
    # Convert 0->-1, 1->0, 2->+1
    return (val.to(tl.int8) - 1)


@triton.jit
def fused_moe_ternary_kernel(
    # Pointers to matrices
    a_ptr,           # Input activations [M, K] in BF16
    b_ptr,           # Packed ternary weights [E, N, K//4] as uint8
    c_ptr,           # Output [M, topk, N] in BF16
    alpha_ptr,       # Per-expert per-K scales [E, K] in float32
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,               # Output dimension
    K,               # Input dimension (unpacked)
    EM,              # num_tokens_post_padded
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,       # Stride for expert dimension
    stride_bk,       # Stride for K//4 dimension
    stride_bn,       # Stride for N dimension
    stride_alphae,   # Stride for alpha expert dimension
    stride_alphak,   # Stride for alpha K dimension
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # Must be multiple of 4 (unpacking granularity)
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
    FP32_ALPHA: tl.constexpr,
):
    """
    Fused MoE kernel with ternary weight support.
    
    Key differences from standard fused_moe_kernel:
    1. Weights are packed (4 values per byte)
    2. Unpacking happens in-kernel
    3. Per-expert, per-K scale is applied in-kernel to dequantize weights
    """
    # Map program ID to output block
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Early exit if beyond padded tokens
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    
    # Load token indices and check validity
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    # Load expert ID for this block
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    
    if off_experts == -1:
        # Write zeros for tokens assigned to non-local experts
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask)
        return

    # Initialize pointers
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    token_idx = offs_token // top_k
    base_a = a_ptr + token_idx[:, None] * stride_am
    alpha_base = alpha_ptr + off_experts * stride_alphae
    
    # Precompute base pointers (A rows and B expert slice) to save inner-loop math.
    b_base = (
        b_ptr
        + off_experts * stride_be
        + offs_bn[None, :] * stride_bn
    )

    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop over K dimension
    #
    # IMPORTANT: Weights are packed 4 values per byte along K. The naive approach
    # (loading one "byte" per K element) causes 4x redundant global loads.
    #
    # We load a [BLOCK_SIZE_K//4, BLOCK_SIZE_N] tile of bytes once, then compute
    # 4 dot products (one per 2-bit lane). This avoids redundant weight loads.
    offs_k_packed = tl.arange(0, BLOCK_SIZE_K // 4)  # [BK/4]

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        k_base = k_start + offs_k_packed * 4  # [BK/4]

        if even_Ks:
            # No K-masking needed (K % BLOCK_SIZE_K == 0)
            b_bytes = tl.load(
                b_base + ((k_start // 4 + offs_k_packed)[:, None]) * stride_bk,
            )

            for lane in tl.static_range(0, 4):
                k_idx = k_base + lane
                alpha_lane = tl.load(alpha_base + k_idx.to(tl.int64) * stride_alphak)
                a_lane = tl.load(
                    base_a + k_idx[None, :] * stride_ak,
                    mask=token_mask[:, None],
                    other=0.0,
                )
                b_lane = ((b_bytes >> (lane * 2)) & 0x3).to(tl.int8) - 1
                if FP32_ALPHA:
                    b_lane = (b_lane.to(tl.float32) * alpha_lane[:, None].to(tl.float32)).to(compute_type)
                else:
                    b_lane = b_lane.to(compute_type) * alpha_lane[:, None].to(compute_type)
                accumulator += tl.dot(a_lane, b_lane)
        else:
            # Masked tail
            k_mask_packed = k_base < K
            b_bytes = tl.load(
                b_base + ((k_start // 4 + offs_k_packed)[:, None]) * stride_bk,
                mask=k_mask_packed[:, None],
                other=0x55,  # encode 0 weight (01 -> 0) for all 2-bit lanes
            )

            for lane in tl.static_range(0, 4):
                k_idx = k_base + lane
                k_mask = k_idx < K
                alpha_lane = tl.load(
                    alpha_base + k_idx.to(tl.int64) * stride_alphak,
                    mask=k_mask,
                    other=0.0,
                )
                a_lane = tl.load(
                    base_a + k_idx[None, :] * stride_ak,
                    mask=token_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                b_lane = ((b_bytes >> (lane * 2)) & 0x3).to(tl.int8) - 1
                if FP32_ALPHA:
                    b_lane = (b_lane.to(tl.float32) * alpha_lane[:, None].to(tl.float32)).to(compute_type)
                else:
                    b_lane = b_lane.to(compute_type) * alpha_lane[:, None].to(compute_type)
                accumulator += tl.dot(a_lane, b_lane)

    # Apply routing weight if needed
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    # Convert and store
    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_ternary_kernel(
    A: torch.Tensor,           # [M, K] BF16 activations
    B_packed: torch.Tensor,    # [E, N, K//4] uint8 packed ternary weights
    alpha: torch.Tensor,       # [E, K] float32 per-expert, per-K scales
    C: torch.Tensor,           # [M, topk, N] BF16 output
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict | None = None,
):
    """
    Invoke the ternary MoE kernel.
    
    Args:
        A: Input activations [M, K]
        B_packed: Packed ternary weights [E, N, K//4] as uint8
        alpha: Per-expert, per-K scales [E, K]
        C: Output buffer [M, topk, N]
        topk_weights: Routing weights [M, topk]
        sorted_token_ids: Sorted token indices
        expert_ids: Expert assignments per block
        num_tokens_post_padded: Total tokens after padding
        mul_routed_weight: Whether to multiply by routing weight
        top_k: Number of experts per token
        config: Triton kernel config
    """
    E, N, K_packed = B_packed.shape
    assert alpha.ndim == 2 and alpha.shape[0] == E, "alpha must have shape [E, K]"
    K = int(alpha.shape[1])
    assert K_packed * 4 >= K, "Packed weights K dimension must cover alpha K"

    if config is None:
        # Heuristic: in the fused_experts_impl pipeline, the first GEMM is top_k>1 (gate_up),
        # the second GEMM uses top_k=1 (down). Use that to pick tuned configs.
        kind = "down" if int(top_k) == 1 else "gate_up"
        config = get_default_ternary_moe_config(kind=kind, m=int(topk_ids.shape[0]))

    even_Ks = (K % int(config["BLOCK_SIZE_K"]) == 0)
    
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16
    
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    
    fused_moe_ternary_kernel[grid](
        A,
        B_packed,
        C,
        alpha,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B_packed.stride(0),
        B_packed.stride(2),
        B_packed.stride(1),
        alpha.stride(0),
        alpha.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        even_Ks=even_Ks,
        **config,
    )


# Default config - should be autotuned per device
DEFAULT_TERNARY_MOE_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,  # Must be multiple of 4
    "GROUP_SIZE_M": 8,
    # Reasonable fallbacks if the caller doesn't set them.
    "num_warps": 4,
    "num_stages": 4,
    # Compute alpha scaling in FP32 for correctness by default.
    "FP32_ALPHA": 1,
}


# Tuned configs for Qwen3-MoE dims on B200 (Jan 2026).
# NOTE: these are heuristics; real optimal settings may differ across GPUs.
_QWEN3_MOE_TERNARY_CONFIG_MAP: dict[str, dict[int, dict]] = {
    # gate_up GEMM: N=2*inter=1536, K=hidden=2048, top_k=8
    "gate_up": {
        2: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 5,
        },
        4: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 64,
            "num_warps": 4,
            "num_stages": 3,
        },
        8: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 5,
        },
    },
    # down GEMM: N=hidden=2048, K=inter=768, top_k=1
    "down": {
        2: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        },
        4: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        },
        8: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 64,
            "num_warps": 4,
            "num_stages": 3,
        },
    },
}


def _bucket_m_for_config(m: int) -> int:
    if m <= 2:
        return 2
    if m <= 4:
        return 4
    return 8


def get_default_ternary_moe_config(kind: str, m: int) -> dict:
    """
    Return a tuned (heuristic) config for `fused_moe_ternary_kernel`.

    Currently tuned for Qwen3-MoE dimensions on B200 and bucketed by M:
      - bucket 2  -> m <= 2
      - bucket 4  -> 2 < m <= 4
      - bucket 8  -> m > 4
    """
    base = dict(DEFAULT_TERNARY_MOE_CONFIG)
    fp32_alpha_env = os.environ.get("SGLANG_TERNARY_MOE_FP32_ALPHA", "1").strip().lower()
    base["FP32_ALPHA"] = 1 if fp32_alpha_env in ("1", "true", "yes", "on") else 0
    bucket = _bucket_m_for_config(int(m))
    cfg = _QWEN3_MOE_TERNARY_CONFIG_MAP.get(kind, {}).get(bucket)
    if cfg is None:
        return base
    base.update(cfg)
    return base


@triton.jit
def fused_moe_ternary_int8_kernel(
    # Pointers to matrices
    a_ptr,           # Input activations [M, K] in INT8 (pre-quantized)
    b_ptr,           # Packed ternary weights [E, N, K//4] as uint8
    c_ptr,           # Output [M, topk, N] in BF16/FP16
    alpha_ptr,       # Per-expert scales [E] in float32
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,               # Output dimension
    K,               # Input dimension (unpacked)
    EM,              # num_tokens_post_padded
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,       # Stride for expert dimension
    stride_bk,       # Stride for K//4 dimension
    stride_bn,       # Stride for N dimension
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # Unpacked K tile (must be multiple of 4)
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """
    INT8 MMA variant of the ternary MoE kernel (designed to hit IMMA/Tensor Cores on B200).

    Assumes activations are already quantized to int8 (e.g. BitNet "pre-scaled activations").
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask)
        return

    expert_alpha = tl.load(alpha_ptr + off_experts).to(tl.float32)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k_packed = tl.arange(0, BLOCK_SIZE_K // 4)  # [BK/4]

    # Treat output rows as flattened [M*topk, N], so we index by offs_token.
    base_a = a_ptr + (offs_token[:, None] // top_k) * stride_am

    acc_i32 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        k_base = k_start + offs_k_packed * 4  # [BK/4]

        if even_Ks:
            b_bytes = tl.load(
                b_ptr
                + off_experts * stride_be
                + offs_bn[:, None] * stride_bn
                + ((k_start // 4 + offs_k_packed)[None, :]) * stride_bk,
            )

            for lane in tl.static_range(0, 4):
                k_idx = k_base + lane
                a_lane = tl.load(
                    base_a + k_idx[None, :] * stride_ak,
                    mask=token_mask[:, None],
                    other=0,
                ).to(tl.int8)
                b_lane = ((b_bytes >> (lane * 2)) & 0x3).to(tl.int8) - 1
                acc_i32 += tl.dot(a_lane, tl.trans(b_lane))
        else:
            k_mask_packed = k_base < K
            b_bytes = tl.load(
                b_ptr
                + off_experts * stride_be
                + offs_bn[:, None] * stride_bn
                + ((k_start // 4 + offs_k_packed)[None, :]) * stride_bk,
                mask=k_mask_packed[None, :],
                other=0x55,  # encode 0 weight (01 -> 0) for all 2-bit lanes
            )

            for lane in tl.static_range(0, 4):
                k_idx = k_base + lane
                k_mask = k_idx < K
                a_lane = tl.load(
                    base_a + k_idx[None, :] * stride_ak,
                    mask=token_mask[:, None] & k_mask[None, :],
                    other=0,
                ).to(tl.int8)
                b_lane = ((b_bytes >> (lane * 2)) & 0x3).to(tl.int8) - 1
                acc_i32 += tl.dot(a_lane, tl.trans(b_lane))

    acc = acc_i32.to(tl.float32) * expert_alpha

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0).to(tl.float32)
        acc = acc * moe_weight[:, None]

    acc = acc.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Default config for INT8 MMA variant (can be tuned separately from BF16 path)
DEFAULT_TERNARY_MOE_INT8_CONFIG = dict(DEFAULT_TERNARY_MOE_CONFIG)

# Tuned (rough) on B200 for the INT8 MMA variant (Jan 2026).
# These differ from the BF16 map (notably: BK must be >=128 for int8 dot, and
# warps/stages choices shift).
_QWEN3_MOE_TERNARY_INT8_CONFIG_MAP: dict[str, dict[int, dict]] = {
    "gate_up": {
        2: {
            "BLOCK_SIZE_M": 16,
            # For very small M (many experts with <=1 token), smaller BN tends to win
            # in end-to-end (launch/occupancy) even if microbench sometimes prefers BN=128.
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        },
        # m==3 is a uniquely "hard" regime (lots of experts with 1 token). Using larger BK
        # reduces unpack overhead on B200.
        3: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 5,
        },
        4: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 64,
            "num_warps": 4,
            "num_stages": 3,
        },
        6: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 64,
            "num_warps": 4,
            "num_stages": 5,
        },
        8: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 64,
            "num_warps": 4,
            "num_stages": 3,
        },
    },
    "down": {
        2: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        },
        3: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 5,
        },
        4: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 64,
            "num_warps": 4,
            "num_stages": 5,
        },
        6: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 64,
            "num_warps": 4,
            "num_stages": 5,
        },
        8: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        },
    },
}


def _bucket_m_for_int8_config(m: int) -> int:
    """
    Bucket M for the INT8 MMA variant.

    We use finer buckets vs BF16 because small-M performance is more sensitive to
    unpack overhead and launch scheduling.
    """
    if m <= 2:
        return 2
    if m <= 3:
        return 3
    if m <= 4:
        return 4
    if m <= 6:
        return 6
    return 8


def get_default_ternary_moe_int8_config(kind: str, m: int) -> dict:
    """
    Return a tuned (heuristic) config for `fused_moe_ternary_int8_kernel`.

    NOTE: currently seeded from the BF16-tuned map; we should retune specifically for INT8.
    """
    base = dict(DEFAULT_TERNARY_MOE_INT8_CONFIG)
    bucket = _bucket_m_for_int8_config(int(m))
    cfg = _QWEN3_MOE_TERNARY_INT8_CONFIG_MAP.get(kind, {}).get(bucket)
    if cfg is None:
        return base
    base.update(cfg)
    return base


def invoke_fused_moe_ternary_int8_kernel(
    A: torch.Tensor,           # [M, K] INT8 activations
    B_packed: torch.Tensor,    # [E, N, K//4] uint8 packed ternary weights
    alpha: torch.Tensor,       # [E] float32 per-expert scales
    C: torch.Tensor,           # [M, topk, N] BF16/FP16 output
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict | None = None,
):
    E, N, K_packed = B_packed.shape
    K = K_packed * 4

    if A.dtype != torch.int8:
        raise TypeError(f"INT8 kernel expects A=int8, got {A.dtype}")

    if config is None:
        kind = "down" if int(top_k) == 1 else "gate_up"
        config = get_default_ternary_moe_int8_config(kind=kind, m=int(topk_ids.shape[0]))

    even_Ks = (K % int(config["BLOCK_SIZE_K"]) == 0)
    compute_type = tl.bfloat16 if C.dtype == torch.bfloat16 else tl.float16

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    fused_moe_ternary_int8_kernel[grid](
        A,
        B_packed,
        C,
        alpha,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B_packed.stride(0),
        B_packed.stride(2),
        B_packed.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        even_Ks=even_Ks,
        **config,
    )
