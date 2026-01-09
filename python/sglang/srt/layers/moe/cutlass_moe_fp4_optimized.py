"""Optimized CUTLASS FP4 MoE with pre-allocated buffers."""

import os
import torch
from dataclasses import dataclass
from sglang.srt.layers.activation import silu_and_mul


@dataclass 
class FP4MoEBuffers:
    """Pre-allocated buffers for FP4 MoE."""
    a_map: torch.Tensor
    c_map: torch.Tensor
    rep_a_shuffled: torch.Tensor
    rep_a_fp4: torch.Tensor
    rep_a_scales: torch.Tensor
    c1: torch.Tensor
    intermediate: torch.Tensor
    int_fp4: torch.Tensor
    int_scales: torch.Tensor
    c2: torch.Tensor
    c2_shuffled: torch.Tensor
    max_expanded: int

    @classmethod
    def create(cls, device, dtype, max_tokens, topk, hidden_size, intermediate_size):
        M = max_tokens * topk
        # Allocate SAFE large buffer size to prevent kernel crashes
        # The kernel seems to require a large buffer (possibly 65536*topk) regardless of actual batch
        # We allocate this ONCE, so the memory cost is fixed (approx 200MB) and performance cost is zero
        SAFE_MAX_TOKENS = 65536 
        S = SAFE_MAX_TOKENS * topk
        
        scales_k = (hidden_size // 16 + 3) // 4
        scales_n = (intermediate_size // 16 + 3) // 4
        
        return cls(
            a_map=torch.empty(M, dtype=torch.int32, device=device),
            c_map=torch.empty(M, dtype=torch.int32, device=device),
            rep_a_shuffled=torch.empty(M, hidden_size, dtype=dtype, device=device),
            rep_a_fp4=torch.empty(M, hidden_size // 2, dtype=torch.uint8, device=device),
            rep_a_scales=torch.zeros(S, scales_k, dtype=torch.int32, device=device),
            c1=torch.empty(M, intermediate_size * 2, dtype=dtype, device=device),
            intermediate=torch.empty(M, intermediate_size, dtype=dtype, device=device),
            int_fp4=torch.empty(M, intermediate_size // 2, dtype=torch.uint8, device=device),
            int_scales=torch.zeros(S, scales_n, dtype=torch.int32, device=device),
            c2=torch.empty(M, hidden_size, dtype=dtype, device=device),
            c2_shuffled=torch.empty(M, hidden_size, dtype=dtype, device=device),
            max_expanded=M,
        )


def cutlass_moe_fp4_optimized(
    a, a1_gscale, w1_fp4, w1_blockscale, w1_alphas,
    a2_gscale, w2_fp4, w2_blockscale, w2_alphas,
    topk_weights, topk_ids, params, buffers,
    apply_router_weight_on_input=False,
):
    from sgl_kernel import prepare_moe_input
    
    m, k = a.shape
    topk = topk_ids.shape[1]
    M = m * topk
    
    # Routing
    prepare_moe_input(
        topk_ids, params.expert_offsets, params.problem_sizes1, params.problem_sizes2,
        buffers.a_map[:M], buffers.c_map[:M],
        params.num_experts, params.intermediate_size_per_partition, params.hidden_size,
        params.blockscale_offsets,
    )
    
    # GEMM1: scatter -> quant -> matmul
    torch.ops.sgl_kernel.shuffle_rows.default(a, buffers.a_map[:M], buffers.rep_a_shuffled[:M])
    
    torch.ops.sgl_kernel.scaled_fp4_experts_quant.default(
        buffers.rep_a_fp4[:M], buffers.rep_a_scales, # Pass FULL scales buffer
        buffers.rep_a_shuffled[:M], a1_gscale,
        params.expert_offsets, params.blockscale_offsets,
    )
    g1 = params.to_gemm1_args()
    torch.ops.sgl_kernel.cutlass_fp4_group_mm.default(
        buffers.c1[:M], buffers.rep_a_fp4[:M], w1_fp4,
        buffers.rep_a_scales.view(torch.float8_e4m3fn), w1_blockscale, w1_alphas, # Pass FULL scales view
        g1["ab_strides"], g1["c_strides"], g1["problem_sizes"],
        g1["expert_offsets"], g1["blockscale_offsets"],
    )
    
    # SiLU
    silu_and_mul(buffers.c1[:M], buffers.intermediate[:M])
    
    # GEMM2: quant -> matmul -> unshuffle
    torch.ops.sgl_kernel.scaled_fp4_experts_quant.default(
        buffers.int_fp4[:M], buffers.int_scales, # Pass FULL scales buffer
        buffers.intermediate[:M], a2_gscale,
        params.expert_offsets, params.blockscale_offsets,
    )
    g2 = params.to_gemm2_args()
    torch.ops.sgl_kernel.cutlass_fp4_group_mm.default(
        buffers.c2[:M], buffers.int_fp4[:M], w2_fp4,
        buffers.int_scales.view(torch.float8_e4m3fn), w2_blockscale, w2_alphas, # Pass FULL scales view
        g2["ab_strides"], g2["c_strides"], g2["problem_sizes"],
        g2["expert_offsets"], g2["blockscale_offsets"],
    )
    torch.ops.sgl_kernel.shuffle_rows.default(buffers.c2[:M], buffers.c_map[:M], buffers.c2_shuffled[:M])
    
    # Combine
    out = buffers.c2_shuffled[:M].view(m, topk, params.hidden_size)
    if not apply_router_weight_on_input:
        out = out * topk_weights.view(m, topk, 1).to(a.dtype)
    return out.sum(dim=1)
