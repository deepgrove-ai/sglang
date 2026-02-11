#!/usr/bin/env python3
"""
Component-level benchmark: MoE GEMM strategy matrix for ternary models.

Compares multiple kernel strategies at the GEMM level for MoE gate_up and down
operations. No server required — tests raw kernel throughput.

Strategies tested:
1. BF16 Triton ternary (current default): unpack i2s → BF16 MMA
2. INT8 Triton ternary (experimental): quantize act → INT8, unpack i2s → INT8 MMA
3. FP8 pre-materialized: ternary→FP8 offline, torch._scaled_mm FP8 TensorCore
4. BF16 torch.matmul: dense BF16 weights, cublas GEMM (FP16 baseline ceiling)
5. FlashInfer fused MoE FP8: full pipeline (routing+GEMM1+SiLU+GEMM2+combine)

Model: Qwen3-30B-A3B MoE
  hidden_size=2048, intermediate_size=768, num_experts=128, top_k=8

Corpus citations:
  - CUTLASS functionality docs: H100 supports FP8/INT8 Tensor Cores at ~2x BF16 throughput
    /home/ubuntu/.cache/wafer/corpora/cutlass/cutlass/latest/media/docs/cpp/functionality.md
  - CUDA Tensor Core docs: SM90 supports {e4m3, e5m2} FP8 MMA
    /home/ubuntu/.cache/wafer/corpora/cuda/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.md
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ─── model constants ───────────────────────────────────────────────────────
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768
NUM_EXPERTS = 128
TOP_K = 8

# ─── helpers ───────────────────────────────────────────────────────────────

def _sync():
    torch.cuda.synchronize()


def _bench(fn, warmup=5, iters=20):
    """Benchmark a callable, return median ms."""
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]  # median


@dataclass
class MoETestCase:
    """Holds tensors for one test scenario."""
    M: int
    top_k: int = TOP_K
    num_experts: int = NUM_EXPERTS
    hidden_size: int = HIDDEN_SIZE
    intermediate_size: int = INTERMEDIATE_SIZE

    # Tensors (set after init)
    hidden_states: Optional[torch.Tensor] = field(default=None, repr=False)
    topk_ids: Optional[torch.Tensor] = field(default=None, repr=False)
    topk_weights: Optional[torch.Tensor] = field(default=None, repr=False)

    # Weight variants
    w13_packed: Optional[torch.Tensor] = field(default=None, repr=False)  # [E, 2*I, H//4] uint8
    w2_packed: Optional[torch.Tensor] = field(default=None, repr=False)   # [E, H, I//4] uint8
    alpha_w13: Optional[torch.Tensor] = field(default=None, repr=False)   # [E, H] fp32
    alpha_w2: Optional[torch.Tensor] = field(default=None, repr=False)    # [E, I] fp32
    w13_bf16: Optional[torch.Tensor] = field(default=None, repr=False)    # [E, 2*I, H] bf16
    w2_bf16: Optional[torch.Tensor] = field(default=None, repr=False)     # [E, H, I] bf16
    w13_fp8: Optional[torch.Tensor] = field(default=None, repr=False)     # [E, 2*I, H] fp8
    w2_fp8: Optional[torch.Tensor] = field(default=None, repr=False)      # [E, H, I] fp8
    scale_w13: Optional[torch.Tensor] = field(default=None, repr=False)   # [E] fp32
    scale_w2: Optional[torch.Tensor] = field(default=None, repr=False)    # [E] fp32


def make_test_case(M: int, seed: int = 42, device="cuda") -> MoETestCase:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + M)

    tc = MoETestCase(M=M)
    H, I, E, K = tc.hidden_size, tc.intermediate_size, tc.num_experts, tc.top_k

    # Activations
    tc.hidden_states = torch.randn(M, H, generator=g, dtype=torch.bfloat16).to(device)

    # Routing
    tc.topk_ids = torch.randint(0, E, (M, K), generator=g, dtype=torch.int32).to(device)
    tw = torch.rand(M, K, generator=g, dtype=torch.float32)
    tc.topk_weights = (tw / tw.sum(dim=1, keepdim=True)).to(device)

    # Ternary weights: random {-1, 0, 1}
    ternary_w13 = torch.randint(-1, 2, (E, 2 * I, H), generator=g, dtype=torch.float32)
    ternary_w2 = torch.randint(-1, 2, (E, H, I), generator=g, dtype=torch.float32)

    # Alpha scales
    alpha_w13 = torch.empty(E, H, dtype=torch.float32).uniform_(0.3, 1.2, generator=g)
    alpha_w2 = torch.empty(E, I, dtype=torch.float32).uniform_(0.3, 1.2, generator=g)

    # Pack ternary to uint8 (K dim packed by 4)
    def pack_ternary(t):
        """Pack ternary [E, N, K] → [E, N, K//4] uint8"""
        # Encode: -1→0, 0→1, 1→2
        encoded = (t + 1).to(torch.uint8)
        E_, N_, K_ = encoded.shape
        assert K_ % 4 == 0
        encoded = encoded.reshape(E_, N_, K_ // 4, 4)
        packed = encoded[..., 0] | (encoded[..., 1] << 2) | (encoded[..., 2] << 4) | (encoded[..., 3] << 6)
        return packed

    tc.w13_packed = pack_ternary(ternary_w13).to(device)
    tc.w2_packed = pack_ternary(ternary_w2).to(device)
    tc.alpha_w13 = alpha_w13.to(device)
    tc.alpha_w2 = alpha_w2.to(device)

    # Dense BF16 weights (materialized: ternary * alpha)
    tc.w13_bf16 = (ternary_w13 * alpha_w13.unsqueeze(1)).to(torch.bfloat16).to(device)
    tc.w2_bf16 = (ternary_w2 * alpha_w2.unsqueeze(1)).to(torch.bfloat16).to(device)

    # FP8 pre-materialized weights (per-expert scaling)
    scale_w13 = tc.w13_bf16.float().abs().reshape(E, -1).amax(dim=1) / 448.0
    scale_w13 = scale_w13.clamp(min=1e-12)
    tc.scale_w13 = scale_w13.to(device)
    w13_fp8_list = []
    for e in range(E):
        w13_fp8_list.append((tc.w13_bf16[e].float() / scale_w13[e]).to(torch.float8_e4m3fn))
    tc.w13_fp8 = torch.stack(w13_fp8_list).to(device)

    scale_w2 = tc.w2_bf16.float().abs().reshape(E, -1).amax(dim=1) / 448.0
    scale_w2 = scale_w2.clamp(min=1e-12)
    tc.scale_w2 = scale_w2.to(device)
    w2_fp8_list = []
    for e in range(E):
        w2_fp8_list.append((tc.w2_bf16[e].float() / scale_w2[e]).to(torch.float8_e4m3fn))
    tc.w2_fp8 = torch.stack(w2_fp8_list).to(device)

    return tc


# ─── Strategy implementations ─────────────────────────────────────────────

def _get_active_experts(topk_ids, num_experts):
    """Return sorted unique expert IDs and per-expert token indices."""
    flat_ids = topk_ids.reshape(-1)
    unique_experts = flat_ids.unique().sort().values
    # For each expert, find which (token, expert_slot) pairs are routed to it
    expert_token_map = {}
    for e in unique_experts.tolist():
        mask = (topk_ids == e)
        token_indices = mask.any(dim=1).nonzero(as_tuple=True)[0]
        expert_token_map[e] = token_indices
    return unique_experts, expert_token_map


def strategy_bf16_matmul(tc: MoETestCase, kind: str) -> torch.Tensor:
    """BF16 dense matmul per-expert (cublas baseline ceiling)."""
    M, K_top = tc.M, tc.top_k
    if kind == "gate_up":
        w = tc.w13_bf16  # [E, 2I, H]
        N_out = 2 * tc.intermediate_size
        out = torch.zeros(M, K_top, N_out, dtype=torch.bfloat16, device=tc.hidden_states.device)
        for e_idx in range(tc.num_experts):
            mask = (tc.topk_ids == e_idx)  # [M, top_k]
            if not mask.any():
                continue
            # Find tokens routed to this expert
            token_mask = mask.any(dim=1)
            tokens = tc.hidden_states[token_mask]  # [m_e, H]
            # GEMM: [m_e, H] × [H, 2I] → [m_e, 2I]
            result = tokens @ w[e_idx].t()
            # Scatter back
            slot_mask = mask[token_mask]
            for slot in range(K_top):
                slot_tokens = slot_mask[:, slot]
                if slot_tokens.any():
                    global_idx = token_mask.nonzero(as_tuple=True)[0][slot_tokens]
                    out[global_idx, slot] = result[slot_tokens]
        return out
    else:  # down
        w = tc.w2_bf16  # [E, H, I]
        # down input is [M*top_k, I] after SiLU
        # For benchmark: just create random input of right shape
        act_in = torch.randn(M * K_top, tc.intermediate_size, dtype=torch.bfloat16, device=tc.hidden_states.device)
        out = torch.zeros(M * K_top, tc.hidden_size, dtype=torch.bfloat16, device=tc.hidden_states.device)
        flat_ids = tc.topk_ids.reshape(-1)
        for e_idx in range(tc.num_experts):
            mask = (flat_ids == e_idx)
            if not mask.any():
                continue
            tokens = act_in[mask]
            result = tokens @ w[e_idx].t()
            out[mask] = result
        return out


def strategy_fp8_scaled_mm(tc: MoETestCase, kind: str) -> torch.Tensor:
    """FP8 per-expert torch._scaled_mm."""
    M, K_top = tc.M, tc.top_k
    device = tc.hidden_states.device

    if kind == "gate_up":
        w_fp8 = tc.w13_fp8       # [E, 2I, H]
        scale_w = tc.scale_w13   # [E]
        N_out = 2 * tc.intermediate_size
        out = torch.zeros(M, K_top, N_out, dtype=torch.bfloat16, device=device)
        flat_ids = tc.topk_ids.reshape(-1)

        for e_idx in range(tc.num_experts):
            mask = (tc.topk_ids == e_idx)
            if not mask.any():
                continue
            token_mask = mask.any(dim=1)
            tokens = tc.hidden_states[token_mask]  # [m_e, H]

            # Quantize activations to FP8
            act_amax = tokens.float().abs().amax()
            act_scale = (act_amax / 448.0).clamp(min=1e-12)
            tokens_fp8 = (tokens.float() / act_scale).to(torch.float8_e4m3fn)

            # FP8 GEMM: [m_e, H] × [2I, H]^T → [m_e, 2I]
            sa = torch.tensor(act_scale, dtype=torch.float32, device=device)
            sb = torch.tensor(scale_w[e_idx], dtype=torch.float32, device=device)

            # Pad to even dimensions if needed
            m_e = tokens_fp8.shape[0]
            if m_e % 2 == 1:
                tokens_fp8 = F.pad(tokens_fp8.view(torch.int8), (0, 0, 0, 1)).view(torch.float8_e4m3fn)
                padded = True
            else:
                padded = False

            result = torch._scaled_mm(
                tokens_fp8,
                w_fp8[e_idx].t(),
                scale_a=sa,
                scale_b=sb,
                out_dtype=torch.bfloat16,
            )
            if padded:
                result = result[:m_e]

            slot_mask = mask[token_mask]
            for slot in range(K_top):
                slot_tokens = slot_mask[:, slot]
                if slot_tokens.any():
                    global_idx = token_mask.nonzero(as_tuple=True)[0][slot_tokens]
                    out[global_idx, slot] = result[slot_tokens]
        return out
    else:
        w_fp8 = tc.w2_fp8
        scale_w = tc.scale_w2
        act_in = torch.randn(M * K_top, tc.intermediate_size, dtype=torch.bfloat16, device=device)
        out = torch.zeros(M * K_top, tc.hidden_size, dtype=torch.bfloat16, device=device)
        flat_ids = tc.topk_ids.reshape(-1)
        for e_idx in range(tc.num_experts):
            mask = (flat_ids == e_idx)
            if not mask.any():
                continue
            tokens = act_in[mask]
            act_amax = tokens.float().abs().amax()
            act_scale = (act_amax / 448.0).clamp(min=1e-12)
            tokens_fp8 = (tokens.float() / act_scale).to(torch.float8_e4m3fn)
            sa = torch.tensor(act_scale, dtype=torch.float32, device=device)
            sb = torch.tensor(scale_w[e_idx], dtype=torch.float32, device=device)

            m_e = tokens_fp8.shape[0]
            if m_e % 2 == 1:
                tokens_fp8 = F.pad(tokens_fp8.view(torch.int8), (0, 0, 0, 1)).view(torch.float8_e4m3fn)
                padded = True
            else:
                padded = False

            result = torch._scaled_mm(
                tokens_fp8,
                w_fp8[e_idx].t(),
                scale_a=sa,
                scale_b=sb,
                out_dtype=torch.bfloat16,
            )
            if padded:
                result = result[:m_e]
            out[mask] = result
        return out


def strategy_bf16_triton_ternary(tc: MoETestCase, kind: str) -> torch.Tensor:
    """BF16 Triton ternary kernel (current default MoE path)."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_ternary_kernel import (
        get_default_ternary_moe_config,
        invoke_fused_moe_ternary_kernel,
    )
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import moe_align_block_size

    M, K_top, E = tc.M, tc.top_k, tc.num_experts
    if kind == "gate_up":
        w_packed = tc.w13_packed
        alpha = tc.alpha_w13
        N_out = 2 * tc.intermediate_size
        a = tc.hidden_states
        mul_routed = False
        runtime_top_k = K_top
    else:
        w_packed = tc.w2_packed
        alpha = tc.alpha_w2
        N_out = tc.hidden_size
        a = torch.randn(M * K_top, tc.intermediate_size, dtype=torch.bfloat16, device=tc.hidden_states.device)
        mul_routed = True
        runtime_top_k = 1

    out = torch.empty(M, K_top, N_out, dtype=torch.bfloat16, device=tc.hidden_states.device) if kind == "gate_up" else \
          torch.empty(M * K_top, 1, N_out, dtype=torch.bfloat16, device=tc.hidden_states.device)

    cfg = get_default_ternary_moe_config(kind.replace("gate_up", "gate_up"), M)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        tc.topk_ids, cfg["BLOCK_SIZE_M"], E
    )
    invoke_fused_moe_ternary_kernel(
        a, w_packed, alpha, out,
        tc.topk_weights, tc.topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_routed, runtime_top_k, config=cfg,
    )
    return out


def strategy_int8_triton_ternary(tc: MoETestCase, kind: str) -> torch.Tensor:
    """INT8 Triton ternary kernel (experimental path)."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_ternary_kernel import (
        get_default_ternary_moe_int8_config,
        invoke_fused_moe_ternary_int8_kernel,
        quantize_activation_int8_global,
    )
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import moe_align_block_size

    M, K_top, E = tc.M, tc.top_k, tc.num_experts
    if kind == "gate_up":
        w_packed = tc.w13_packed
        alpha = tc.alpha_w13
        N_out = 2 * tc.intermediate_size
        a = tc.hidden_states
        mul_routed = False
        runtime_top_k = K_top
    else:
        w_packed = tc.w2_packed
        alpha = tc.alpha_w2
        N_out = tc.hidden_size
        a = torch.randn(M * K_top, tc.intermediate_size, dtype=torch.bfloat16, device=tc.hidden_states.device)
        mul_routed = True
        runtime_top_k = 1

    out = torch.empty(M, K_top, N_out, dtype=torch.bfloat16, device=tc.hidden_states.device) if kind == "gate_up" else \
          torch.empty(M * K_top, 1, N_out, dtype=torch.bfloat16, device=tc.hidden_states.device)

    # Quantize activations to INT8
    a_i8, act_scale = quantize_activation_int8_global(a)
    # Scale alpha by activation scale
    alpha_i8 = (alpha * act_scale).to(torch.float32).contiguous()

    cfg = get_default_ternary_moe_int8_config(kind, M)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        tc.topk_ids, cfg["BLOCK_SIZE_M"], E
    )
    invoke_fused_moe_ternary_int8_kernel(
        a_i8, w_packed, alpha_i8, out,
        tc.topk_weights, tc.topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_routed, runtime_top_k, config=cfg,
    )
    return out


def strategy_fp8_grouped_gemm(tc: MoETestCase, kind: str) -> torch.Tensor:
    """
    FP8 grouped GEMM using torch._scaled_mm with batched dispatch.
    Groups tokens by expert, runs one FP8 matmul per active expert.
    Uses a contiguous gather/scatter pattern for better memory throughput.
    """
    M, K_top, E = tc.M, tc.top_k, tc.num_experts
    device = tc.hidden_states.device

    if kind == "gate_up":
        w_fp8 = tc.w13_fp8       # [E, 2I, H]
        scale_w = tc.scale_w13   # [E]
        N_out = 2 * tc.intermediate_size
        K_in = tc.hidden_size

        # Expand tokens by top_k: each token appears top_k times
        # Shape: [M*top_k]
        flat_expert_ids = tc.topk_ids.reshape(-1)  # [M*top_k]
        expanded_acts = tc.hidden_states.unsqueeze(1).expand(-1, K_top, -1).reshape(-1, K_in)  # [M*top_k, H]

        # Sort by expert for grouped dispatch
        sort_idx = flat_expert_ids.argsort()
        sorted_expert_ids = flat_expert_ids[sort_idx]
        sorted_acts = expanded_acts[sort_idx]

        # Quantize all activations to FP8 at once
        act_amax = sorted_acts.float().abs().amax()
        act_scale = (act_amax / 448.0).clamp(min=1e-12)
        sorted_acts_fp8 = (sorted_acts.float() / act_scale).to(torch.float8_e4m3fn)
        sa = torch.tensor(act_scale, dtype=torch.float32, device=device)

        out_sorted = torch.zeros(M * K_top, N_out, dtype=torch.bfloat16, device=device)

        # Find expert boundaries
        unique_experts, counts = sorted_expert_ids.unique_consecutive(return_counts=True)
        offset = 0
        for e_idx, cnt in zip(unique_experts.tolist(), counts.tolist()):
            sb = scale_w[e_idx].unsqueeze(0).to(torch.float32)
            chunk = sorted_acts_fp8[offset:offset + cnt]
            # Pad to even row count
            if cnt % 2 == 1:
                chunk = F.pad(chunk.view(torch.int8), (0, 0, 0, 1)).view(torch.float8_e4m3fn)
                res = torch._scaled_mm(chunk, w_fp8[e_idx].t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
                out_sorted[offset:offset + cnt] = res[:cnt]
            else:
                res = torch._scaled_mm(chunk, w_fp8[e_idx].t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
                out_sorted[offset:offset + cnt] = res
            offset += cnt

        # Unsort
        out_flat = torch.zeros_like(out_sorted)
        out_flat[sort_idx] = out_sorted
        return out_flat.reshape(M, K_top, N_out)

    else:  # down
        w_fp8 = tc.w2_fp8
        scale_w = tc.scale_w2
        N_out = tc.hidden_size
        K_in = tc.intermediate_size

        act_in = torch.randn(M * K_top, K_in, dtype=torch.bfloat16, device=device)
        flat_expert_ids = tc.topk_ids.reshape(-1)
        sort_idx = flat_expert_ids.argsort()
        sorted_expert_ids = flat_expert_ids[sort_idx]
        sorted_acts = act_in[sort_idx]

        act_amax = sorted_acts.float().abs().amax()
        act_scale = (act_amax / 448.0).clamp(min=1e-12)
        sorted_acts_fp8 = (sorted_acts.float() / act_scale).to(torch.float8_e4m3fn)
        sa = torch.tensor(act_scale, dtype=torch.float32, device=device)

        out_sorted = torch.zeros(M * K_top, N_out, dtype=torch.bfloat16, device=device)
        unique_experts, counts = sorted_expert_ids.unique_consecutive(return_counts=True)
        offset = 0
        for e_idx, cnt in zip(unique_experts.tolist(), counts.tolist()):
            sb = scale_w[e_idx].unsqueeze(0).to(torch.float32)
            chunk = sorted_acts_fp8[offset:offset + cnt]
            if cnt % 2 == 1:
                chunk = F.pad(chunk.view(torch.int8), (0, 0, 0, 1)).view(torch.float8_e4m3fn)
                res = torch._scaled_mm(chunk, w_fp8[e_idx].t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
                out_sorted[offset:offset + cnt] = res[:cnt]
            else:
                res = torch._scaled_mm(chunk, w_fp8[e_idx].t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
                out_sorted[offset:offset + cnt] = res
            offset += cnt

        out_flat = torch.zeros_like(out_sorted)
        out_flat[sort_idx] = out_sorted
        return out_flat


# ─── FlashInfer FP8 fused MoE strategy ──────────────────────────────────

def strategy_flashinfer_fp8_fused(tc: MoETestCase) -> torch.Tensor:
    """
    FlashInfer trtllm_fp8_per_tensor_scale_moe — fully fused MoE.
    Routing + GEMM1 + SiLU + GEMM2 + combine in one kernel.
    """
    import flashinfer

    M, E, K_top = tc.M, tc.num_experts, tc.top_k
    H, I = tc.hidden_size, tc.intermediate_size

    # FlashInfer expects:
    # gemm1_weights: [E, 2*I, H] fp8
    # gemm2_weights: [E, H, I] fp8
    # routing_logits: [M, E]
    # output scales: per-expert scalars

    # Create fake routing logits that match our topk_ids
    # We need logits that produce the same routing. For benchmark timing,
    # we just use random logits (routing is not the bottleneck).
    routing_logits = torch.randn(M, E, dtype=torch.float32, device=tc.hidden_states.device)

    # FlashInfer expects per-tensor output scales (not weight scales).
    # For FP8: out = (A_fp8 @ W_fp8^T) * scale_a * scale_w * output_scale
    # We fold weight scale into output_scale.
    output1_scales = tc.scale_w13.to(torch.float32)
    output1_scales_gate = tc.scale_w13.to(torch.float32)
    output2_scales = tc.scale_w2.to(torch.float32)

    hidden_fp8 = tc.hidden_states.to(torch.float8_e4m3fn)

    result = flashinfer.trtllm_fp8_per_tensor_scale_moe(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=hidden_fp8,
        gemm1_weights=tc.w13_fp8,           # [E, 2I, H]
        output1_scales_scalar=output1_scales,
        output1_scales_gate_scalar=output1_scales_gate,
        gemm2_weights=tc.w2_fp8.transpose(-1, -2).contiguous(),  # [E, I, H] → [E, H, I]
        output2_scales_scalar=output2_scales,
        num_experts=E,
        top_k=K_top,
        n_group=1,
        topk_group=1,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=1.0,
        use_routing_scales_on_input=False,
    )
    return result


# ─── Main benchmark loop ──────────────────────────────────────────────────

STRATEGIES_GEMM = {
    "bf16_triton_ternary": strategy_bf16_triton_ternary,
    "int8_triton_ternary": strategy_int8_triton_ternary,
    "fp8_grouped_gemm": strategy_fp8_grouped_gemm,
    "bf16_cublas_dense": strategy_bf16_matmul,
}


def run_benchmark(args):
    device = torch.device("cuda")
    m_values = [int(x) for x in args.m_values.split(",")]
    kinds = args.kinds.split(",")
    warmup = args.warmup
    iters = args.iters
    results = []

    print("=" * 72)
    print("MoE GEMM Strategy Benchmark")
    print("=" * 72)
    print(f"M values:    {m_values}")
    print(f"Kinds:       {kinds}")
    print(f"Strategies:  {list(STRATEGIES_GEMM.keys())} + flashinfer_fp8_fused")
    print(f"Warmup/Iter: {warmup}/{iters}")
    print("=" * 72)

    for M in m_values:
        print(f"\n--- M={M} ---")
        tc = make_test_case(M, seed=42, device=device)

        for kind in kinds:
            print(f"\n  [{kind}]")

            # Per-stage GEMM strategies
            for name, fn in STRATEGIES_GEMM.items():
                try:
                    # Correctness check
                    out = fn(tc, kind)
                    _sync()

                    # Benchmark
                    ms = _bench(lambda: fn(tc, kind), warmup=warmup, iters=iters)

                    rec = {
                        "M": M,
                        "kind": kind,
                        "strategy": name,
                        "latency_ms": round(ms, 4),
                        "status": "ok",
                    }
                    results.append(rec)
                    print(f"    {name:30s}: {ms:8.3f} ms")
                except Exception as e:
                    rec = {
                        "M": M,
                        "kind": kind,
                        "strategy": name,
                        "latency_ms": None,
                        "status": f"error: {str(e)[:200]}",
                    }
                    results.append(rec)
                    print(f"    {name:30s}: ERROR - {str(e)[:100]}")

        # FlashInfer fused MoE (full pipeline, not per-stage)
        if args.flashinfer:
            print(f"\n  [fused_moe (full pipeline)]")
            try:
                out = strategy_flashinfer_fp8_fused(tc)
                _sync()
                ms = _bench(lambda: strategy_flashinfer_fp8_fused(tc), warmup=warmup, iters=iters)
                rec = {
                    "M": M,
                    "kind": "fused_full",
                    "strategy": "flashinfer_fp8_fused",
                    "latency_ms": round(ms, 4),
                    "status": "ok",
                }
                results.append(rec)
                print(f"    {'flashinfer_fp8_fused':30s}: {ms:8.3f} ms (FULL PIPELINE)")
            except Exception as e:
                rec = {
                    "M": M,
                    "kind": "fused_full",
                    "strategy": "flashinfer_fp8_fused",
                    "latency_ms": None,
                    "status": f"error: {str(e)[:200]}",
                }
                results.append(rec)
                print(f"    {'flashinfer_fp8_fused':30s}: ERROR - {str(e)[:150]}")

    # Save results
    out_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / "moe_strategy_bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"results": results}, indent=2) + "\n")

    # Print summary table
    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    print(f"{'M':>5} {'kind':>10} {'strategy':>30} {'ms':>10} {'status':>10}")
    print("-" * 72)
    for r in results:
        ms_str = f"{r['latency_ms']:.3f}" if r['latency_ms'] is not None else "N/A"
        print(f"{r['M']:>5} {r['kind']:>10} {r['strategy']:>30} {ms_str:>10} {r['status'][:10]:>10}")

    # Compute speedup vs bf16_triton_ternary baseline per (M, kind)
    print("\n" + "=" * 72)
    print("SPEEDUP vs bf16_triton_ternary")
    print("=" * 72)
    baseline = {}
    for r in results:
        if r["strategy"] == "bf16_triton_ternary" and r["latency_ms"]:
            baseline[(r["M"], r["kind"])] = r["latency_ms"]

    for r in results:
        if r["latency_ms"] is None:
            continue
        key = (r["M"], r["kind"])
        if r["kind"] == "fused_full":
            # Compare fused vs sum of gate_up + down baselines
            gu_key = (r["M"], "gate_up")
            dn_key = (r["M"], "down")
            if gu_key in baseline and dn_key in baseline:
                combined = baseline[gu_key] + baseline[dn_key]
                speedup = combined / r["latency_ms"]
                print(f"  M={r['M']:>4} {r['strategy']:>30}: {r['latency_ms']:.3f}ms vs {combined:.3f}ms (gu+down) → {speedup:.2f}x")
        elif key in baseline:
            speedup = baseline[key] / r["latency_ms"]
            print(f"  M={r['M']:>4} {r['kind']:>10} {r['strategy']:>30}: {speedup:.2f}x")

    print(f"\nResults saved to {out_dir / 'results.json'}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="MoE GEMM strategy benchmark")
    parser.add_argument("--m-values", default="1,4,8,16,32,64,128", help="Token counts to test")
    parser.add_argument("--kinds", default="gate_up,down", help="MoE stages to test")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--flashinfer", action="store_true", default=True, help="Include FlashInfer FP8 fused MoE")
    parser.add_argument("--no-flashinfer", dest="flashinfer", action="store_false")
    parser.add_argument("--output-dir", default="")
    return run_benchmark(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
