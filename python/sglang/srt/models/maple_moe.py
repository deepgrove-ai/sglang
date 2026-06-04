# coding=utf-8
"""Inference-only Maple model for SGLang.

SGLang-flavored port of VeOmni's modeling_maple.py:
  - QKVParallelLinear / RowParallelLinear / VocabParallelEmbedding / ParallelLMHead
  - FusedMoE (Triton) for the MoE experts
  - RadixAttention replaces FA3 + DynamicCache

Router replay: ``record_topk`` is called inside the MoE forward so slime
can dump SGLang's top-k decisions for the trainer to replay.
"""

import logging
import math
import re
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

try:
    from slime.router_replay import sglang_capture as _slime_router_replay
except ImportError:  # pragma: no cover
    _slime_router_replay = None

# sonicmoe MoE kernel — same kernel ve's trainer uses. Required for bit-exact
# MoE output parity between ve↔sgl; FusedMoE (the default sgl path) diverges
# from sonicmoe at small post-norm input magnitudes. Optional unless
# USE_SONIC_MOE=True (default path uses FusedMoE and does not need sonicmoe).
try:
    from sonicmoe.functional import moe_TC_softmax_topk_layer
    from sonicmoe.enums import ActivationType
except ImportError:  # pragma: no cover
    moe_TC_softmax_topk_layer = None
    ActivationType = None

from sglang.srt.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsMetadata
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.utils import add_prefix
from .maple_fa.nope import triton_qk_norm_and_split_forward
from .maple_fa.rope import triton_qk_norm_and_half_rope_forward


logger = logging.getLogger(__name__)

USE_FUSED_EXPERTS = True
# Use sonicmoe (ve trainer's MoE kernel) instead of FusedMoE for bit-exact
# parity at small post-norm input magnitudes. Assumes TP=1 (no expert
# sharding across ranks) — sonicmoe doesn't implement TP shards here.
USE_SONIC_MOE = False
USE_SWIGLU_CLAMP = True
MAPLE_SWIGLU_CLAMP_LIMIT = 7.0


def _use_swiglu_clamp(config) -> bool:
    return bool(getattr(config, "use_swiglu_clamp", USE_SWIGLU_CLAMP))


# Dump-hooks no-op unless MAPLE_DUMP_HIDDEN=1 — keeps training runs from
# producing millions of .pt files. Set MAPLE_DUMP_HIDDEN=1 for diagnostic
# scripts like compare_full_stack.py.
import os as _early_os
_DUMP_HIDDEN_ENABLED = _early_os.environ.get("MAPLE_DUMP_HIDDEN", "0") == "1"
# NaN detection between decoder substeps. Off by default — each check forces
# a GPU sync. Enable with MAPLE_NAN_CHECK=1 for diagnostics.
_NAN_CHECK_ENABLED = _early_os.environ.get("MAPLE_NAN_CHECK", "0") == "1"


def _nan_check(hidden_states: torch.Tensor, layer_idx: int, tag: str) -> None:
    """If hidden_states contains a NaN/Inf, print a one-line summary including
    layer, tag, shape, and counts. No-op unless MAPLE_NAN_CHECK=1."""
    if not _NAN_CHECK_ENABLED:
        return
    nan_count = int(torch.isnan(hidden_states).sum())
    inf_count = int(torch.isinf(hidden_states).sum())
    if nan_count or inf_count:
        print(
            f"[NAN-CHECK] sgl L{layer_idx:02d} tag={tag} "
            f"shape={tuple(hidden_states.shape)} dtype={hidden_states.dtype} "
            f"nan={nan_count} inf={inf_count}",
            flush=True,
        )

# Dump pre/post input_layernorm hidden_states for every call to ./dumps/sgl/
# (overridable via MAPLE_SGL_DUMP_DIR). Filenames are
# rank{R}_L{layer_idx:02d}_call{N:04d}_{pre,post}.pt — N increments per
# (layer, label, rank). Rank prefix prevents TP/multi-worker race conditions.
import os as _os
import threading as _threading
_SGL_DUMP_DIR = _os.environ.get(
    "MAPLE_SGL_DUMP_DIR", _os.path.join(_os.getcwd(), "dumps", "sgl")
)
_SGL_DUMP_COUNTERS: dict = {}
_SGL_WEIGHT_DUMPED: set = set()
_SGL_DUMP_LOCK = _threading.Lock()

def _sgl_resolve_rank() -> int:
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            return _dist.get_rank()
    except Exception:
        pass
    return _os.getpid()

def _sgl_dump_hidden(hidden_states, layer_idx: int, tag: str) -> None:
    """Dump a per-call hidden_states tensor. See ve modeling_maple.py for the
    matching contract — filenames and counter semantics are identical so the
    analysis scripts can pair ve↔sgl by (layer, tag, shape).
    No-op unless MAPLE_DUMP_HIDDEN=1."""
    if not _DUMP_HIDDEN_ENABLED:
        return
    _os.makedirs(_SGL_DUMP_DIR, exist_ok=True)
    rank = _sgl_resolve_rank()
    with _SGL_DUMP_LOCK:
        key = (rank, layer_idx, tag)
        n = _SGL_DUMP_COUNTERS.get(key, 0)
        fname = _os.path.join(
            _SGL_DUMP_DIR,
            f"rank{rank}_L{layer_idx:02d}_call{n:04d}_{tag}.pt",
        )
        torch.save(hidden_states.detach().to("cpu"), fname)
        _SGL_DUMP_COUNTERS[key] = n + 1

def _sgl_dump_weight(weight, layer_idx: int, tag: str) -> None:
    """Dump a parameter tensor once per (rank, layer, tag).
    No-op unless MAPLE_DUMP_HIDDEN=1."""
    if not _DUMP_HIDDEN_ENABLED:
        return
    rank = _sgl_resolve_rank()
    key = (rank, layer_idx, tag)
    if key in _SGL_WEIGHT_DUMPED:
        return
    _os.makedirs(_SGL_DUMP_DIR, exist_ok=True)
    with _SGL_DUMP_LOCK:
        if key in _SGL_WEIGHT_DUMPED:
            return
        fname = _os.path.join(
            _SGL_DUMP_DIR,
            f"rank{rank}_L{layer_idx:02d}_call0000_{tag}.pt",
        )
        torch.save(weight.detach().to("cpu"), fname)
        _SGL_WEIGHT_DUMPED.add(key)


# ── building blocks ──────────────────────────────────────────────────────────

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
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate = F.linear(x, self.gate_proj.weight)
        up = F.linear(x, self.up_proj.weight)
        gate = gate.float()
        up = up.float()
        if _use_swiglu_clamp(self.config):
            gate = torch.clamp(gate, max=MAPLE_SWIGLU_CLAMP_LIMIT)
            up = torch.clamp(
                up, min=-MAPLE_SWIGLU_CLAMP_LIMIT, max=MAPLE_SWIGLU_CLAMP_LIMIT
            )
        hidden = (self.act_fn(gate) * up).to(x.dtype)
        return F.linear(hidden, self.down_proj.weight)


class MapleRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, _fp64=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        # Diagnostic: compute in fp64 to mask any fp32 reduction-order drift
        # from CUDA kernel dispatch differences (2D vs 3D inputs, eager vs
        # inductor). Toggled on for input_layernorm only.
        self._fp64 = _fp64

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        compute_dtype = torch.float64 if self._fp64 else torch.float32
        hidden_states = hidden_states.to(compute_dtype)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(compute_dtype) * hidden_states).to(input_dtype)


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


# ── MoE: Python-reference path (used when USE_FUSED_EXPERTS=False) ───────────

class MapleSparseMoeBlockOld(nn.Module):
    """Unfused MoE block — identical math to HF; loops over experts in Python."""

    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id  # exposed for router_replay capture
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([
            MapleMLP(config=config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.gate = MapleGate(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        is_2d = hidden_states.dim() == 2
        if is_2d:
            hidden_states = hidden_states.unsqueeze(0)
        bsz, seq_len, h = hidden_states.shape

        topk_idx, topk_weight, _router_logits = self.gate(hidden_states)
        if _slime_router_replay is not None:
            _slime_router_replay.record_topk(self.layer_id, topk_idx, topk_weight)

        hidden_states = hidden_states.view(-1, h)
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(bsz, seq_len, h)
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
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
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
        final_out = weighted.sum(dim=1).type(new_x.dtype)
        return final_out


# ── MoE: FusedMoE (Triton) path — the default ────────────────────────────────

class MapleSparseMoeBlock(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id  # exposed for router_replay capture
        self.num_experts_per_tok = config.num_experts_per_tok

        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=None,
            prefix="experts",
            activation=config.hidden_act,
            gemm1_clamp_limit=(
                MAPLE_SWIGLU_CLAMP_LIMIT if _use_swiglu_clamp(config) else None
            ),
            no_combine=True,  # return (T, K, H) so we accumulate in fp32 ourselves
            inplace=False,    # required when no_combine=True
        )
        self.gate = MapleGate(config)

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight, router_logits):
        hidden_states = x
        # Pass ones so the W2 kernel multiplies by 1.0 (no-op); apply the
        # actual weights below in fp32 to match the reference moe_infer's
        # accumulation precision.
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones_like(topk_weight),
            topk_ids=topk_ids.to(torch.int32),
            router_logits=router_logits,
        )

        dispatch_output = self.experts.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        combine_input = self.experts.run_moe_core(dispatch_output=dispatch_output)
        expert_outs = combine_input.hidden_states  # (T, K, H) in model dtype, unweighted

        weighted = expert_outs.to(torch.float32).mul_(topk_weight.unsqueeze(dim=-1))
        final_out = weighted.sum(dim=1)
        if get_tensor_model_parallel_world_size() > 1:
            final_out = tensor_model_parallel_all_reduce(final_out)
        return final_out.to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        is_2d = hidden_states.dim() == 2
        if is_2d:
            hidden_states = hidden_states.unsqueeze(0)
        bsz, seq_len, h = hidden_states.shape

        topk_idx, topk_weight, router_logits = self.gate(hidden_states)
        if _slime_router_replay is not None:
            _slime_router_replay.record_topk(self.layer_id, topk_idx, topk_weight)

        hidden_states = hidden_states.view(-1, h)
        y = self.moe_infer(hidden_states, topk_idx, topk_weight, router_logits).view(bsz, seq_len, h)
        if is_2d:
            y = y.squeeze(0)
        return y


# ── MoE: sonicmoe path (matches ve's trainer kernel exactly) ─────────────────

class MapleSonicMoeExperts(nn.Module):
    """Holds the fused expert weights consumed by moe_TC_softmax_topk_layer.

    Layout matches ve's MapleSonicMoeExperts:
      gate_up_proj : (E, 2*I, H) with gate row at 2*i and up row at 2*i+1
      down_proj    : (E, H, I)
    Permutation to (2I, H, E) / (H, I, E) happens at call time.
    """

    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(config.num_experts, 2 * config.moe_intermediate_size, config.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, config.moe_intermediate_size)
        )


class MapleSparseMoeBlockSonic(nn.Module):
    """sgl MoE block that calls the same sonicmoe kernel ve's trainer uses.

    Routing is captured for slime router_replay via a separate gate forward
    so the recorded topk matches what sonicmoe will internally pick.
    """

    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.experts = MapleSonicMoeExperts(config)
        self.gate = MapleGate(config)
        # NOTE: do NOT capture the stream at __init__. sgl runs forwards on
        # internal scheduler streams that differ from whatever was current
        # during engine startup, and sonicmoe launching on the wrong stream
        # reads stale memory → NaN. Read current_stream() per-forward instead.

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        is_2d = hidden_states.dim() == 2
        if is_2d:
            hidden_states = hidden_states.unsqueeze(0)
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)

        # Let sonicmoe compute the gate internally in bf16 — the same CUDA
        # kernel runs on both sgl and ve, so internal-routing is bit-identical
        # across processes. When slime router_replay is active, run the gate
        # in Python (so we can record what it picked) and hand sonicmoe the
        # routing decisions back as tensors. The tensor-handoff path replaces
        # the older ``mod=callable`` route, which forced a Python callback
        # inside the kernel and prevented CUDA graph capture.
        replay_active = (
            _slime_router_replay is not None
            and _slime_router_replay.enabled()
        )

        topk_idx_override = None
        topk_w_override = None
        router_logits_override = None
        if replay_active:
            topk_idx_override, topk_w_override, router_logits_override = self.gate(
                hidden_states
            )
            _slime_router_replay.record_topk(
                self.layer_id, topk_idx_override, topk_w_override
            )

        # Dynamic stream capture — matches whichever stream sgl is using
        # for this layer's forward.
        stream_id = torch.cuda.current_stream().cuda_stream

        y, _router_logits, _expert_freq = moe_TC_softmax_topk_layer(
            hidden_states,
            self.gate.weight,
            self.experts.gate_up_proj.permute(1, 2, 0),
            None,                               # b1
            self.experts.down_proj.permute(1, 2, 0),
            None,                               # b2
            self.top_k,
            stream_id,
            ActivationType.SWIGLU,
            False,                              # is_inference_mode_enabled
            bias=None,
            scaling_factor=1.0,
            norm_topk=True,
            topk_indices_override=topk_idx_override,
            topk_scores_override=topk_w_override,
            router_logits_override=router_logits_override,
        )

        y = y.view(bsz, seq_len, h)
        if is_2d:
            y = y.squeeze(0)
        return y


# ── attention (RadixAttention is the only inference-specific substitution) ────

class MapleAttention(nn.Module):
    def __init__(self, config, layer_idx: int, prefix: str = ""):
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

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        assert self.num_heads % attn_tp_size == 0, (
            f"num_heads={self.num_heads} not divisible by attn_tp_size={attn_tp_size}"
        )
        assert self.num_key_value_heads % attn_tp_size == 0, (
            f"num_key_value_heads={self.num_key_value_heads} not divisible by attn_tp_size={attn_tp_size}"
        )
        self.num_local_heads = self.num_heads // attn_tp_size
        self.num_local_kv_heads = self.num_key_value_heads // attn_tp_size

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_key_value_heads,
            bias=False,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.q_norm = MapleRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MapleRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        sliding_window_size = self.sliding_window if self.sliding_window is not None else -1
        self.attn = RadixAttention(
            num_heads=self.num_local_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            layer_id=layer_idx,
            sliding_window_size=sliding_window_size,
        )

        # reduce_results=False so we can do the cross-rank all-reduce in fp32 below.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]

        out_qkv, _ = self.qkv_proj(hidden_states)
        print(f"qkv: {out_qkv.shape}")
        # qkv = out_qkv.view(num_tokens, self.num_local_heads + 2 * self.num_local_kv_heads, self.head_dim)
        if len(out_qkv.shape) == 2:
            out_qkv = out_qkv.unsqueeze(0)
        # query_states, key_states, value_states = qkv.split(
        #     [self.num_local_heads, self.num_local_kv_heads, self.num_local_kv_heads], dim=-2
        # )

        # query_states = self.q_norm(query_states)
        # key_states = self.k_norm(key_states)
        # if self.sliding_window is not None and position_embeddings is not None:
        #     cos, sin, _ = position_embeddings
        #     cos = cos.squeeze(0)
        #     sin = sin.squeeze(0)
        #     query_states, key_states = apply_rotary_pos_emb(
        #         query_states, key_states, cos, sin, unsqueeze_dim=1
        #     )

        # q = query_states.reshape(num_tokens, self.num_local_heads * self.head_dim)
        # k = key_states.reshape(num_tokens, self.num_local_kv_heads * self.head_dim)
        # v = value_states.reshape(num_tokens, self.num_local_kv_heads * self.head_dim)
        cos, sin, _freqs = position_embeddings
        print(f"freqs: {_freqs.shape}")
        # fused linghe kernel
        if self.sliding_window is not None:
            query_states, key_states, value_states = triton_qk_norm_and_half_rope_forward(
                out_qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                _freqs.contiguous(),
                self.num_local_heads,
                self.num_local_kv_heads,
                eps=1e-6,
            )
        else:
            query_states, key_states, value_states = triton_qk_norm_and_split_forward(
                out_qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                _freqs.contiguous(),
                self.num_local_heads,
                self.num_local_kv_heads,
                eps=1e-6,
            )
        # Returns:
        # - qo: shape [B, S, H, head_dim]
        # - ko: shape [B, S, h, head_dim]
        # - vo: shape [B, S, h, head_dim]
        # need in the non fused path shape
        n_heads = self.num_local_heads
        n_kv_heads = self.num_local_kv_heads
        query_states = query_states.reshape(num_tokens, n_heads, self.head_dim)
        key_states = key_states.reshape(num_tokens, n_kv_heads, self.head_dim)
        value_states = value_states.reshape(num_tokens, n_kv_heads, self.head_dim)
        
        q = query_states.reshape(num_tokens, n_heads * self.head_dim)
        k = key_states.reshape(num_tokens, n_kv_heads * self.head_dim)
        v = value_states.reshape(num_tokens, n_kv_heads * self.head_dim)
        
        attn_output = self.attn(q, k, v, forward_batch)

        attn_output, _ = self.o_proj(attn_output)
        if get_tensor_model_parallel_world_size() > 1:
            attn_output = tensor_model_parallel_all_reduce(attn_output.to(torch.float32)).to(hidden_states.dtype)
        return attn_output


# ── decoder layer ─────────────────────────────────────────────────────────────

class MapleDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = MapleAttention(config=config, layer_idx=layer_idx)

        if USE_SONIC_MOE:
            if moe_TC_softmax_topk_layer is None:
                raise ImportError(
                    "USE_SONIC_MOE=True requires the sonicmoe package. "
                    "Install from mono/repos/sonic-moe, or set USE_SONIC_MOE=False."
                )
            self.mlp = MapleSparseMoeBlockSonic(config, layer_id=layer_idx)
        elif USE_FUSED_EXPERTS:
            self.mlp = MapleSparseMoeBlock(config, layer_id=layer_idx)
        else:
            self.mlp = MapleSparseMoeBlockOld(config, layer_id=layer_idx)

        self.input_layernorm = MapleRMSNorm(config.hidden_size, eps=config.rms_norm_eps, _fp64=True)
        self.post_attention_layernorm = MapleRMSNorm(config.hidden_size, eps=config.rms_norm_eps, _fp64=True)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        initial_dtype = hidden_states.dtype
        residual = hidden_states
        # one-shot dtype probe: mirror image of the ve probe to confirm
        # whether input_layernorm.weight matches across sides.
        if not getattr(MapleDecoderLayer, "_dtype_logged", False) and self.layer_idx == 0:
            with open("/tmp/sgl_dtype_probe.log", "a") as _f:
                _f.write(
                    f"[sgl L0] input_layernorm.weight.dtype={self.input_layernorm.weight.dtype} "
                    f"post_attention_layernorm.weight.dtype={self.post_attention_layernorm.weight.dtype} "
                    f"hidden_states.dtype={hidden_states.dtype}\n"
                )
            MapleDecoderLayer._dtype_logged = True
        # Dumps + NaN checks at the same 6 tagged points per layer.
        _sgl_dump_weight(self.input_layernorm.weight, self.layer_idx, "input_layernorm_weight")
        _sgl_dump_hidden(hidden_states, self.layer_idx, "before_input_norm")
        _nan_check(hidden_states, self.layer_idx, "before_input_norm")

        hidden_states = self.input_layernorm(hidden_states)
        _sgl_dump_hidden(hidden_states, self.layer_idx, "after_input_norm")
        _nan_check(hidden_states, self.layer_idx, "after_input_norm")

        hidden_states = self.self_attn(positions, hidden_states, forward_batch, position_embeddings)
        _nan_check(hidden_states, self.layer_idx, "attn_out_only")
        hidden_states = (residual.to(torch.float32) + hidden_states.to(torch.float32)).to(initial_dtype)
        _sgl_dump_hidden(hidden_states, self.layer_idx, "after_attn")
        _nan_check(hidden_states, self.layer_idx, "after_attn")

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        _sgl_dump_hidden(hidden_states, self.layer_idx, "after_post_attn_norm")
        _nan_check(hidden_states, self.layer_idx, "after_post_attn_norm")

        hidden_states = self.mlp(hidden_states)
        # mlp output BEFORE the residual add — isolates MoE kernel vs add.
        _sgl_dump_hidden(hidden_states, self.layer_idx, "after_mlp_only")
        _nan_check(hidden_states, self.layer_idx, "after_mlp_only")

        hidden_states = (residual.to(torch.float32) + hidden_states.to(torch.float32)).to(initial_dtype)
        _sgl_dump_hidden(hidden_states, self.layer_idx, "after_moe")
        _nan_check(hidden_states, self.layer_idx, "after_moe")

        return hidden_states


# ── backbone ──────────────────────────────────────────────────────────────────

class MapleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # VocabParallelEmbedding is weight-compatible with nn.Embedding.
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

        # positions: (num_tokens,) → rotary_emb expects (B, T)
        position_embeddings = self.rotary_emb(hidden_states, positions.unsqueeze(0))

        for i, layer in enumerate(self.layers):
            # if i > 5: 
            #     break 
            
            hidden_states = layer(positions, hidden_states, forward_batch, position_embeddings)

        hidden_states = self.norm(hidden_states)
        return hidden_states


# ── fp32 lm_head matmul (mirrors VeOmni's MapleForCausalLM.forward exactly) ──

class _FP32MatmulLogitsProcessor(LogitsProcessor):
    """Force the lm_head matmul into fp32 unconditionally.

    Bypasses ``enable_fp32_lm_head``, ``rl_on_policy_target``,
    ``use_intel_amx_backend``, the bf16 fallback, and the logit_scale /
    final_logit_softcapping / DP-scatter tails. The resulting matmul is:

        local_logits = hidden_states.fp32 @ lm_head.weight.fp32.T
        logits       = all_gather(local_logits)  # only when TP>1

    which is bit-identical to ``MapleForCausalLM.forward``'s lm_head call
    in VeOmni's modeling_maple.py. At TP=1 the all-gather is a no-op; at
    TP>1 ``lm_head.weight`` is column-sharded along the vocab dim by
    ``ParallelLMHead``, so each rank computes a vocab-slice of the logits
    and we gather them along ``dim=-1`` to reconstitute the full
    ``[num_tokens, vocab_size]`` tensor before the buffer copy. NCCL
    preserves fp32 across the gather, so the fp32-parity property holds at
    any TP.
    """

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head,
        logits_metadata: LogitsMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Per-rank shard matmul. With ParallelLMHead column-sharded along
        # vocab, ``lm_head.weight`` is ``[vocab_size // TP, hidden]`` on each
        # rank — except possibly on the last rank if vocab_size doesn't
        # divide evenly, in which case ParallelLMHead pads internally.
        local_logits = torch.matmul(
            hidden_states.to(torch.float32),
            lm_head.weight.to(torch.float32).T,
        )
        if get_tensor_model_parallel_world_size() > 1:
            # All-gather along the vocab dim. Result shape:
            # ``[num_tokens, padded_vocab_size]`` where padded_vocab_size
            # equals ``(vocab_size // TP) * TP`` and is >= self.config.vocab_size.
            logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
        else:
            logits = local_logits
        if logits_metadata.next_token_logits_buffer is not None:
            buf = logits_metadata.next_token_logits_buffer
            assert buf.dtype == torch.float
            buf.copy_(logits[:, : self.config.vocab_size])
            return buf
        return logits[:, : self.config.vocab_size]


# ── causal LM head ────────────────────────────────────────────────────────────

class MapleForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.model = MapleModel(config)
        self.vocab_size = config.vocab_size

        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = _FP32MatmulLogitsProcessor(config)

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
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    # Regex for per-expert HF tensor names that we fuse into the sonic layout.
    _SONIC_EXPERT_RE = re.compile(
        r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
        r"(gate_proj|up_proj|down_proj)\.weight$"
    )

    def _load_sonic_expert(self, params_dict, name, loaded_weight) -> bool:
        """If `name` is a per-expert HF tensor, pack it into the sonic
        gate_up_proj / down_proj layout and return True. Otherwise return
        False so the caller can try the next loader path.
        """
        m = self._SONIC_EXPERT_RE.match(name)
        if m is None:
            return False
        layer = int(m.group(1))
        eid = int(m.group(2))
        proj = m.group(3)
        if proj == "down_proj":
            target = f"model.layers.{layer}.mlp.experts.down_proj"
            param = params_dict[target]
            # down_proj shape (E, H, I), loaded_weight shape (H, I).
            param.data[eid].copy_(loaded_weight)
        else:
            target = f"model.layers.{layer}.mlp.experts.gate_up_proj"
            param = params_dict[target]
            # gate_up_proj shape (E, 2*I, H): gate at 2*i, up at 2*i+1.
            # loaded_weight shape (I, H).
            offset = 0 if proj == "gate_proj" else 1
            param.data[eid, offset::2, :].copy_(loaded_weight)
        return True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Map HF-format separate q/k/v projections onto the fused QKVParallelLinear.
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]

        # FusedMoE mapping only used when USE_SONIC_MOE is False.
        expert_params_mapping = (
            []
            if USE_SONIC_MOE
            else FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_experts,
            )
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.endswith(".mlp.gate.expert_bias"):
                continue

            bailing_v2_mapping = (
                (".self_attn.qkv_proj", ".attention.query_key_value"),
                (".self_attn.o_proj", ".attention.dense"),
                (".self_attn.q_norm", ".attention.query_layernorm"),
                (".self_attn.k_norm", ".attention.key_layernorm"),
            )
            loaded_bailing_v2_weight = False
            for param_name, weight_name in bailing_v2_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_bailing_v2_weight = True
                break
            if loaded_bailing_v2_weight:
                continue

            # 1) qkv fusion
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                fused_name = name.replace(weight_name, param_name)
                if fused_name not in params_dict:
                    continue
                param = params_dict[fused_name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # 2) sonic expert fusion (only matches if USE_SONIC_MOE=True
                # since the target tensor names exist only in that case).
                if USE_SONIC_MOE and self._load_sonic_expert(
                    params_dict, name, loaded_weight
                ):
                    continue
                # 3) FusedMoE expert mapping (USE_SONIC_MOE=False path)
                for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped_name = name.replace(weight_name, param_name)
                    if mapped_name not in params_dict:
                        continue
                    param = params_dict[mapped_name]
                    param.weight_loader(
                        param, loaded_weight, mapped_name,
                        shard_id=shard_id, expert_id=expert_id,
                    )
                    break
                else:
                    # 4) fallthrough — plain copy via weight_loader.
                    if name not in params_dict:
                        logger.debug(f"Skipping {name} — not in model")
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)


EntryClass = MapleForCausalLM
