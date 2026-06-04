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
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

try:
    from slime.router_replay import sglang_capture as _slime_router_replay
except ImportError:  # pragma: no cover
    _slime_router_replay = None

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
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size, is_dp_attention_enabled
from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.utils import add_prefix


logger = logging.getLogger(__name__)

USE_SWIGLU_CLAMP = True
MAPLE_SWIGLU_CLAMP_LIMIT = 7.0

def _use_swiglu_clamp(config) -> bool:
    return bool(getattr(config, "use_swiglu_clamp", USE_SWIGLU_CLAMP))

def get_attention_sliding_window_size(config: PretrainedConfig):
    # Aligned with HF sliding window semantics (inclusive),
    # while SGLang attention backend expects exclusive window size.
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None or "sliding_attention" not in layer_types:
        return None
    if not hasattr(config, "sliding_window"):
        return None
    sliding_window = config.sliding_window
    if sliding_window is None:
        return None
    return sliding_window - 1

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


class MapleRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        compute_dtype = torch.float32
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


# ── attention (RadixAttention is the only inference-specific substitution) ────

class MapleAttention(nn.Module):
    def __init__(self, config, layer_id: int, prefix: str = ""):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

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

        self.layer_type = config.layer_types[layer_id] if hasattr(config, "layer_types") else None
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

        sliding_window_size = self.sliding_window - 1 if self.sliding_window is not None else -1
        self.attn = RadixAttention(
            num_heads=self.num_local_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            layer_id=layer_id,
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
        qkv = out_qkv.view(num_tokens, self.num_local_heads + 2 * self.num_local_kv_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_local_heads, self.num_local_kv_heads, self.num_local_kv_heads], dim=-2
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if self.sliding_window is not None and position_embeddings is not None:
            cos, sin, _ = position_embeddings
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

        q = query_states.reshape(num_tokens, self.num_local_heads * self.head_dim)
        k = key_states.reshape(num_tokens, self.num_local_kv_heads * self.head_dim)
        v = value_states.reshape(num_tokens, self.num_local_kv_heads * self.head_dim)

        attn_output = self.attn(q, k, v, forward_batch)

        attn_output, _ = self.o_proj(attn_output)
        if get_tensor_model_parallel_world_size() > 1:
            attn_output = tensor_model_parallel_all_reduce(attn_output.to(torch.float32)).to(hidden_states.dtype)
        return attn_output


# ── decoder layer ─────────────────────────────────────────────────────────────

class MapleDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id # this is used by sglang, so make sure that the name stays exactly the same!

        self.self_attn = MapleAttention(config=config, layer_id=layer_id)

        self.mlp = MapleSparseMoeBlock(config, layer_id=layer_id)

        self.input_layernorm = MapleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MapleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        initial_dtype = hidden_states.dtype
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(positions, hidden_states, forward_batch, position_embeddings)
        hidden_states = (residual.to(torch.float32) + hidden_states.to(torch.float32)).to(initial_dtype)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = (residual.to(torch.float32) + hidden_states.to(torch.float32)).to(initial_dtype)

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
            MapleDecoderLayer(config, layer_id=i)
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
            hidden_states = layer(positions, hidden_states, forward_batch, position_embeddings)

        hidden_states = self.norm(hidden_states)
        return hidden_states


# ── causal LM head ────────────────────────────────────────────────────────────

class MapleForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        assert not is_dp_attention_enabled(), "MapleForCausalLM does not support DP attention"
        self.config = config
        self.model = MapleModel(config)
        self.vocab_size = config.vocab_size

        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)
        self.logits_processor.use_fp32_lm_head = True

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.word_embeddings

    def get_attention_sliding_window_size(self):
        return get_attention_sliding_window_size(self.config)

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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Map HF-format separate q/k/v projections onto the fused QKVParallelLinear.
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
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
                # 2) FusedMoE expert mapping
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
                    # 3) fallthrough — plain copy via weight_loader.
                    if name not in params_dict:
                        logger.debug(f"Skipping {name} — not in model")
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)


EntryClass = MapleForCausalLM
