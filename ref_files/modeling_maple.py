# coding=utf-8
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from torch.library import triton_op, wrap_triton
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.utils import logging as hf_logging
from transformers.utils.import_utils import is_torch_fx_available

from .configuration_maple import MapleConfig
# from .fa3 import flash_attention_forward
from .simple_varlen_flash_attention import flash_attention_forward

logger = hf_logging.get_logger(__name__)
_CONFIG_FOR_DOC = "MapleConfig"

# ── debug tensor dumps ────────────────────────────────────────────────────────
import atexit as _atexit
_LOGIT_DEBUG = os.environ.get("LOGIT_DEBUG", "").lower() == "true"
_HF_DBG  = {}
_HF_STEP = [0]
_HF_LAYER = [0]

def _hf_dbg(name, t):
    if not _LOGIT_DEBUG:
        return
    key = f"s{_HF_STEP[0]:03d}_l{_HF_LAYER[0]:02d}_{name}"
    _HF_DBG[key] = t.reshape(-1, t.shape[-1])[-1].detach().cpu().to(torch.float32).clone()

def _hf_save():
    torch.save(_HF_DBG, "/tmp/hf_debug.pt")
    print(f"[HF debug] saved {len(_HF_DBG)} tensors → /tmp/hf_debug.pt")

if _LOGIT_DEBUG:
    _atexit.register(_hf_save)
# ─────────────────────────────────────────────────────────────────────────────


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx  # noqa: F401

@dataclass
class MapleOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    z_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None


class MoeV2ModelOutputWithPast(MoeModelOutputWithPast):
    def __init__(self, aux_loss=0.0, **kwargs):
        super().__init__(**kwargs)
        self.aux_loss = aux_loss


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
    def __init__(self, config: MapleConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, quantize_inplace_now: bool = False):
        gate_weight, up_weight, down_weight = self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight
        gate = F.linear(x, gate_weight)
        up = F.linear(x, up_weight)
        hidden = (self.act_fn(gate.float()) * up.float()).to(x.dtype)
        return F.linear(hidden, down_weight)


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
        res =  weight_float * hidden_states
        return res.to(input_dtype)


class MapleGate(nn.Module):

    def __init__(self, config: MapleConfig):
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
    """
    Unfused MoE block matching HF layout (ModuleList experts).
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self._setup_experts()
        self.gate = MapleGate(config)

    def _setup_experts(self):
        self.experts = nn.ModuleList(
            [
                MapleMLP(
                    config=self.config,
                    intermediate_size=self.config.moe_intermediate_size,
                )
                for _ in range(self.config.num_experts)
            ]
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        original_shape = hidden_states.shape
        identity = hidden_states

        bsz, seq_len, h = hidden_states.shape
        topk_idx, topk_weight, router_logits = self.gate(hidden_states)
        # _hf_dbg("router_logits", router_logits)
        # _hf_dbg("router_probs", topk_weight)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(bsz, seq_len, h)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(bsz, seq_len, h)

        # _hf_dbg("ffn_out", y.reshape(-1, y.shape[-1]))
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
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


class MapleAttention(nn.Module):
    """Fixed wiring for modern HF attention APIs: uses prepared causal_mask + cache_position + Cache.update()."""

    def __init__(self, config: MapleConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please pass `layer_idx`."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5

        partial_rotary_factor = 0.5 # getattr(config, "partial_rotary_factor", 0.5)
        self.rope_dim = int(self.head_dim * partial_rotary_factor)

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True

        self.sliding_window = None
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )

        self.q_norm = MapleRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MapleRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # IMPORTANT: pass prepared causal_mask here
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,  # IMPORTANT: needed for modern cache update
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        bsz, q_len, _ = hidden_states.size()
        qkv_weight = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        out_qkv = torch.nn.functional.linear(hidden_states, qkv_weight)
        cos, sin, _freqs = position_embeddings
        qkv = out_qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        # _hf_dbg("q_post_norm", query_states.permute(0, 2, 1, 3).reshape(bsz, q_len, -1))
        # _hf_dbg("k_post_norm", key_states.permute(0, 2, 1, 3).reshape(bsz, q_len, -1))
        if self.sliding_window is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            # _hf_dbg("q_post_rope", query_states.permute(0, 2, 1, 3).reshape(bsz, q_len, -1))
            # _hf_dbg("k_post_rope", key_states.permute(0, 2, 1, 3).reshape(bsz, q_len, -1))

        # dump q/k (post-norm + post-RoPE) and v (post-norm) for the current tokens only
        # _hf_dbg("q", query_states.permute(0, 2, 1, 3).reshape(bsz, q_len, -1))
        # _hf_dbg("k", key_states.permute(0, 2, 1, 3).reshape(bsz, q_len, -1))
        # _hf_dbg("v", value_states.permute(0, 2, 1, 3).reshape(bsz, q_len, -1))

        # ---- Modern Cache update wiring (DynamicCache / StaticCache compatible) ----
        if use_cache and past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # attn_probs: mean-over-heads softmax(QK^T/√dk) for the last query position.
        # key_states here includes the full KV cache (past + current).
        # SGL equivalent requires KV-pool access; added HF-side only for now.
        # with torch.no_grad():
        #     _scale  = self.head_dim ** -0.5
        #     _q_last = query_states[:, :, -1:, :]                              # [B, H, 1, D]
        #     _ngrp   = self.num_heads // self.num_key_value_heads
        #     _k_exp  = key_states.repeat_interleave(_ngrp, dim=1)              # [B, H, S, D]
        #     _scores = torch.matmul(_q_last.float(), _k_exp.float().transpose(-1, -2)) * _scale
        #     _aprobs = torch.softmax(_scores, dim=-1)[0, :, 0, :].mean(0)     # [S]
        # _hf_dbg("attn_probs", _aprobs.unsqueeze(0))  # stored as [S] (grows each step)

        # Clip attention_mask to the actual KV length so _upad_input's indices
        # never exceed batch_size * kv_seq_len (can diverge during long generation).
        if attention_mask is not None:
            attention_mask = attention_mask[:, -key_states.shape[-2]:]

        attn_output, attn_weights = flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,  # prepared causal mask (or None for varlen flash path)
            dropout=0.0,
            position_ids=position_ids,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # keep your prototype behavior
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        # _hf_dbg("attn_pre_o", attn_output)
        dense_weight = self.o_proj.weight
        attn_output = torch.nn.functional.linear(attn_output, dense_weight)
        # _hf_dbg("o_proj", attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MapleDecoderLayer(nn.Module):
    def __init__(self, config: MapleConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = MapleAttention(config=config, layer_idx=layer_idx)

        self.mlp = MapleSparseMoeBlock(config)

        self.input_layernorm = MapleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MapleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # prepared causal mask
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,  # your MOE doesn't return router logits; kept for API
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Cache],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        initial_dtype = hidden_states.dtype
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # _hf_dbg("post_input_norm", hidden_states)

        attn_out, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=bool(output_attentions),
            use_cache=bool(use_cache),
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # _hf_dbg("post_attn", attn_out)
        hidden_states = (residual.to(torch.float32) + attn_out.to(torch.float32)).to(initial_dtype)
        # hidden_states = residual + attn_out
        # _hf_dbg("post_attn_residual", hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # _hf_dbg("post_post_attn_norm", hidden_states)

        mlp_out = self.mlp(hidden_states)
        if isinstance(mlp_out, tuple):
            hidden_states, aux_loss = mlp_out
        else:
            hidden_states, aux_loss = mlp_out, 0.0
        # _hf_dbg("post_mlp", hidden_states)

        hidden_states = (residual.to(torch.float32) + hidden_states.to(torch.float32)).to(initial_dtype)
        # hidden_states = residual + hidden_states.to(residual.device)
        # _hf_dbg("post_ffn_residual", hidden_states)
        router_logits = None

        return (
            hidden_states,
            self_attn_weights,
            present_key_value,
            aux_loss,
            router_logits,
        )


@add_start_docstrings(
    "The bare Maple Model outputting raw hidden-states without any specific head on top.",
)
class MaplePreTrainedModel(PreTrainedModel):
    config_class = MapleConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MapleDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_attention_backend = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@add_start_docstrings(
    "The bare Maple Model outputting raw hidden-states without any specific head on top.",
)
class MapleModel(MaplePreTrainedModel):
    def __init__(self, config: MapleConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        layers = []
        for layer_idx in range(config.num_hidden_layers):
            layers.append(MapleDecoderLayer(config, layer_idx))
        self.layers = nn.ModuleList(layers)
        self.config = config

        self.norm = MapleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MapleRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self._cache_debug_calls = 0
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    def prepare_fa2_from_position_ids(self, position_ids: torch.Tensor):
        position_ids = position_ids.flatten()
        T = position_ids.numel()
        indices_q = torch.arange(T, device=position_ids.device, dtype=torch.int32)

        starts = indices_q[position_ids == 0]

        # If no segment-start markers exist (common in decoding where pos ids are offset),
        # treat as a single sequence.
        if starts.numel() == 0:
            cu_seq_lens = torch.tensor([0, T], device=position_ids.device, dtype=torch.int32)
        else:
            # ensure boundaries valid
            if starts[0].item() != 0:
                starts = torch.cat([starts.new_zeros(1), starts], dim=0)
            if starts[-1].item() != T:
                starts = torch.cat([starts, starts.new_tensor([T])], dim=0)
            cu_seq_lens = starts

        max_length = (cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item()
        return (indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length))

    # def prepare_fa2_from_position_ids(self, position_ids: torch.Tensor):
    #     position_ids = position_ids.flatten()
    #     indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)

    #     cu_seq_lens = torch.cat(
    #         (
    #             indices_q[position_ids == 0],
    #             torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
    #         )
    #     )

    #     # max_length在不同的model里面type不同
    #     # modeling_qwen3_moe_foundation/modeling_qwen2_5_omni里为tensor
    #     # modeling_qwen2_vl的为int
    #     # 此处采用有.item()的写法，在decoder layers之前拿到int type的max_length
    #     # 否则在decoder里面仍然每一层都会触发.item()
    #     max_length = cu_seq_lens.diff().max().item()

    #     return (indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,  # 2D padding mask (B, S) coming in
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, MoeV2ModelOutputWithPast]:
        debug_cache = bool(kwargs.pop("debug_cache", False))
        print("on", position_ids)
        if debug_cache:
            print(f"Debug cache enabled for call {self._cache_debug_calls}")
        debug_call_id = self._cache_debug_calls
        self._cache_debug_calls += 1

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # exactly one of input_ids / inputs_embeds
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        _HF_LAYER[0] = 0
        # _hf_dbg("embed", inputs_embeds)

        # SGLang transformers backend passes `forward_batch`; use it to identify
        # decode mode (can have S>1 tokens due to token packing) and avoid
        # decode-only dynamic metadata that harms CUDA graph capture.
        forward_batch = kwargs.get("forward_batch", None)
        is_decode_step = False
        forward_mode = getattr(forward_batch, "forward_mode", None) if forward_batch is not None else None
        if forward_mode is not None:
            for mode_name in (
                "is_decode",
                "is_decode_or_idle",
                "is_target_verify",
                "is_draft_decode",
            ):
                mode_fn = getattr(forward_mode, mode_name, None)
                if callable(mode_fn) and bool(mode_fn()):
                    is_decode_step = True
                    break

        # Perform one-time in-place weight quantization during prefill (S > 1),

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is not None:
            # For bsh cases, expand [1, S] position_ids to [B, S] before FA2 metadata prep.
            batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            if position_ids.shape[0] != batch_size:
                position_ids = position_ids.expand(batch_size, -1)

            # Decode does not need cu_seq_lens/max_length metadata and creating
            # them every step hurts CUDA graph capture stability.
            if (not is_decode_step) and inputs_embeds.shape[1] > 1:
                _, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = self.prepare_fa2_from_position_ids(
                    position_ids
                )
                kwargs["cu_seq_lens_q"] = cu_seq_lens_q
                kwargs["cu_seq_lens_k"] = cu_seq_lens_k
                kwargs["max_length_q"] = max_length_q
                kwargs["max_length_k"] = max_length_k

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # IMPORTANT: build prepared causal_mask and pass it into layers (NOT raw attention_mask)
        # mask_function = create_causal_mask  # swap to create_sliding_window_causal_mask if you enable sliding window
        # causal_mask = mask_function(
        #     config=self.config,
        #     input_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     cache_position=cache_position,
        #     past_key_values=past_key_values,
        #     position_ids=position_ids,
        # )
        causal_mask = attention_mask
        # _mask_dbg = attention_mask.float().reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else torch.zeros(1, 1, device=inputs_embeds.device)
        # _hf_dbg("causal_mask", _mask_dbg)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # _hf_dbg("rope_cos", position_embeddings[0].reshape(-1, position_embeddings[0].shape[-1]))
        # _hf_dbg("rope_sin", position_embeddings[1].reshape(-1, position_embeddings[1].shape[-1]))

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        aux_loss_sum = 0.0

        for idx, decoder_layer in enumerate(self.layers):
            _HF_LAYER[0] = idx
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    **kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,  # <-- FIXED
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,  # <-- FIXED
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # aux loss is at index 3 in our layer return
            aux_loss_sum = aux_loss_sum + layer_outputs[3]

            if output_router_logits:
                all_router_logits += (layer_outputs[4],)

        hidden_states = self.norm(hidden_states)
        _HF_LAYER[0] = 99
        # _hf_dbg("final_norm", hidden_states)
        _HF_STEP[0] += 1

        if debug_cache:
            past_after_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_start = int(cache_position[0].item()) if cache_position.numel() > 0 else -1
            cache_end = int(cache_position[-1].item()) if cache_position.numel() > 0 else -1
            cache_hit = bool(use_cache and inputs_embeds.shape[1] == 1 and past_seen_tokens > 0)
            print(
                "[cache-debug] "
                f"call={debug_call_id} use_cache={use_cache} "
                f"input_len={inputs_embeds.shape[1]} "
                f"past_before={past_seen_tokens} past_after={past_after_tokens} "
                f"cache_pos=[{cache_start},{cache_end}] "
                f"cache_hit_expected={cache_hit}"
            )

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        moe_layer_count = len(self.layers) - 1
        out = MoeV2ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
            aux_loss=aux_loss_sum / moe_layer_count,  # keeping your prototype behavior
        )
        return (
            out
            if return_dict
            else (
                out.last_hidden_state,
                out.past_key_values,
                out.hidden_states,
                out.attentions,
            )
        )


class MapleForCausalLM(MaplePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MapleConfig):
        super().__init__(config)
        self.model = MapleModel(config)
        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = 0.001
        self.post_init()

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # @add_start_docstrings_to_model_forward(BAILINGMOEV2_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=MoEV2CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, MapleOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=True,  # ensure attribute access
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        assert isinstance(hidden_states, torch.Tensor)

        # slice logits if requested
        loss = None
        logits = None
        if labels is not None:
            loss, logits = self.loss_function(hidden_states, self.lm_head.weight, labels)
        else:
            logits = self.lm_head(hidden_states)
        # dump full [seq_len, vocab] logit matrix for KL analysis (step already bumped)
        if _LOGIT_DEBUG:
            _step = _HF_STEP[0] - 1
            _logits_2d = logits.reshape(-1, logits.shape[-1]).detach().cpu().to(torch.float32)
            _HF_DBG[f"s{_step:03d}_output_logits"] = _logits_2d.clone()
            _HF_DBG[f"s{_step:03d}_output_probs"] = torch.softmax(_logits_2d[-1], dim=-1).clone()
        out = MapleOutputWithPast(
            loss=loss,
            aux_loss=getattr(outputs, "aux_loss", 0.0),
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
            router_logits=outputs.router_logits if hasattr(outputs, "router_logits") else None,
        )
        return out

