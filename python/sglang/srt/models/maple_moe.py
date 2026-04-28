# Adapted from qwen3_moe.py

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Inference-only Maple model compatible with HuggingFace weights."""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding, get_rope
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP as MapleMLP
from sglang.srt.models.qwen2_moe import Qwen2MoeModel
from sglang.srt.models.utils import (
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    add_prefix,
    is_cuda,
    is_flashinfer_available,
    is_non_idle_and_non_empty,
    make_layers
)

MapleConfig = None

_is_flashinfer_available = is_flashinfer_available()

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()

# ── debug tensor dumps ────────────────────────────────────────────────────────
import atexit as _atexit
_SGL_DBG   = {}
_SGL_STEP  = [0]
_SGL_LAYER = [0]

def _sgl_dbg(name, t):
    key = f"s{_SGL_STEP[0]:03d}_l{_SGL_LAYER[0]:02d}_{name}"
    _SGL_DBG[key] = t[-1].detach().cpu().to(torch.float32).clone()

def _sgl_save():
    torch.save(_SGL_DBG, "/tmp/sgl_debug.pt")
    print(f"[SGL debug] saved {len(_SGL_DBG)} tensors → /tmp/sgl_debug.pt")

_atexit.register(_sgl_save)
# ─────────────────────────────────────────────────────────────────────────────

# Unchanged from qwen3
class MapleSparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: MapleConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=True,  #config.norm_topk_prob,
            use_grouped_topk=False,
        )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts
            + get_global_server_args().ep_num_redundant_experts,
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        if get_moe_a2a_backend().is_deepep():
            # TODO: we will support tp < ep in the future
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.num_experts + get_global_server_args().ep_num_redundant_experts
            )
            self.top_k = config.num_experts_per_tok

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:

        if not get_moe_a2a_backend().is_deepep():
            return self.forward_normal(
                hidden_states, should_allreduce_fusion, use_reduce_scatter
            )
        else:
            return self.forward_deepep(hidden_states, forward_batch)

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        # Match HF reference: gate and routing entirely in fp32.
        # topk_softmax CUDA kernel only supports fp16/bf16, so bypass it with
        # an inline Python routing path (softmax → topk → renormalize) instead.
        router_logits_fp32 = torch.nn.functional.linear(
            hidden_states.to(torch.float32), self.gate.weight.to(torch.float32)
        )
        routing_weights = torch.softmax(router_logits_fp32, dim=-1)
        topk_weights, topk_ids = torch.topk(
            routing_weights, self.topk.topk_config.top_k, dim=-1
        )
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids.to(torch.int32),
            router_logits=router_logits_fp32,
        )
        # Use native sequential per-expert forward to match HF's moe_infer numerics.
        from sglang.srt.layers.moe.fused_moe_native import moe_forward_native
        final_hidden_states = moe_forward_native(
            self.experts,
            hidden_states,
            topk_output,
            self.experts.moe_runner_config,
        )
        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        return final_hidden_states

    def op_gate(self, state):
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):
            # router_logits: (num_tokens, n_experts)
            state.router_logits, _ = self.gate(state.hidden_states_mlp_input)
        else:
            state.router_logits = None

    def op_select_experts(self, state):
        router_logits = state.pop("router_logits")
        hidden_states = state.hidden_states_mlp_input
        if router_logits is not None:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.topk_output = self.topk(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    num_token_non_padded=state.forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    ),
                )
        else:
            state.topk_output = self.topk.empty_topk_output(hidden_states.device)

    def op_dispatch_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.dispatch_a(
                hidden_states=state.pop("hidden_states_mlp_input"),
                topk_output=state.pop("topk_output"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_dispatch_b(self, state):
        if self.ep_size > 1:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.dispatch_output = self.experts.dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )

    def op_experts(self, state):
        state.combine_input = self.experts.run_moe_core(
            dispatch_output=state.dispatch_output,
        )

    def op_combine_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.combine_a(
                combine_input=state.pop("combine_input"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )
            state.pop("dispatch_output")

    def op_combine_b(self, state):
        if self.ep_size > 1:
            state.hidden_states_after_combine = self.experts.dispatcher.combine_b(
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_output(self, state):
        state.hidden_states_mlp_output = state.pop("hidden_states_after_combine")

class MapleRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        input_dtype = hidden_states.dtype
        if residual is not None:
            # Match HF's explicit bf16 residual add (not fused fp32 add).
            new_residual = hidden_states + residual
            _sgl_dbg("post_residual", new_residual)  # final layer's post_residual
            hidden_states = new_residual
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        out = self.weight.to(input_dtype) * hidden_states.to(input_dtype)
        if residual is not None:
            return out, new_residual
        return out

# adapted from qwen3: added NoPe global + half RoPe SWA
class MapleAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        layer_type = "full_attention",
        sliding_window_size: int = -1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_type = layer_type
        use_sliding_window = layer_type == "sliding_attention"
        # print(f"layer {layer_id} sliding attention: {use_sliding_window}")
        self.use_rope = use_sliding_window
        self.sliding_window_size = sliding_window_size - 1 


        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        # main change is using rotary embed half dim
        self.use_sliding_window = use_sliding_window
        if use_sliding_window:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=int(self.head_dim*0.5),
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                dual_chunk_attention_config=dual_chunk_attention_config,
            )
        else:
            self.rotary_emb = None
        # self.compatible_with_fused_kv_buffer = (
        #     False if isinstance(self.rotary_emb, MRotaryEmbedding) else True
        # )
        # self.compatible_with_fused_kv_buffer = (
        #     self.rotary_emb is not None
        #     and not isinstance(self.rotary_emb, MRotaryEmbedding)
        # )
        self.compatible_with_fused_kv_buffer = False

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
            sliding_window_size=(self.sliding_window_size if use_sliding_window else -1),
        )

        self.q_norm = MapleRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = MapleRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.alt_stream = alt_stream

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # overlap qk norm
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            with torch.cuda.stream(self.alt_stream):
                k_by_head = k.reshape(-1, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            k_by_head = k.reshape(-1, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def op_prepare(self, state):
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
        )

    def op_core(self, state):
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        print(positions, self.sliding_window_size, self.use_sliding_window)
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        if self.rotary_emb is not None:
            # Use native path to match HF: casts cos/sin to query dtype (bf16)
            # before multiply, rather than using float32 cos_sin_cache directly.
            q, k = self.rotary_emb.forward_native(positions, q, k)
        # print(f"q: {q[..., :1]}")
        # print(f"k: {k[..., :1]}")
        # print(f"v: {v[..., :1]}")
        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        
        q, k, v, fb = inner_state
        if self.attn.layer_id == 0:
            print(f"[layer 0] q:{q.shape} seq_lens:{fb.seq_lens}")

        attn_output = self.attn(
            *inner_state,
            save_kv_cache=not (
                enable_fused_set_kv_buffer(forward_batch)
                and self.compatible_with_fused_kv_buffer
            ),
        )
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        return self.forward_core(s)


class MapleDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MapleConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rms_norm_eps = config.rms_norm_eps
        attention_bias = False # config.attention_bias
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        layer_type = config.layer_types[layer_id]
        sliding_window_size = config.sliding_window
        # TODO: wire in all of the args
        self.self_attn = MapleAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            dual_chunk_attention_config=dual_chunk_attention_config,
            alt_stream=alt_stream,
            layer_type=layer_type,
            sliding_window_size=sliding_window_size,
        )

        self.layer_id = layer_id

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        # Qwen3MoE all layers are sparse and have no nextn now
        self.is_layer_sparse = True
        is_previous_layer_sparse = True

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = MapleSparseMoeBlock(
                layer_id=self.layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = MapleMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(self.layer_id == self.config.num_hidden_layers - 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _SGL_LAYER[0] = self.layer_id
        # The communicator defers the residual add to the next layer's prepare_attn.
        # Capture it here so post_residual aligns with HF's layer N-1 post_residual.
        if residual is not None:
            _SGL_LAYER[0] = self.layer_id - 1
            _sgl_dbg("post_residual", hidden_states + residual.to(torch.float32))
            _SGL_LAYER[0] = self.layer_id

        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
            )
        )
        _sgl_dbg("post_input_norm", hidden_states)

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        _sgl_dbg("post_attn", hidden_states)

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        _sgl_dbg("post_post_attn_norm", hidden_states)

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        hidden_states = self.mlp(
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )
        _sgl_dbg("post_mlp", hidden_states)

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tbo_subbatch_index: Optional[int] = None,
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_mlp(self, state):
        hidden_states = state.pop("hidden_states_mlp_input")
        state.hidden_states_mlp_output = self.mlp(hidden_states, state.forward_batch)

    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "tbo_subbatch_index",
            }
        )
        return output


class MapleModel(nn.Module):
    def __init__(
        self,
        config: MapleConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type=MapleDecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config 
        
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to MapleDecoderLayer
        decoder_layer_type = decoder_layer_type or MapleDecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = MapleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []
        
    def set_eagle3_layers_to_capture(self, layers_to_capture: List[int]):
        self.layers_to_capture = layers_to_capture
        for layer_id in self.layers_to_capture:
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        if forward_batch.can_run_tbo:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers,
                enable_tbo=True,
                input_data_scatter_mode=ScatterMode.model_input_output(),
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:
            for i in range(self.start_layer, self.end_layer):
                ctx = (
                    nullcontext()
                    if get_global_server_args().enable_piecewise_cuda_graph
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    layer = self.layers[i]
                    hidden_states, residual = layer(
                        positions,
                        hidden_states,
                        forward_batch,
                        residual,
                        captured_last_layer_outputs=(
                            aux_hidden_states
                            if getattr(layer, "_is_layer_to_capture", False)
                            else None
                        ),
                    )
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class MapleForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: MapleConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.model = MapleModel(
            config, quant_config, prefix=add_prefix("model", prefix),
            alt_stream=alt_stream
        )
        # print(self.model)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        _SGL_LAYER[0] = 99
        _sgl_dbg("final_norm", hidden_states)

        if self.pp_group.is_last_rank:
            logits_out = self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
            # dump [num_selected_tokens, vocab] for KL analysis
            # LogitsProcessor returns LogitsProcessorOutput, not a raw Tensor
            _logits_t = getattr(logits_out, "next_token_logits", None)
            if _logits_t is not None:
                _SGL_DBG[f"s{_SGL_STEP[0]:03d}_logits"] = (
                    _logits_t.detach().cpu().to(torch.float32).clone()
                )
            _SGL_STEP[0] += 1
            torch.save(_SGL_DBG, "/tmp/sgl_debug.pt")
            return logits_out
        else:
            _SGL_STEP[0] += 1
            torch.save(_SGL_DBG, "/tmp/sgl_debug.pt")
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds

        # decoder layer
        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                forward_batch.hidden_states, forward_batch.residual = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.set_eagle3_layers_to_capture(
                [
                    2,
                    num_layers // 2,
                    num_layers - 3,
                ]
            )  # Specific layers for EAGLE3 support
        else:
            self.model.set_eagle3_layers_to_capture([val + 1 for val in layer_ids])

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        # Cache params_dict to avoid repeated expensive traversal of model parameters
        if not hasattr(self, "_cached_params_dict"):
            self._cached_params_dict = dict(self.named_parameters())
        params_dict = self._cached_params_dict
        for name, loaded_weight in weights:
            if "word_embeddings" in name:
                name = name.replace("word_embeddings", "embed_tokens")
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Mark as expert weight regardless of whether we can process it
                    is_expert_weight = True

                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        # Expert weight not on this rank, will be skipped below
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

        # TODO mimic deepseek
        # Lazy initialization of expert weights cache to avoid slowing down load_weights
        if not hasattr(self, "routed_experts_weights_of_layer"):
            self.routed_experts_weights_of_layer = {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.start_layer, self.end_layer)
                if isinstance(self.model.layers[layer_id].mlp, MapleSparseMoeBlock)
            }

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


EntryClass = MapleForCausalLM
