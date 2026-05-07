from typing import Optional, Tuple

import torch

import inspect
import os
from typing import Optional, TypedDict

import torch
import torch.nn.functional as F


try:
    from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
except:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input


# Detect supported kwargs in FA3
_sig = inspect.signature(flash_attn_func)
_flash_supports_window_size = "window_size" in _sig.parameters
_flash_accepts_deterministic = "deterministic" in _sig.parameters
_flash_accepts_softcap = "softcap" in _sig.parameters


def _get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def _upad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )

    if query_length == kv_seq_len:
        query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, -1, head_dim), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, *_ = unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def prepare_fa3_from_position_ids(query, key, value, position_ids):
    query = query.view(-1, query.size(-2), query.size(-1))
    key = key.contiguous().view(-1, key.size(-2), key.size(-1))
    value = value.contiguous().view(-1, value.size(-2), value.size(-1))
    position_ids = position_ids.flatten()
    indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)

    cu_seq_lens = torch.cat(
        (
            indices_q[position_ids == 0],
            torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
        )
    )

    max_length = position_ids.max() + 1
    return query, key, value, indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length)


def fa_peft_integration_check(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    target_dtype: Optional[torch.dtype] = None,
):
    if target_dtype is None:
        return query, key, value
    if query.dtype == torch.float32:
        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)
    return query, key, value


deterministic_g = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    causal = is_causal if not use_top_left_mask else (is_causal and query_length != 1)

    flash_kwargs = {}
    if _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window:
        flash_kwargs["window_size"] = (sliding_window, 0)
    if _flash_accepts_deterministic:
        if deterministic is None:
            deterministic = deterministic_g
        flash_kwargs["deterministic"] = deterministic

    if attention_mask is not None:
        batch_size = query_states.shape[0]
        q_unpad, k_unpad, v_unpad, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_q, max_k) = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )
        attn_output_unpad = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            # dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    elif position_ids is not None and (
        max_length_q is not None
        # This fucks up compile
        # or (query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all())
    ):
        batch_size = query_states.size(0)
        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            q_unpad, k_unpad, v_unpad, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = (
                prepare_fa3_from_position_ids(query_states, key_states, value_states, position_ids)
            )
        else:
            q_unpad = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
            k_unpad = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
            v_unpad = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))

        attn_output = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            # dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))

    else:
        # print(f"scale {softmax_scale}")
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            # dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

    return attn_output


class FlashAttentionKwargs(TypedDict, total=False):
    cu_seq_lens_q: Optional[torch.LongTensor]
    cu_seq_lens_k: Optional[torch.LongTensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]


# _use_top_left_mask = flash_attn_supports_top_left_mask()

_use_top_left_mask = False


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    seq_len = query.shape[1]

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=module.is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        target_dtype=target_dtype,
        **kwargs,
    )

    return attn_output, None
