"""
Klear MoE model - Uses the Qwen3 MoE implementation as the base
since architectures are nearly identical.

Key differences from Qwen3 MoE:
- Uses sigmoid routing instead of softmax (handled in ternary quantization)
- Has routed_scaling_factor
- May have n_shared_experts (not used in this pruned version)
"""

from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM


class KlearMoeForCausalLM(Qwen3MoeForCausalLM):
    """
    Klear MoE model for causal language modeling.
    
    Uses the Qwen3 MoE implementation since the architectures are nearly identical:
    - Same attention mechanism (GQA with RoPE)
    - Same MoE structure (gated experts with top-k routing)
    - Same layer norm (RMSNorm)
    - Same activation (SiLU)
    
    Differences handled:
    - routed_scaling_factor: Applied during routing (in FusedMoE)
    - n_shared_experts: Not used in this pruned model
    
    The config is loaded by transformers using the auto_map in config.json,
    which points to configuration_klear.KlearConfig. That config is compatible
    with Qwen3MoeConfig since both have the same MoE-related fields.
    """
    pass


# Register the model
EntryClass = KlearMoeForCausalLM
