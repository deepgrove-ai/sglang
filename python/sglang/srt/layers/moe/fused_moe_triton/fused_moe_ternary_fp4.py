"""
Ternary MoE via FP4 tensor cores.

This module converts ternary weights ({-1, 0, +1}) to FP4 E2M1 format and
uses the existing CUTLASS FP4 MoE kernel for native tensor-core acceleration.

Key insight:
- FP4 E2M1 values: [0, 0.5, 1, 1.5, 2, 3, 4, 6] (+ negatives)
- Ternary values {-1, 0, +1} map directly to FP4 encodings:
  - +1 → 0b0010 (sign=0, ord=2)
  - -1 → 0b1010 (sign=1, ord=2)
  -  0 → 0b0000 (sign=0, ord=0)

This avoids unpacking overhead and leverages native FP4 tensor cores on Blackwell.

Usage:
- FP4 weights are pre-converted at model load time (in ternary.py)
- The converted weights are stored on the layer to avoid runtime conversion
- Enable via TERNARY_MOE_FP4_MAX_TOKENS=64 environment variable
"""

from __future__ import annotations

import torch


def unpack_ternary_to_fp4(
    ternary_packed: torch.Tensor,
    K_unpacked: int,
) -> torch.Tensor:
    """
    Convert ternary packed weights to FP4 packed format.
    
    Args:
        ternary_packed: [E, N, K//4] uint8 (4 ternary values per byte, 2-bit each)
        K_unpacked: Original K dimension before ternary packing
        
    Returns:
        fp4_packed: [E, N, K//2] uint8 (2 FP4 values per byte, 4-bit each)
    """
    E, N, K_packed = ternary_packed.shape
    assert K_packed * 4 >= K_unpacked, f"K_packed={K_packed}*4 < K_unpacked={K_unpacked}"
    
    device = ternary_packed.device
    
    # Unpack ternary: each byte has 4 x 2-bit values
    # Encoding: 00=-1, 01=0, 10=+1
    # Extract each 2-bit lane
    lane0 = (ternary_packed & 0x03)        # bits 0-1
    lane1 = (ternary_packed >> 2) & 0x03   # bits 2-3
    lane2 = (ternary_packed >> 4) & 0x03   # bits 4-5
    lane3 = (ternary_packed >> 6) & 0x03   # bits 6-7
    
    # Convert ternary encoding (0=-1, 1=0, 2=+1) to FP4 encoding
    # FP4: +1=0b0010, -1=0b1010, 0=0b0000
    def ternary_to_fp4_nibble(t):
        # t: 0 → -1 → FP4=0b1010=10
        # t: 1 → 0  → FP4=0b0000=0
        # t: 2 → +1 → FP4=0b0010=2
        fp4 = torch.zeros_like(t, dtype=torch.uint8)
        fp4 = torch.where(t == 0, torch.tensor(10, dtype=torch.uint8, device=device), fp4)  # -1
        fp4 = torch.where(t == 1, torch.tensor(0, dtype=torch.uint8, device=device), fp4)   # 0
        fp4 = torch.where(t == 2, torch.tensor(2, dtype=torch.uint8, device=device), fp4)   # +1
        return fp4
    
    fp4_0 = ternary_to_fp4_nibble(lane0)
    fp4_1 = ternary_to_fp4_nibble(lane1)
    fp4_2 = ternary_to_fp4_nibble(lane2)
    fp4_3 = ternary_to_fp4_nibble(lane3)
    
    # Pack FP4: 2 values per byte (low nibble = even index, high nibble = odd index)
    # From 4 FP4 values, create 2 bytes
    byte0 = fp4_0 | (fp4_1 << 4)  # values 0,1
    byte1 = fp4_2 | (fp4_3 << 4)  # values 2,3
    
    # Interleave to get [E, N, K//2]
    # Stack along last dim and reshape
    fp4_packed = torch.stack([byte0, byte1], dim=-1).reshape(E, N, K_packed * 2)
    
    # Trim to exact K//2 if needed
    K_fp4 = (K_unpacked + 1) // 2
    if fp4_packed.shape[-1] > K_fp4:
        fp4_packed = fp4_packed[..., :K_fp4]
    
    return fp4_packed.contiguous()


def create_fp4_blockscales_from_ternary_alpha(
    alpha: torch.Tensor,
    N: int,
    K: int,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create FP4 block scales and per-expert alphas from ternary alpha table.
    
    The CUTLASS FP4 MoE uses:
    - w_blockscale: [E, N, K // block_size] as float8_e4m3fn
    - w_alphas: [E] as float32 (per-tensor scale per expert)
    
    For ternary, alpha is [E, K] (per expert, per K dimension).
    We fold the K-variation into blockscales and use mean as per-expert alpha.
    
    Args:
        alpha: [E, K] per-expert, per-K-dim scale factors
        N: Output dimension
        K: Input dimension (unpacked)
        block_size: FP4 block size (typically 16)
        
    Returns:
        blockscales: [E, N, K // block_size] float8_e4m3fn
        alphas: [E] float32
    """
    E = alpha.shape[0]
    device = alpha.device
    
    # Per-expert alpha: mean across K dimension
    per_expert_alpha = alpha.mean(dim=1)  # [E]
    
    # Blockscales: need to broadcast to [E, N, K // block_size]
    # Compute per-block scale as mean within each block
    K_blocks = (K + block_size - 1) // block_size
    
    # Pad alpha to multiple of block_size
    if K % block_size != 0:
        pad = block_size - (K % block_size)
        alpha_padded = torch.nn.functional.pad(alpha, (0, pad), value=1.0)
    else:
        alpha_padded = alpha
    
    # Reshape to [E, K_blocks, block_size] and compute mean per block
    alpha_blocks = alpha_padded.view(E, K_blocks, block_size).mean(dim=2)  # [E, K_blocks]
    
    # Normalize by per-expert alpha to get blockscale adjustment
    block_adjustment = alpha_blocks / per_expert_alpha.unsqueeze(1).clamp(min=1e-8)  # [E, K_blocks]
    
    # Broadcast to [E, N, K_blocks]
    blockscales = block_adjustment.unsqueeze(1).expand(E, N, K_blocks)  # [E, N, K_blocks]
    
    # Convert to float8_e4m3fn
    # Clamp to valid range and convert
    blockscales = blockscales.clamp(min=1e-4, max=448.0)  # FP8 E4M3 range
    blockscales = blockscales.to(torch.float8_e4m3fn)
    
    return blockscales.contiguous(), per_expert_alpha.to(torch.float32).contiguous()
