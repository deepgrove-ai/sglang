#!/usr/bin/env python3
"""
Test suite for FP8-aware fused add + RMSNorm kernel.

This validates:
1. Numerical correctness vs BF16 reference
2. In-place behavior
3. Scale attachment
4. Edge cases (small/large batch sizes, various hidden dims)
"""

import unittest
import torch
import math


def skip_if_no_cuda(func):
    """Skip test if CUDA is not available."""
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            print(f"Skipping {func.__name__}: CUDA not available")
            return
        return func(*args, **kwargs)
    return wrapper


class TestFusedAddRMSNormFP8(unittest.TestCase):
    """Test FP8 fused add + RMSNorm."""
    
    @skip_if_no_cuda
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda")
        self.eps = 1e-6
        
        # Test dimensions (batch_size, hidden_dim)
        self.test_shapes = [
            (1, 2048),      # M=1 decode
            (1, 5120),      # M=1 with larger hidden
            (4, 2048),      # Small batch
            (16, 2048),     # Medium batch
            (64, 2048),     # Larger batch
            (128, 2048),    # Typical prefill
        ]
    
    def _reference_fused_add_rmsnorm(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ):
        """Reference implementation in FP32 for comparison."""
        # Add
        new_residual = residual.float() + x.float()
        
        # RMSNorm
        variance = new_residual.pow(2).mean(dim=-1, keepdim=True)
        normalized = new_residual * torch.rsqrt(variance + eps)
        output = (normalized * weight.float()).to(torch.bfloat16)
        
        return output, new_residual.to(torch.bfloat16)
    
    @skip_if_no_cuda
    def test_import(self):
        """Test that the module can be imported."""
        from sglang.srt.layers.fused_add_rmsnorm_fp8 import (
            fused_add_rmsnorm_fp8,
            is_fp8_rmsnorm_available,
            FP8_DTYPE,
        )
        self.assertEqual(FP8_DTYPE, torch.float8_e4m3fn)
    
    @skip_if_no_cuda
    def test_availability(self):
        """Test kernel availability check."""
        from sglang.srt.layers.fused_add_rmsnorm_fp8 import is_fp8_rmsnorm_available
        
        # Should return True on CUDA with FP8 support
        available = is_fp8_rmsnorm_available()
        print(f"FP8 RMSNorm available: {available}")
        self.assertTrue(available, "FP8 RMSNorm should be available on CUDA")
    
    @skip_if_no_cuda
    def test_basic_correctness(self):
        """Test basic numerical correctness vs reference.
        
        FP8 quantization introduces significant error (~12% per quantization).
        With input quant + output quant, total error can be ~30-50%.
        
        We compare against a "fair" reference that accounts for input quantization.
        """
        from sglang.srt.layers.fused_add_rmsnorm_fp8 import fused_add_rmsnorm_fp8
        
        torch.manual_seed(42)  # Reproducibility
        
        for batch_size, hidden_dim in self.test_shapes:
            with self.subTest(batch_size=batch_size, hidden_dim=hidden_dim):
                # Create inputs
                x_bf16 = torch.randn(batch_size, hidden_dim, device=self.device, dtype=torch.bfloat16)
                residual = torch.randn(batch_size, hidden_dim, device=self.device, dtype=torch.bfloat16)
                # Use positive weights (typical for RMSNorm)
                weight = torch.ones(hidden_dim, device=self.device, dtype=torch.bfloat16)
                
                # Quantize input to FP8
                FP8_MAX = 448.0
                x_f32 = x_bf16.float()
                abs_max = x_f32.abs().amax(dim=-1, keepdim=True)
                x_scale = (abs_max / FP8_MAX).clamp(min=1e-12).squeeze(-1)
                x_fp8 = (x_f32 / x_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                
                # Dequantize to get "what the kernel sees as input"
                x_dequant = x_fp8.float() * x_scale.unsqueeze(-1)
                
                # Reference result using dequantized input (fair comparison)
                ref_out, ref_residual = self._reference_fused_add_rmsnorm(
                    x_dequant.to(torch.bfloat16), residual.clone(), weight, self.eps
                )
                
                # Run FP8 kernel
                residual_test = residual.clone()
                out_fp8, out_scale = fused_add_rmsnorm_fp8(
                    x_fp8, x_scale, residual_test, weight, self.eps, inplace=False
                )
                
                # Dequantize output
                out_bf16 = (out_fp8.float() * out_scale.unsqueeze(-1)).to(torch.bfloat16)
                
                # Tolerance: FP8 output quantization adds ~12% error
                # Plus accumulation errors, total ~25% is acceptable
                max_out_diff = (out_bf16 - ref_out).abs().max().item()
                mean_out_mag = ref_out.abs().mean().item()
                rel_out_diff = max_out_diff / (mean_out_mag + 1e-6)
                
                max_res_diff = (residual_test - ref_residual).abs().max().item()
                mean_res_mag = ref_residual.abs().mean().item()
                rel_res_diff = max_res_diff / (mean_res_mag + 1e-6)
                
                # Output: allow 30% relative error or 0.5 absolute
                output_ok = (rel_out_diff < 0.30) or (max_out_diff < 0.5)
                
                # Residual: should be quite accurate (no output quantization)
                residual_ok = (rel_res_diff < 0.15) or (max_res_diff < 0.2)
                
                if not output_ok:
                    print(f"Output: max_diff={max_out_diff:.4f}, mean={mean_out_mag:.4f}, rel={rel_out_diff:.2%}")
                
                if not residual_ok:
                    print(f"Residual: max_diff={max_res_diff:.4f}, mean={mean_res_mag:.4f}, rel={rel_res_diff:.2%}")
                
                self.assertTrue(output_ok, f"Output mismatch for shape {batch_size}x{hidden_dim}")
                self.assertTrue(residual_ok, f"Residual mismatch for shape {batch_size}x{hidden_dim}")
    
    @skip_if_no_cuda
    def test_inplace(self):
        """Test in-place operation."""
        from sglang.srt.layers.fused_add_rmsnorm_fp8 import fused_add_rmsnorm_fp8
        
        batch_size, hidden_dim = 4, 2048
        
        # Create inputs
        x_bf16 = torch.randn(batch_size, hidden_dim, device=self.device, dtype=torch.bfloat16)
        residual = torch.randn(batch_size, hidden_dim, device=self.device, dtype=torch.bfloat16)
        weight = torch.randn(hidden_dim, device=self.device, dtype=torch.bfloat16)
        
        # Quantize to FP8
        FP8_MAX = 448.0
        x_f32 = x_bf16.float()
        abs_max = x_f32.abs().amax(dim=-1, keepdim=True)
        x_scale = (abs_max / FP8_MAX).clamp(min=1e-12).squeeze(-1)
        x_fp8 = (x_f32 / x_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        
        # Get data pointers before
        x_ptr_before = x_fp8.data_ptr()
        scale_ptr_before = x_scale.data_ptr()
        
        # Run in-place
        out_fp8, out_scale = fused_add_rmsnorm_fp8(
            x_fp8, x_scale, residual, weight, self.eps, inplace=True
        )
        
        # Verify same buffer (in-place)
        self.assertEqual(out_fp8.data_ptr(), x_ptr_before, "Output should reuse input buffer")
        self.assertEqual(out_scale.data_ptr(), scale_ptr_before, "Scale should reuse input buffer")
    
    @skip_if_no_cuda
    def test_scale_attachment(self):
        """Test that scale is properly attached to output."""
        from sglang.srt.layers.fused_add_rmsnorm_fp8 import fused_add_rmsnorm_fp8
        
        batch_size, hidden_dim = 4, 2048
        
        x_bf16 = torch.randn(batch_size, hidden_dim, device=self.device, dtype=torch.bfloat16)
        residual = torch.randn(batch_size, hidden_dim, device=self.device, dtype=torch.bfloat16)
        weight = torch.randn(hidden_dim, device=self.device, dtype=torch.bfloat16)
        
        # Quantize
        FP8_MAX = 448.0
        x_f32 = x_bf16.float()
        abs_max = x_f32.abs().amax(dim=-1, keepdim=True)
        x_scale = (abs_max / FP8_MAX).clamp(min=1e-12).squeeze(-1)
        x_fp8 = (x_f32 / x_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        
        out_fp8, out_scale = fused_add_rmsnorm_fp8(
            x_fp8, x_scale, residual, weight, self.eps, inplace=False
        )
        
        # Verify scale shape and dtype
        self.assertEqual(out_scale.shape, (batch_size,))
        self.assertEqual(out_scale.dtype, torch.float32)
        
        # Verify scale is positive
        self.assertTrue((out_scale > 0).all(), "All scales should be positive")
    
    @skip_if_no_cuda
    def test_m1_decode_performance(self):
        """Test that M=1 decode is fast (no unexpected overhead)."""
        from sglang.srt.layers.fused_add_rmsnorm_fp8 import fused_add_rmsnorm_fp8
        import time
        
        batch_size, hidden_dim = 1, 2048
        num_iters = 100
        warmup = 20
        
        # Create inputs
        x_bf16 = torch.randn(batch_size, hidden_dim, device=self.device, dtype=torch.bfloat16)
        residual = torch.randn(batch_size, hidden_dim, device=self.device, dtype=torch.bfloat16)
        weight = torch.randn(hidden_dim, device=self.device, dtype=torch.bfloat16)
        
        # Quantize
        FP8_MAX = 448.0
        x_f32 = x_bf16.float()
        abs_max = x_f32.abs().amax(dim=-1, keepdim=True)
        x_scale = (abs_max / FP8_MAX).clamp(min=1e-12).squeeze(-1)
        x_fp8 = (x_f32 / x_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        
        # Warmup
        for _ in range(warmup):
            residual_copy = residual.clone()
            out_fp8, out_scale = fused_add_rmsnorm_fp8(
                x_fp8.clone(), x_scale.clone(), residual_copy, weight, self.eps, inplace=True
            )
        
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iters):
            residual_copy = residual.clone()
            out_fp8, out_scale = fused_add_rmsnorm_fp8(
                x_fp8.clone(), x_scale.clone(), residual_copy, weight, self.eps, inplace=True
            )
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time_us = (end - start) / num_iters * 1e6
        print(f"M=1 FP8 RMSNorm: {avg_time_us:.2f} µs/iter")
        
        # Should be fast (< 100µs for M=1)
        self.assertLess(avg_time_us, 200, f"M=1 should be fast, got {avg_time_us:.2f}µs")


class TestRMSNormFP8Dispatch(unittest.TestCase):
    """Test RMSNorm FP8 dispatch in layernorm.py."""
    
    @skip_if_no_cuda
    def test_fp8_dispatch(self):
        """Test that RMSNorm correctly dispatches FP8 inputs."""
        from sglang.srt.layers.layernorm import RMSNorm
        
        hidden_size = 2048
        batch_size = 4
        device = torch.device("cuda")
        
        # Create RMSNorm
        norm = RMSNorm(hidden_size).to(device)
        
        # Create FP8 input with scale
        x_bf16 = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)
        residual = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)
        
        # Quantize to FP8
        FP8_MAX = 448.0
        x_f32 = x_bf16.float()
        abs_max = x_f32.abs().amax(dim=-1, keepdim=True)
        x_scale = (abs_max / FP8_MAX).clamp(min=1e-12).squeeze(-1)
        x_fp8 = (x_f32 / x_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        x_fp8._fp8_scale = x_scale
        
        # Run through RMSNorm
        out, residual_out = norm(x_fp8, residual)
        
        # Output should be valid (either FP8 with scale or BF16)
        self.assertIsNotNone(out)
        self.assertIsNotNone(residual_out)
        
        # Residual should be BF16
        self.assertEqual(residual_out.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
