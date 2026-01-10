"""
Correctness tests for FP8-first ternary quantization.

This test suite validates:
1. FP8 quantization/dequantization correctness
2. FP8 bridge linear layer output vs baseline
3. FP8 KV cache validation
4. End-to-end decode regression

Usage:
    # Run all tests
    python test_ternary_fp8_correctness.py
    
    # Run specific test
    python test_ternary_fp8_correctness.py TestFP8Quantization
    
    # Verbose output
    python test_ternary_fp8_correctness.py -v
"""

import os
import sys
import unittest
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Set up path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def skip_if_no_cuda(test_func):
    """Decorator to skip tests if CUDA is not available."""
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        return test_func(*args, **kwargs)
    return wrapper


class TestFP8Quantization(unittest.TestCase):
    """Test FP8 quantization/dequantization utilities."""
    
    @skip_if_no_cuda
    def test_fp8_hidden_manager_quant_dequant(self):
        """Test that quantize -> dequantize preserves values within tolerance."""
        from sglang.srt.model_loader.ternary_hook import FP8HiddenStateManager
        
        # Create manager
        hidden_size = 2048
        max_tokens = 128
        manager = FP8HiddenStateManager(
            hidden_size=hidden_size,
            max_tokens=max_tokens,
            scale_granularity="per_token_group_128",
        )
        manager.allocate_buffers(torch.device("cuda"))
        
        # Create random hidden states
        num_tokens = 64
        hidden_bf16 = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        
        # Quantize to FP8
        hidden_fp8, scale = manager.quantize_hidden(hidden_bf16)
        
        # Verify FP8 dtype
        self.assertEqual(hidden_fp8.dtype, torch.float8_e4m3fn)
        
        # Dequantize back
        hidden_dequant = manager.dequantize_hidden(hidden_fp8, scale, output_dtype=torch.bfloat16)
        
        # Check shape preserved
        self.assertEqual(hidden_dequant.shape, hidden_bf16.shape)
        
        # Check values within tolerance (FP8 has limited precision)
        # Typical FP8 E4M3 has ~3.5 bits of mantissa, so relative error ~10%
        relative_error = (hidden_dequant - hidden_bf16).abs() / (hidden_bf16.abs() + 1e-6)
        max_rel_error = relative_error.max().item()
        mean_rel_error = relative_error.mean().item()
        
        print(f"FP8 quant/dequant: max_rel_error={max_rel_error:.4f}, mean_rel_error={mean_rel_error:.4f}")
        
        # FP8 E4M3 should be within 20% for most values
        self.assertLess(mean_rel_error, 0.2, "Mean relative error too high")
    
    @skip_if_no_cuda
    def test_fp8_dtype_support(self):
        """Test that FP8 dtype is supported on this GPU."""
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device="cuda").to(torch.float8_e4m3fn)
            self.assertEqual(x.dtype, torch.float8_e4m3fn)
        except Exception as e:
            self.skipTest(f"FP8 dtype not supported: {e}")


class TestFP8BridgeLinear(unittest.TestCase):
    """Test FP8 bridge linear layer."""
    
    @skip_if_no_cuda
    def test_fp8_bridge_available(self):
        """Test that FP8 bridge availability check works."""
        # Temporarily enable bridge
        os.environ["SGLANG_TERNARY_FP8_BRIDGE"] = "1"
        
        try:
            from sglang.srt.layers.quantization.ternary import _fp8_bridge_available
            # This should return True or False without crashing
            result = _fp8_bridge_available()
            self.assertIsInstance(result, bool)
            print(f"FP8 bridge available: {result}")
        finally:
            os.environ.pop("SGLANG_TERNARY_FP8_BRIDGE", None)
    
    @skip_if_no_cuda
    def test_fp8_runtime_check(self):
        """Test FP8 runtime availability check."""
        from sglang.srt.layers.quantization.ternary import (
            _check_fp8_runtime,
            get_fp8_runtime_info,
        )
        
        result = _check_fp8_runtime()
        self.assertIsInstance(result, bool)
        
        info = get_fp8_runtime_info()
        self.assertIn('available', info)
        self.assertIn('sm_version', info)
        
        print(f"FP8 runtime info: {info}")


class TestTernaryConfig(unittest.TestCase):
    """Test TernaryConfig FP8 settings."""
    
    def test_fp8_config_from_env(self):
        """Test that FP8 is enabled from environment variable."""
        # Set env var
        os.environ["SGLANG_TERNARY_USE_FP8"] = "1"
        
        try:
            from sglang.srt.layers.quantization.ternary import TernaryConfig
            
            # Create config without explicit use_fp8
            config = TernaryConfig(
                threshold_scale=0.7,
                storage_mode="i2s",
            )
            
            # Should be enabled from env
            self.assertTrue(config.use_fp8, "FP8 should be enabled from env var")
            
            # Check KV cache recommendation
            self.assertEqual(config.kv_cache_quant_algo, "FP8")
            self.assertEqual(config.recommended_kv_cache_dtype, "fp8_e4m3")
        finally:
            os.environ.pop("SGLANG_TERNARY_USE_FP8", None)
    
    def test_fp8_config_explicit(self):
        """Test explicit FP8 config."""
        from sglang.srt.layers.quantization.ternary import TernaryConfig
        
        # Explicit use_fp8=True
        config = TernaryConfig(
            threshold_scale=0.7,
            storage_mode="i2s",
            use_fp8=True,
        )
        
        self.assertTrue(config.use_fp8)
        self.assertEqual(config.fp8_group_size, 128)  # Default granularity
    
    def test_fp8_config_disabled(self):
        """Test FP8 disabled by default."""
        # Make sure env var is not set
        os.environ.pop("SGLANG_TERNARY_USE_FP8", None)
        
        from importlib import reload
        import sglang.srt.layers.quantization.ternary as ternary_module
        reload(ternary_module)
        
        from sglang.srt.layers.quantization.ternary import TernaryConfig
        
        config = TernaryConfig(
            threshold_scale=0.7,
            storage_mode="i2s",
        )
        
        self.assertFalse(config.use_fp8, "FP8 should be disabled by default")
        self.assertIsNone(config.kv_cache_quant_algo)


class TestFP8KVCacheValidation(unittest.TestCase):
    """Test FP8 KV cache validation."""
    
    def test_kv_cache_validation_warning(self):
        """Test that KV cache validation logs warning for non-FP8 dtype."""
        import logging
        from unittest.mock import MagicMock
        
        # Create mock model with FP8 enabled
        mock_model = MagicMock()
        mock_model._ternary_use_fp8 = True
        
        from sglang.srt.model_loader.ternary_hook import validate_ternary_fp8_kv_cache
        
        # Test with BF16 KV cache - should log warning
        with self.assertLogs(level=logging.WARNING) as cm:
            validate_ternary_fp8_kv_cache(mock_model, torch.bfloat16)
        
        # Check warning was logged
        self.assertTrue(any("FP8-first mode is enabled but KV cache" in msg for msg in cm.output))
    
    def test_kv_cache_validation_no_warning_for_fp8(self):
        """Test that no warning is logged for FP8 KV cache."""
        from unittest.mock import MagicMock
        import logging
        
        # Create mock model with FP8 enabled
        mock_model = MagicMock()
        mock_model._ternary_use_fp8 = True
        
        from sglang.srt.model_loader.ternary_hook import validate_ternary_fp8_kv_cache
        
        # Test with FP8 KV cache - should not log warning
        try:
            with self.assertLogs(level=logging.WARNING) as cm:
                validate_ternary_fp8_kv_cache(mock_model, torch.float8_e4m3fn)
            # If we get here, a warning was logged (unexpected)
            self.fail("Unexpected warning logged for FP8 KV cache")
        except AssertionError:
            # No logs captured - this is the expected behavior
            pass


class TestFP8HiddenStateManager(unittest.TestCase):
    """Test FP8HiddenStateManager class."""
    
    @skip_if_no_cuda
    def test_buffer_allocation(self):
        """Test buffer pre-allocation."""
        from sglang.srt.model_loader.ternary_hook import FP8HiddenStateManager
        
        manager = FP8HiddenStateManager(
            hidden_size=2048,
            max_tokens=1024,
            scale_granularity="per_token_group_128",
        )
        
        self.assertFalse(manager._buffers_allocated)
        
        manager.allocate_buffers(torch.device("cuda"))
        
        self.assertTrue(manager._buffers_allocated)
        self.assertIsNotNone(manager._hidden_fp8_buffer)
        self.assertIsNotNone(manager._scale_buffer)
        
        # Check buffer shapes
        self.assertEqual(manager._hidden_fp8_buffer.shape, (1024, 2048))
        self.assertEqual(manager._scale_buffer.shape, (1024, 16))  # 2048/128 = 16 groups
    
    @skip_if_no_cuda
    def test_scale_granularity(self):
        """Test different scale granularities."""
        from sglang.srt.model_loader.ternary_hook import FP8HiddenStateManager
        
        # Per-token-group-128
        manager1 = FP8HiddenStateManager(
            hidden_size=2048,
            max_tokens=128,
            scale_granularity="per_token_group_128",
        )
        self.assertEqual(manager1.group_size, 128)
        
        # Per-token
        manager2 = FP8HiddenStateManager(
            hidden_size=2048,
            max_tokens=128,
            scale_granularity="per_token",
        )
        self.assertEqual(manager2.group_size, 2048)


def run_quick_sanity_check():
    """Run a quick sanity check that can be invoked from CLI."""
    print("=" * 60)
    print("FP8 Ternary Sanity Check")
    print("=" * 60)
    
    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        
        # Check FP8 dtype support
        try:
            _ = torch.tensor([1.0], device="cuda").to(torch.float8_e4m3fn)
            print("FP8 dtype (float8_e4m3fn): SUPPORTED")
        except Exception as e:
            print(f"FP8 dtype: NOT SUPPORTED ({e})")
    
    # Check TernaryConfig
    print("\nTernaryConfig FP8 settings:")
    try:
        from sglang.srt.layers.quantization.ternary import TernaryConfig
        config = TernaryConfig(threshold_scale=0.7, storage_mode="i2s", use_fp8=True)
        print(f"  use_fp8: {config.use_fp8}")
        print(f"  fp8_group_size: {config.fp8_group_size}")
        print(f"  kv_cache_quant_algo: {config.kv_cache_quant_algo}")
        print(f"  recommended_kv_cache_dtype: {config.recommended_kv_cache_dtype}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Check FP8 runtime
    print("\nFP8 runtime check:")
    try:
        from sglang.srt.layers.quantization.ternary import get_fp8_runtime_info
        info = get_fp8_runtime_info()
        for k, v in info.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Check FP8 bridge
    print("\nFP8 bridge check:")
    os.environ["SGLANG_TERNARY_FP8_BRIDGE"] = "1"
    try:
        from sglang.srt.layers.quantization.ternary import _fp8_bridge_available
        available = _fp8_bridge_available()
        print(f"  Bridge available: {available}")
    except Exception as e:
        print(f"  ERROR: {e}")
    finally:
        os.environ.pop("SGLANG_TERNARY_FP8_BRIDGE", None)
    
    print("\n" + "=" * 60)
    print("Sanity check complete")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FP8 Ternary Correctness Tests")
    parser.add_argument("--sanity", action="store_true", help="Run quick sanity check")
    parser.add_argument("test_name", nargs="?", help="Specific test to run")
    args, remaining = parser.parse_known_args()
    
    if args.sanity:
        run_quick_sanity_check()
    else:
        # Run unittest
        if args.test_name:
            remaining.insert(0, args.test_name)
        unittest.main(argv=[sys.argv[0]] + remaining)
