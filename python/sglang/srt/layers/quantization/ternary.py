"""Ternary quantization method for SGLang.

This module implements ternary quantization (weights in {-1, 0, 1} × alpha).

Supports two storage modes:
- i2s: 2-bit packed format (8x memory reduction)
- fp16: Direct ternary storage (no compression, for debugging)

Features:
- 8× memory savings with 2-bit weight storage
- Per-column alpha scaling for superior accuracy
- Optimized int8 quantization for activations and alpha
"""

import ctypes
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
)
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)

TERNARY_USE_CUDA_ACT_QUANT = os.environ.get("TERNARY_USE_CUDA_ACT_QUANT", "0") == "1"
DEFAULT_PREFILL_SKIP_M = int(os.environ.get("TERNARY_PREFILL_SKIP_M", "64"))
SUPPORTED_V4_NK_SHAPES = {
    (5120, 2048),
    (2048, 4096),
    (2048, 2048),
}

# ============================================================================
# Macroscale Profiler (function-level timing for decode batch breakdown)
# ============================================================================

class _MacroscaleProfiler:
    """Macroscale profiler for tracking function-level timing in decode batches."""
    
    def __init__(self):
        # Gate via env var so profiling can be toggled at runtime
        self.enabled = os.environ.get("TERNARY_MACRO_PROFILE", "0") == "1"
        self.macro_stats = defaultdict(lambda: {
            'count': 0,
            'total_time_ms': 0.0,
            'min_time_ms': float('inf'),
            'max_time_ms': 0.0,
        })
        self.macro_output_file = os.environ.get("TERNARY_MACRO_PROFILE_OUTPUT", "/tmp/ternary_macro_profile.json")
        self.active_timers = {}  # Track nested timers - supports multiple concurrent timers with same name
        if self.enabled:
            logger.info(f"[MACRO PROFILER] Enabled. Output: {self.macro_output_file}")
    
    def _safe_sync(self):
        """Safely synchronize CUDA, skipping during graph capture."""
        try:
            from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
            if get_is_capture_mode():
                return
        except (ImportError, AttributeError):
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
    
    def start(self, name: str):
        """Start timing a function/operation. Supports overlapping timers with same name."""
        if not self.enabled:
            return
        self._safe_sync()
        # Use a unique ID for each timer instance to support overlapping calls
        timer_id = id(self)  # Simple unique ID
        if name not in self.active_timers:
            self.active_timers[name] = []
        self.active_timers[name].append((timer_id, time.time()))
    
    def end(self, name: str):
        """End timing a function/operation. Accumulates time for overlapping calls."""
        if not self.enabled or name not in self.active_timers or len(self.active_timers[name]) == 0:
            return
        self._safe_sync()
        # Pop the most recent timer for this name
        timer_id, start_time = self.active_timers[name].pop()
        duration_ms = (time.time() - start_time) * 1000
        
        # Accumulate stats (supports multiple overlapping calls)
        stats = self.macro_stats[name]
        stats['count'] += 1
        stats['total_time_ms'] += duration_ms
        stats['min_time_ms'] = min(stats['min_time_ms'], duration_ms)
        stats['max_time_ms'] = max(stats['max_time_ms'], duration_ms)
        
        # Save to file periodically
        if stats['count'] % 10 == 0:
            try:
                summary = self._get_summary()
                with open(self.macro_output_file, 'w') as f:
                    json.dump(summary, f, indent=2)
            except Exception:
                pass
    
    def _get_summary(self):
        """Get summary statistics."""
        total_time = sum(s['total_time_ms'] for s in self.macro_stats.values())
        
        summary = {}
        for name, stats in sorted(self.macro_stats.items(), key=lambda x: -x[1]['total_time_ms']):
            avg_time = stats['total_time_ms'] / stats['count'] if stats['count'] > 0 else 0
            summary[name] = {
                'count': stats['count'],
                'total_ms': stats['total_time_ms'],
                'avg_ms': avg_time,
                'min_ms': stats['min_time_ms'] if stats['min_time_ms'] != float('inf') else 0,
                'max_ms': stats['max_time_ms'],
                'percentage': (stats['total_time_ms'] / total_time * 100) if total_time > 0 else 0,
            }
        
        return {
            'total_time_ms': total_time,
            'operations': summary
        }
    
    def print_summary(self):
        """Print summary to stderr."""
        if not self.enabled or not self.macro_stats:
            return
        
        summary = self._get_summary()
        
        print("\n" + "="*80, file=sys.stderr)
        print("MACROSCALE PROFILE (Decode Batch Breakdown)", file=sys.stderr)
        print("="*80, file=sys.stderr)
        print(f"\nTotal accumulated time: {summary['total_time_ms']:.2f} ms\n", file=sys.stderr)
        
        print(f"{'Operation':<50} {'Count':<8} {'Total (ms)':<12} {'Avg (ms)':<12} {'%':<8}", file=sys.stderr)
        print("-" * 90, file=sys.stderr)
        
        # Group by operation type for better readability
        decode_ops = {}
        prefill_ops = {}
        other_ops = {}
        
        for op_name, stats in summary['operations'].items():
            if '(DECODE)' in op_name or 'decode' in op_name.lower():
                decode_ops[op_name] = stats
            elif '(EXTEND)' in op_name or 'prefill' in op_name.lower():
                prefill_ops[op_name] = stats
            else:
                other_ops[op_name] = stats
        
        # Print decode operations first (most relevant)
        if decode_ops:
            print("\n[DECODE OPERATIONS]", file=sys.stderr)
            for op_name, stats in sorted(decode_ops.items(), key=lambda x: -x[1]['total_ms']):
                print(f"{op_name:<50} {stats['count']:<8} {stats['total_ms']:<12.2f} "
                      f"{stats['avg_ms']:<12.3f} {stats['percentage']:<8.1f}", file=sys.stderr)
        
        # Print prefill operations
        if prefill_ops:
            print("\n[PREFILL OPERATIONS]", file=sys.stderr)
            for op_name, stats in sorted(prefill_ops.items(), key=lambda x: -x[1]['total_ms']):
                print(f"{op_name:<50} {stats['count']:<8} {stats['total_ms']:<12.2f} "
                      f"{stats['avg_ms']:<12.3f} {stats['percentage']:<8.1f}", file=sys.stderr)
        
        # Print other operations
        if other_ops:
            print("\n[OTHER OPERATIONS]", file=sys.stderr)
            for op_name, stats in sorted(other_ops.items(), key=lambda x: -x[1]['total_ms']):
                print(f"{op_name:<50} {stats['count']:<8} {stats['total_ms']:<12.2f} "
                      f"{stats['avg_ms']:<12.3f} {stats['percentage']:<8.1f}", file=sys.stderr)
        
        # Analysis section
        print("\n" + "="*80, file=sys.stderr)
        print("ANALYSIS", file=sys.stderr)
        print("="*80, file=sys.stderr)
        
        # Calculate decode batch breakdown
        decode_batch_time = decode_ops.get('scheduler.run_batch(decode)', {}).get('total_ms', 0)
        decode_forward_time = decode_ops.get('model_runner.forward(DECODE)', {}).get('total_ms', 0)
        decode_sample_time = decode_ops.get('model_runner.sample', {}).get('total_ms', 0)
        decode_attention_time = decode_ops.get('attention_all_layers(DECODE)', {}).get('total_ms', 0)
        decode_mlp_time = decode_ops.get('mlp_all_layers(DECODE)', {}).get('total_ms', 0)
        decode_embedding_time = decode_ops.get('embedding(DECODE)', {}).get('total_ms', 0)
        decode_lm_head_time = decode_ops.get('lm_head(DECODE)', {}).get('total_ms', 0)
        decode_norm_time = decode_ops.get('layer_norm(DECODE)', {}).get('total_ms', 0)
        
        if decode_batch_time > 0:
            print(f"\nDecode Batch Breakdown (per batch, avg over {decode_ops.get('scheduler.run_batch(decode)', {}).get('count', 1)} batches):", file=sys.stderr)
            print(f"  Total decode batch time: {decode_batch_time / max(1, decode_ops.get('scheduler.run_batch(decode)', {}).get('count', 1)):.3f} ms", file=sys.stderr)
            if decode_forward_time > 0:
                print(f"  ├─ Forward pass: {decode_forward_time / max(1, decode_ops.get('model_runner.forward(DECODE)', {}).get('count', 1)):.3f} ms", file=sys.stderr)
                if decode_attention_time > 0:
                    print(f"  │  ├─ Attention (all layers): {decode_attention_time / max(1, decode_ops.get('attention_all_layers(DECODE)', {}).get('count', 1)):.3f} ms", file=sys.stderr)
                if decode_mlp_time > 0:
                    print(f"  │  ├─ MLP (all layers): {decode_mlp_time / max(1, decode_ops.get('mlp_all_layers(DECODE)', {}).get('count', 1)):.3f} ms", file=sys.stderr)
                if decode_embedding_time > 0:
                    print(f"  │  ├─ Embedding: {decode_embedding_time / max(1, decode_ops.get('embedding(DECODE)', {}).get('count', 1)):.3f} ms", file=sys.stderr)
                if decode_norm_time > 0:
                    print(f"  │  └─ Layer norm: {decode_norm_time / max(1, decode_ops.get('layer_norm(DECODE)', {}).get('count', 1)):.3f} ms", file=sys.stderr)
            if decode_lm_head_time > 0:
                print(f"  ├─ LM head: {decode_lm_head_time / max(1, decode_ops.get('lm_head(DECODE)', {}).get('count', 1)):.3f} ms", file=sys.stderr)
            if decode_sample_time > 0:
                print(f"  └─ Sampling: {decode_sample_time / max(1, decode_ops.get('model_runner.sample', {}).get('count', 1)):.3f} ms", file=sys.stderr)
            
            # Calculate overhead
            accounted_time = decode_forward_time + decode_sample_time + decode_lm_head_time
            overhead = decode_batch_time - accounted_time
            if overhead > 0:
                print(f"\n  Overhead (scheduler, communication, etc.): {overhead / max(1, decode_ops.get('scheduler.run_batch(decode)', {}).get('count', 1)):.3f} ms", file=sys.stderr)
        
        print(f"\n[MACRO PROFILER] Profile saved to: {self.macro_output_file}", file=sys.stderr)

# Global macroscale profiler instance
_macro_profiler = _MacroscaleProfiler()

# Register cleanup hook
if _macro_profiler.enabled:
    import atexit
    atexit.register(_macro_profiler.print_summary)

# ============================================================================
# Inline Profiler (controlled by TERNARY_ENABLE_PROFILING env var)
# ============================================================================

class _TernaryProfiler:
    """Simple inline profiler for ternary operations."""
    
    def __init__(self):
        # Enable when TERNARY_ENABLE_PROFILING=1
        self.enabled = os.environ.get("TERNARY_ENABLE_PROFILING", "0") == "1"
        self.stats = defaultdict(lambda: {
            'count': 0,
            'total_time_ms': 0.0,
            'min_time_ms': float('inf'),
            'max_time_ms': 0.0,
        })
        self.output_file = os.environ.get("TERNARY_PROFILE_OUTPUT", "/tmp/ternary_profile.json")
        if self.enabled:
            logger.info(f"[TERNARY PROFILER] Enabled. Output: {self.output_file}")
    
    def _is_graph_capturing(self):
        """Check if we're in a CUDA graph capture context."""
        try:
            # Method 1: Check PyTorch's built-in function (PyTorch 2.0+)
            if hasattr(torch.cuda, 'is_current_stream_capturing'):
                if torch.cuda.is_current_stream_capturing():
                    return True
            # Method 2: Check SGLang's capture mode flag
            try:
                from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
                if get_is_capture_mode():
                    return True
            except (ImportError, AttributeError):
                pass
            return False
        except Exception:
            return False
    
    def _safe_sync(self):
        """Safely synchronize CUDA, skipping during graph capture."""
        if self._is_graph_capturing():
            return  # Skip sync during graph capture
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass  # Ignore sync errors
    
    def record(self, name: str, duration_ms: float):
        """Record an operation."""
        if not self.enabled:
            return
        
        stats = self.stats[name]
        stats['count'] += 1
        stats['total_time_ms'] += duration_ms
        stats['min_time_ms'] = min(stats['min_time_ms'], duration_ms)
        stats['max_time_ms'] = max(stats['max_time_ms'], duration_ms)
        
        # Save to file after every operation
        try:
            summary = self._get_summary()
            with open(self.output_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass  # Don't fail if file write fails
    
    def _get_summary(self):
        """Get summary statistics."""
        total_time = sum(s['total_time_ms'] for s in self.stats.values())
        
        summary = {}
        for name, stats in sorted(self.stats.items(), key=lambda x: -x[1]['total_time_ms']):
            avg_time = stats['total_time_ms'] / stats['count'] if stats['count'] > 0 else 0
            summary[name] = {
                'count': stats['count'],
                'total_ms': stats['total_time_ms'],
                'avg_ms': avg_time,
                'min_ms': stats['min_time_ms'] if stats['min_time_ms'] != float('inf') else 0,
                'max_ms': stats['max_time_ms'],
                'percentage': (stats['total_time_ms'] / total_time * 100) if total_time > 0 else 0,
            }
        
        return {
            'total_time_ms': total_time,
            'operations': summary
        }
    
    def print_summary(self):
        """Print summary to stderr."""
        if not self.enabled or not self.stats:
            return
        
        summary = self._get_summary()
        
        print("\n" + "="*80, file=sys.stderr)
        print("TERNARY OPERATION PROFILE", file=sys.stderr)
        print("="*80, file=sys.stderr)
        print(f"\nTotal time: {summary['total_time_ms']:.2f} ms\n", file=sys.stderr)
        
        print(f"{'Operation':<60} {'Count':<8} {'Total (ms)':<12} {'Avg (ms)':<12} {'%':<8}", file=sys.stderr)
        print("-" * 100, file=sys.stderr)
        
        # Group operations by category
        quantize_ops = []
        v4_ops = []
        fallback_ops = []
        skip_ops = []
        
        for op_name, stats in summary['operations'].items():
            if 'quantize' in op_name:
                quantize_ops.append((op_name, stats))
            elif 'v4_kernel' in op_name:
                v4_ops.append((op_name, stats))
            elif 'prefill_skip' in op_name:
                skip_ops.append((op_name, stats))
            elif 'fallback' in op_name:
                fallback_ops.append((op_name, stats))
        
        # Print quantization operations
        if quantize_ops:
            print("\n[QUANTIZATION]", file=sys.stderr)
            for op_name, stats in sorted(quantize_ops, key=lambda x: -x[1]['total_ms'])[:20]:
                print(f"{op_name:<60} {stats['count']:<8} {stats['total_ms']:<12.2f} "
                      f"{stats['avg_ms']:<12.3f} {stats['percentage']:<8.1f}", file=sys.stderr)
        
        # Print V4 kernel operations
        if v4_ops:
            print("\n[V4 KERNEL EXECUTION]", file=sys.stderr)
            for op_name, stats in sorted(v4_ops, key=lambda x: -x[1]['total_ms'])[:20]:
                print(f"{op_name:<60} {stats['count']:<8} {stats['total_ms']:<12.2f} "
                      f"{stats['avg_ms']:<12.3f} {stats['percentage']:<8.1f}", file=sys.stderr)
        
        # Print skip operations
        if skip_ops:
            print("\n[PREFILL SKIP (M > threshold)]", file=sys.stderr)
            skip_count = sum(s['count'] for _, s in skip_ops)
            print(f"Total skipped operations: {skip_count}", file=sys.stderr)
        
        # Print fallback operations
        if fallback_ops:
            print("\n[FALLBACK (FP16 Path)]", file=sys.stderr)
            for op_name, stats in sorted(fallback_ops, key=lambda x: -x[1]['total_ms'])[:20]:
                print(f"{op_name:<60} {stats['count']:<8} {stats['total_ms']:<12.2f} "
                      f"{stats['avg_ms']:<12.3f} {stats['percentage']:<8.1f}", file=sys.stderr)
        
        # Analysis
        print("\n" + "="*80, file=sys.stderr)
        print("ANALYSIS", file=sys.stderr)
        print("="*80, file=sys.stderr)
        
        v4_time = sum(s['total_ms'] for name, s in summary['operations'].items() if 'v4_kernel' in name)
        fallback_time = sum(s['total_ms'] for name, s in summary['operations'].items() 
                           if 'fallback' in name and 'v4' not in name)
        quant_time = sum(s['total_ms'] for name, s in summary['operations'].items() if 'quantize' in name)
        
        total_ops_time = summary['total_time_ms']
        
        if total_ops_time > 0:
            print(f"\nV4 Kernel Operations: {v4_time:.2f} ms ({v4_time/total_ops_time*100:.1f}%)", file=sys.stderr)
            print(f"Fallback Operations: {fallback_time:.2f} ms ({fallback_time/total_ops_time*100:.1f}%)", file=sys.stderr)
            print(f"Quantization Overhead: {quant_time:.2f} ms ({quant_time/total_ops_time*100:.1f}%)", file=sys.stderr)
            
            # Breakdown quantization by M value
            m1_quant = sum(s['total_ms'] for name, s in summary['operations'].items() if 'quantize_activation_M1_' in name)
            m_small_quant = sum(s['total_ms'] for name, s in summary['operations'].items() 
                               if 'quantize_activation' in name and any(f'_M{m}_' in name for m in [2,4,6,8,12,16,24,32,40,48,56,64]))
            
            print(f"  ├─ M=1 (decode): {m1_quant:.2f} ms ({m1_quant/total_ops_time*100:.1f}%)", file=sys.stderr)
            print(f"  └─ M=2-64 (small batch): {m_small_quant:.2f} ms ({m_small_quant/total_ops_time*100:.1f}%)", file=sys.stderr)
            
            if v4_time + fallback_time > 0:
                print(f"\nV4 Kernel Efficiency: {v4_time/(v4_time+fallback_time)*100:.1f}% of quantized ops use V4", file=sys.stderr)
            
            # Performance estimate
            if m1_quant > 0:
                m1_count = sum(s['count'] for name, s in summary['operations'].items() if 'quantize_activation_M1_' in name)
                if m1_count > 0:
                    print(f"\nM=1 Quantization: {m1_quant/m1_count:.3f} ms per call (avg)", file=sys.stderr)
        
        print(f"\n[PROFILER] Profile saved to: {self.output_file}", file=sys.stderr)
        
        # Also print macroscale summary if enabled
        if _macro_profiler.enabled:
            _macro_profiler.print_summary()

# Global profiler instance
_ternary_profiler = _TernaryProfiler()

# Register cleanup hook
if _ternary_profiler.enabled:
    import atexit
    atexit.register(_ternary_profiler.print_summary)

class _FastApplyProfiler:
    """Extremely lightweight profiler using CUDA events with deferred synchronization."""
    def __init__(self):
        self.enabled = os.environ.get("TERNARY_FAST_PROFILE", "0") == "1"
        self.events = []
        self.max_events = 50000  # Store up to 50k events
        
        if self.enabled:
            logger.info("[FastProfiler] Enabled for apply()")
            import atexit
            atexit.register(self.report)
            
    def start(self):
        if not self.enabled:
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return (start, end)
        
    def stop(self, ctx):
        if not self.enabled or ctx is None:
            return
        start, end = ctx
        end.record()
        if len(self.events) < self.max_events:
            self.events.append((start, end))
            
    def report(self):
        if not self.events:
            return
            
        print(f"[FastProfiler] Synchronizing {len(self.events)} events...", file=sys.stderr)
        try:
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[FastProfiler] Warning: Sync failed ({e}), attempting to read events anyway", file=sys.stderr)
        
        times = []
        for s, e in self.events:
            try:
                # Check if event is recorded before querying
                if s.query() and e.query():
                    times.append(s.elapsed_time(e))
            except Exception:
                pass
            
        if not times:
            print(f"[FastProfiler] No valid timed events found (out of {len(self.events)})", file=sys.stderr)
            return
            
        avg = sum(times) / len(times)
        times.sort()
        p50 = times[len(times)//2]
        p95 = times[int(len(times)*0.95)]
        p99 = times[int(len(times)*0.99)]
        
        print("\n" + "="*60, file=sys.stderr)
        print(f"FastProfiler Stats (apply method)", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(f"Total Calls: {len(times)}", file=sys.stderr)
        print(f"Avg Time:    {avg:.4f} ms", file=sys.stderr)
        print(f"P50 Time:    {p50:.4f} ms", file=sys.stderr)
        print(f"P95 Time:    {p95:.4f} ms", file=sys.stderr)
        print(f"P99 Time:    {p99:.4f} ms", file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)

_fast_apply_profiler = _FastApplyProfiler()

BITNET_PACK_AVAILABLE = False
convert_weight_int8_to_int2 = None
try:
    bitnet_gpu_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../../../../BitNet/gpu'))
    if os.path.isdir(bitnet_gpu_path) and bitnet_gpu_path not in sys.path:
        sys.path.append(bitnet_gpu_path)
    from pack_weight import convert_weight_int8_to_int2 as _bitnet_pack_fn
    convert_weight_int8_to_int2 = _bitnet_pack_fn
    BITNET_PACK_AVAILABLE = True
except Exception as e:
    logger.debug(f"[TERNARY] BitNet weight packer not available ({e}), kernel path will be disabled")

TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass

# Try to import optimized BitNet CUDA kernel (V4 - production)
BITNET_CUDA_AVAILABLE = False
BITNET_LIB = None
BITNET_CUDA_ACT_QUANT_AVAILABLE = False
_BITNET_ACT_FAST_FN = None
_bitnet_act_fast_error_logged = False
try:
    # Try to load the shared library directly
    lib_paths = [
        os.path.join(os.path.dirname(__file__), '../../../../../libternary_bitnet.so'),
        './libternary_bitnet.so',
        '/usr/local/lib/libternary_bitnet.so',
    ]
    
    for lib_path in lib_paths:
        if os.path.exists(lib_path):
            BITNET_LIB = ctypes.CDLL(lib_path)
            
            # Define C function signatures for V4 kernel (Pre-scaled activations)
            # int bitlinear_int8xint2_v4_simple(...)
            BITNET_LIB.bitlinear_int8xint2_v4_simple.argtypes = [
                ctypes.c_void_p,  # input0 (int8 activations)
                ctypes.c_void_p,  # input1 (packed weights)
                ctypes.c_void_p,  # alpha_q (int8)
                ctypes.c_void_p,  # alpha_scale (float)
                ctypes.c_void_p,  # output0 (bf16)
                ctypes.c_void_p,  # s (bf16 activation scale)
                ctypes.c_int,     # M
                ctypes.c_int,     # N
                ctypes.c_int,     # K
                ctypes.c_void_p,  # stream
            ]
            BITNET_LIB.bitlinear_int8xint2_v4_simple.restype = ctypes.c_int
            
            BITNET_CUDA_AVAILABLE = True
            logger.info(f"[TERNARY] BitNet CUDA V4 kernel loaded successfully from {lib_path}")
            logger.info("[TERNARY] V4 kernel features: Pre-scaled activations, 1.8-2.0x speedup, correct production output")

            if hasattr(BITNET_LIB, 'ternary_quantize_activation_fast'):
                BITNET_LIB.ternary_quantize_activation_fast.argtypes = [
                    ctypes.c_void_p,  # activations (bf16)
                    ctypes.c_void_p,  # alpha (fp32)
                    ctypes.c_void_p,  # output int8
                    ctypes.c_void_p,  # per-row scale (bf16)
                    ctypes.c_int,     # M
                    ctypes.c_int,     # K
                    ctypes.c_void_p,  # cuda stream
                ]
                BITNET_LIB.ternary_quantize_activation_fast.restype = ctypes.c_int
                _BITNET_ACT_FAST_FN = BITNET_LIB.ternary_quantize_activation_fast
                BITNET_CUDA_ACT_QUANT_AVAILABLE = True
                if TERNARY_USE_CUDA_ACT_QUANT:
                    logger.info("[TERNARY] CUDA fast activation quantizer enabled (ternary_quantize_activation_fast)")
            break
    
    if not BITNET_CUDA_AVAILABLE:
        logger.debug("[TERNARY] BitNet CUDA kernel not found in any search path, will use Triton fallback")
except Exception as e:
    logger.debug(f"[TERNARY] BitNet CUDA kernel not available ({e}), will use Triton fallback")
    pass

# Triton kernel for I2S unpacking (faster than PyTorch operations)
if TRITON_AVAILABLE:
    @triton.jit
    def _i2s_unpack_kernel(
        packed_ptr,
        alpha_ptr,
        output_ptr,
        N,
        K,
        num_packed_cols,
        stride_packed_n,
        stride_packed_k,
        stride_output_n,
        stride_output_k,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """
        Unpack I2S weights using optimized Triton kernel.
        
        Optimizations:
        - Efficient memory access patterns
        - Reduced redundant mask computations
        - Optimized bit extraction
        """
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        
        n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        k_offsets = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        n_mask = n_offsets < N
        k_mask = k_offsets < K
        
        n_indices = n_offsets[:, None]
        k_indices = k_offsets[None, :]
        
        packed_k_idx = k_indices // 4
        bit_pos = (k_indices % 4) * 2
        
        packed_offsets = n_indices * stride_packed_n + packed_k_idx * stride_packed_k
        valid_mask = n_mask[:, None] & k_mask[None, :] & (packed_k_idx < num_packed_cols)
        
        packed_bytes = tl.load(
            packed_ptr + packed_offsets,
            mask=valid_mask,
            other=0
        )
        
        extracted = (packed_bytes >> bit_pos) & 0b11
        val_ternary = extracted.to(tl.float32) - 1.0
        
        alpha_vals = tl.load(alpha_ptr + k_offsets, mask=k_mask, other=1.0)
        output_values = val_ternary * alpha_vals[None, :]
        
        output_mask = n_mask[:, None] & k_mask[None, :]
        output_offsets = n_offsets[:, None] * stride_output_n + k_offsets[None, :] * stride_output_k
        
        tl.store(output_ptr + output_offsets, output_values, mask=output_mask)

    @triton.jit
    def _quantize_activation_prescale_kernel(
        x_ptr,              # Input activations (M, K) [BF16]
        alpha_ptr,          # Alpha scaling factors (K,) [FP32]
        output_ptr,         # Output quantized (M, K) [Int8]
        scale_ptr,          # Output scales (M,) [BF16]
        K,                  # Number of columns
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused kernel: Pre-scale activations by alpha and quantize to int8.
        
        Per-row quantization (each row gets its own scale factor).
        """
        # Parallelize over rows (M)
        pid = tl.program_id(0)
        
        # Pointers for this row
        x_row_ptr = x_ptr + pid * K
        output_row_ptr = output_ptr + pid * K
        
        # 1. First pass: Compute max(abs(x * alpha)) to find scale
        max_val = 0.0
        
        # Loop over K in chunks to find max
        for off in range(0, K, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < K
            
            # Load x and alpha
            x_val = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            alpha_val = tl.load(alpha_ptr + cols, mask=mask, other=0.0)  # alpha is FP32
            
            # Pre-scale: x * alpha
            val = x_val * alpha_val
            abs_val = tl.abs(val)
            
            # Update max
            max_val = tl.maximum(max_val, tl.max(abs_val, axis=0))
        
        # Compute scale = 127 / max_val
        # Handle close-to-zero max_val to avoid NaN/Inf
        scale = 127.0 / (max_val + 1e-8)
        
        # Store scale (convert to BF16)
        tl.store(scale_ptr + pid, scale.to(tl.bfloat16))
        
        # 2. Second pass: Quantize
        # output = clamp(round(x * alpha * scale), -128, 127)
        for off in range(0, K, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < K
            
            x_val = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            alpha_val = tl.load(alpha_ptr + cols, mask=mask, other=0.0)
            
            # Apply scaling
            val = x_val * alpha_val * scale
            
            # Round and clamp
            # Manual round to nearest (standard round half up behavior)
            q_val = val + 0.5
            q_val = tl.floor(q_val)
            q_val = tl.clamp(q_val, -128.0, 127.0)
            
            # Store as int8
            tl.store(output_row_ptr + cols, q_val.to(tl.int8), mask=mask)

    @triton.jit
    def _quantize_activation_prescale_kernel_m1(
        x_ptr,              # Input activations (1, K) [BF16]
        alpha_ptr,          # Alpha scaling factors (K,) [FP32]
        output_ptr,         # Output quantized (1, K) [Int8]
        scale_ptr,          # Output scales (1,) [BF16]
        K,                  # Number of columns
        BLOCK_SIZE: tl.constexpr,
    ):
        """Optimized fused kernel for M=1: Loads data once into SRAM."""
        # For M=1, we process the single row in one go (assuming K <= BLOCK_SIZE)
        # BLOCK_SIZE should be next_power_of_2(K)
        
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < K
        
        # 1. Load all data once (fits in SRAM for K<=4096)
        # x and alpha are contiguous
        x_val = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        alpha_val = tl.load(alpha_ptr + cols, mask=mask, other=0.0)
        
        # 2. Pre-scale and find max
        val = x_val * alpha_val
        abs_val = tl.abs(val)
        max_val = tl.max(abs_val, axis=0)
        
        # 3. Compute scale
        scale = 127.0 / (max_val + 1e-8)
        tl.store(scale_ptr, scale.to(tl.bfloat16))
        
        # 4. Quantize
        q_val = val * scale
        q_val = q_val + 0.5
        q_val = tl.floor(q_val)
        q_val = tl.clamp(q_val, -128.0, 127.0)
        
        # 5. Store
        tl.store(output_ptr + cols, q_val.to(tl.int8), mask=mask)

def get_tensor_memory_bytes(t: torch.Tensor) -> int:
    """Get the memory usage of a tensor in bytes."""
    if t is None or not hasattr(t, 'element_size'):
        return 0
    return t.numel() * t.element_size()


def get_layer_memory_bytes(layer: torch.nn.Module) -> int:
    """Get total memory usage of a layer's tensors."""
    total = 0
    for name, param in layer.named_parameters():
        total += get_tensor_memory_bytes(param)
    for name, buffer in layer.named_buffers():
        total += get_tensor_memory_bytes(buffer)
    return total


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def get_actual_gpu_memory_bytes(device: Optional[torch.device] = None) -> int:
    """Get actual GPU memory allocated (not just tensor sizes).
    
    This measures the real GPU memory usage, accounting for:
    - Actual allocated memory (may be larger than tensor sizes due to alignment)
    - Memory fragmentation
    - CUDA memory pool overhead
    
    Args:
        device: CUDA device to measure. If None, uses current device.
    
    Returns:
        Memory allocated in bytes, or 0 if CUDA not available.
    """
    if not torch.cuda.is_available():
        return 0
    
    if device is None:
        device = torch.cuda.current_device()
    
    return torch.cuda.memory_allocated(device)


def force_cleanup_and_sync(device: Optional[torch.device] = None) -> None:
    """Force cleanup of Python objects and CUDA cache.
    
    This ensures that:
    1. Python garbage collector runs
    2. CUDA cache is cleared
    3. All CUDA operations are synchronized
    
    Args:
        device: CUDA device to sync. If None, uses current device.
    """
    gc.collect()
    
    if torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def measure_layer_memory_accurate(
    layer: torch.nn.Module,
    device: Optional[torch.device] = None,
    include_gpu_actual: bool = True
) -> Dict[str, Any]:
    """Accurately measure layer memory usage using multiple methods.
    
    Measures both theoretical tensor sizes and actual GPU memory allocation.
    This helps verify that quantization actually reduces memory.
    
    Args:
        layer: The layer to measure
        device: CUDA device. If None, uses current device.
        include_gpu_actual: Whether to measure actual GPU memory (requires CUDA)
    
    Returns:
        Dictionary with memory measurements:
            - param_memory_bytes: Theoretical parameter memory
            - buffer_memory_bytes: Theoretical buffer memory
            - total_theoretical_bytes: Total theoretical memory
            - gpu_allocated_bytes: Actual GPU memory allocated (if CUDA available)
            - gpu_reserved_bytes: GPU memory reserved by CUDA (if CUDA available)
    """
    # Theoretical memory (tensor sizes)
    param_memory = sum(get_tensor_memory_bytes(p) for p in layer.parameters())
    buffer_memory = sum(get_tensor_memory_bytes(b) for b in layer.buffers())
    total_theoretical = param_memory + buffer_memory
    
    result = {
        'param_memory_bytes': param_memory,
        'buffer_memory_bytes': buffer_memory,
        'total_theoretical_bytes': total_theoretical,
    }
    
    # Actual GPU memory (if available)
    if include_gpu_actual and torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        
        # Synchronize to ensure accurate measurement
        torch.cuda.synchronize(device)
        
        result['gpu_allocated_bytes'] = torch.cuda.memory_allocated(device)
        result['gpu_reserved_bytes'] = torch.cuda.memory_reserved(device)
        result['gpu_max_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    
    return result


def get_memory_snapshot(device: Optional[torch.device] = None) -> Dict[str, int]:
    """Get a snapshot of current GPU memory state.
    
    Useful for tracking memory changes before/after operations.
    
    Args:
        device: CUDA device. If None, uses current device.
    
    Returns:
        Dictionary with memory statistics:
            - allocated: Currently allocated memory
            - reserved: Currently reserved memory
            - max_allocated: Peak allocated memory
    """
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}
    
    if device is None:
        device = torch.cuda.current_device()
    
    torch.cuda.synchronize(device)
    
    return {
        'allocated': torch.cuda.memory_allocated(device),
        'reserved': torch.cuda.memory_reserved(device),
        'max_allocated': torch.cuda.max_memory_allocated(device),
    }

@torch.inference_mode()
def _quantize_activation_fast_cuda(
    x: torch.Tensor,
    alpha: torch.Tensor,
    out_int8: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    global _bitnet_act_fast_error_logged

    if not (TERNARY_USE_CUDA_ACT_QUANT and BITNET_CUDA_ACT_QUANT_AVAILABLE and _BITNET_ACT_FAST_FN is not None):
        return None

    if not (x.is_cuda and alpha.is_cuda):
        return None

    if x.dtype != torch.bfloat16 or alpha.dtype != torch.float32:
        return None

    if x.dim() != 2:
        return None

    x_c = x.contiguous()
    alpha_c = alpha.contiguous()
    M, K = x_c.shape

    if out_int8 is None or out_int8.shape != x_c.shape or out_int8.dtype != torch.int8 or not out_int8.is_contiguous():
        x_int8 = torch.empty_like(x_c, dtype=torch.int8)
    else:
        x_int8 = out_int8

    if (
        out_scale is None
        or out_scale.shape[0] != M
        or out_scale.dtype != torch.bfloat16
        or out_scale.device != x.device
        or not out_scale.is_contiguous()
    ):
        scale = torch.empty(M, device=x.device, dtype=torch.bfloat16)
    else:
        scale = out_scale

    err = _BITNET_ACT_FAST_FN(
        ctypes.c_void_p(x_c.data_ptr()),
        ctypes.c_void_p(alpha_c.data_ptr()),
        ctypes.c_void_p(x_int8.data_ptr()),
        ctypes.c_void_p(scale.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(K),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )
    if err != 0:
        if not _bitnet_act_fast_error_logged:
            logger.warning(f"[TERNARY] CUDA fast activation quantizer failed (code {err}), disabling")
            _bitnet_act_fast_error_logged = True
        return None

    return x_int8, scale


@torch.inference_mode()
def quantize_activation_prescale_fused(
    x: torch.Tensor, 
    alpha: torch.Tensor,
    out_int8: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused operation: Pre-scale activations by alpha and quantize to int8.
    
    Uses Triton kernel when available for 5x speedup, falls back to CUDA helper
    and finally to PyTorch.
    """
    M, K = x.shape
    
    if TRITON_AVAILABLE and x.is_cuda and alpha.is_cuda:
        try:
            if out_int8 is not None:
                x_int8 = out_int8
            else:
                x_int8 = torch.empty_like(x, dtype=torch.int8)
                
            if out_scale is not None:
                scale = out_scale
            else:
                scale = torch.empty(M, device=x.device, dtype=torch.bfloat16)
            
            # M=1 optimization: Use specialized kernel that loads data once
            # SAFETY: Only enable for power-of-2 K to avoid out-of-bounds reads
            if M == 1 and K in (2048, 4096, 8192):
                # BLOCK_SIZE = K exactly (no overread risk)
                BLOCK_SIZE = K
                _quantize_activation_prescale_kernel_m1[(1,)](
                    x.contiguous(), alpha.contiguous(), x_int8, scale,
                    K,
                    BLOCK_SIZE=BLOCK_SIZE
                )
                return x_int8, scale

            # General case
            # Heuristic for block size
            BLOCK_SIZE = 1024
            if K >= 4096:
                BLOCK_SIZE = 4096
            elif K >= 2048:
                BLOCK_SIZE = 2048
            
            # Limit block size to Triton limits
            BLOCK_SIZE = min(BLOCK_SIZE, 4096)
            
            grid = (M,)
            
            _quantize_activation_prescale_kernel[grid](
                x.contiguous(), alpha.contiguous(), x_int8, scale,
                K,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            return x_int8, scale
        except Exception as e:
            logger.debug(f"[TERNARY] Triton fused quantization failed ({e}), trying CUDA fallback")

    fast_result = _quantize_activation_fast_cuda(x, alpha, out_int8=out_int8, out_scale=out_scale)
    if fast_result is not None:
        return fast_result
    
    # PyTorch fallback
    # 1. Pre-scale
    x_scaled = x.to(torch.float32) * alpha.view(1, -1)
    
    # 2. Quantize (per-row scaling)
    x_max = x_scaled.abs().amax(dim=1, keepdim=True)
    scale = (127.0 / (x_max + 1e-8))
    
    x_q = x_scaled * scale
    x_int8 = torch.round(x_q).clamp(-128, 127).to(torch.int8)
    
    return x_int8, scale.squeeze(1).to(torch.bfloat16)


def quantize_activation_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activation to int8 with per-tensor scaling.
    
    Args:
        x: Input activation tensor (M, K)
    
    Returns:
        x_int8: Quantized int8 tensor (M, K)
        scale: Scale factor per row (M,), bf16
    """
    # Per-tensor quantization (all rows use same scale)
    x_max = x.abs().max()
    inv_scale = (127.0 / x_max).clamp(min=1e-8)  # Avoid division by zero
    
    x_scaled = x * inv_scale
    x_int8 = torch.round(x_scaled).clamp(-128, 127).to(torch.int8)
    
    # Kernel expects per-row scale with shape (M,), so broadcast scalar to (M,)
    M = x.shape[0]
    scale = inv_scale.to(torch.bfloat16).expand(M).clone()  # clone() to get actual memory
    
    return x_int8, scale


def quantize_alpha_int8(alpha: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize per-column alpha to int8 with global scaling.
    
    Args:
        alpha: Per-column alpha values (K,), fp32
    
    Returns:
        alpha_q: Quantized alpha (K,), int8
        alpha_scale: Global scale factor (scalar, fp32)
    """
    alpha_max = alpha.abs().max()
    alpha_max = alpha_max.clamp(min=1e-8)
    alpha_scale = (alpha_max / 127.0).item()
    
    alpha_q = torch.round(alpha / alpha_scale).clamp(-128, 127).to(torch.int8)
    
    return alpha_q, alpha_scale


def pack_i2s_weights(weight_ternary: torch.Tensor) -> torch.Tensor:
    """Pack ternary weights {-1, 0, 1} into 2-bit format.
    
    Packing: 4 values per byte
    - -1 -> 00 (0)
    - 0  -> 01 (1) 
    - 1  -> 10 (2)
    """
    N, K = weight_ternary.shape
    
    weight_mapped = (weight_ternary + 1).clamp(0, 2).to(torch.uint8)
    
    pad_K = (4 - (K % 4)) % 4
    if pad_K > 0:
        weight_mapped = torch.nn.functional.pad(weight_mapped, (0, pad_K), value=1)
    
    K_padded = K + pad_K
    num_packed_cols = K_padded // 4
    
    weight_reshaped = weight_mapped.view(N, num_packed_cols, 4)
    
    weight_packed = (
        weight_reshaped[:, :, 0] |
        (weight_reshaped[:, :, 1] << 2) |
        (weight_reshaped[:, :, 2] << 4) |
        (weight_reshaped[:, :, 3] << 6)
    ).to(torch.uint8)
    
    return weight_packed


def unpack_i2s_weights(weight_packed: torch.Tensor, K: int, alpha: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Unpack I2S weights using Triton kernel (if available) or optimized PyTorch operations.
    
    Tries Triton kernel first for better performance, falls back to PyTorch if unavailable.
    """
    assert weight_packed.dtype == torch.uint8, f"Expected uint8 packed weights, got {weight_packed.dtype}"
    assert alpha.dim() == 1 and alpha.shape[0] == K, f"Alpha shape {alpha.shape} doesn't match K={K}"
    assert alpha.dtype == torch.float32, f"Alpha must be stored in FP32 for precision, got {alpha.dtype}"
    
    N, num_packed_cols = weight_packed.shape
    device = weight_packed.device
    
    if TRITON_AVAILABLE and device.type == 'cuda':
        try:
            weight_packed_contig = weight_packed.contiguous()
            alpha_contig = alpha.contiguous()
            weight_unpacked = torch.empty(N, K, device=device, dtype=dtype)
            
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
            _i2s_unpack_kernel[grid](
                weight_packed_contig, alpha_contig, weight_unpacked,
                N, K, num_packed_cols,
                weight_packed_contig.stride(0), weight_packed_contig.stride(1),
                weight_unpacked.stride(0), weight_unpacked.stride(1),
                BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
            )
            
            return weight_unpacked
        except Exception as e:
            logger.debug(f"[TERNARY] Triton unpack kernel failed, using PyTorch fallback: {e}")
    
    packed_expanded = weight_packed.unsqueeze(-1)
    shift_positions = torch.arange(4, device=device, dtype=torch.uint8) * 2
    extracted_all = (packed_expanded >> shift_positions.view(1, 1, -1)) & 0b11
    
    K_padded = num_packed_cols * 4
    if K_padded == K:
        extracted = extracted_all.reshape(N, K)
    else:
        extracted = extracted_all.reshape(N, K_padded)[:, :K]
    
    val_ternary = extracted.to(torch.float32) - 1.0
    
    if alpha.is_contiguous():
        weight_unpacked_fp32 = val_ternary * alpha.view(1, -1)
    else:
        weight_unpacked_fp32 = val_ternary * alpha.contiguous().view(1, -1)
    
    weight_unpacked = weight_unpacked_fp32.to(dtype)
    
    return weight_unpacked


def _unpack_i2s_and_linear_impl(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    K: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Unpack I2S weights and perform linear layer computation.
    
    Core implementation that can be compiled with torch.compile.
    """
    weight_unpacked = unpack_i2s_weights(weight_packed, K, alpha, dtype)
    out = F.linear(x, weight_unpacked, bias)
    return out




def _unpack_i2s_and_linear(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    K: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Unpack I2S weights and perform linear layer computation.
    
    Uses Triton kernel for unpacking followed by F.linear.
    """
    # Directly use the implementation (Triton unpack + F.linear)
    return _unpack_i2s_and_linear_impl(x, weight_packed, alpha, bias, K, dtype)


def _unpack_i2s_and_linear_fp8(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    K: int,
    N: int,
) -> torch.Tensor:
    """Unpack I2S ternary weights to FP8 and perform FP8 tensor core matmul.
    
    Uses torch._scaled_mm for FP8 tensor core computation.
    Falls back to FP16 path if FP8 is not available.
    
    Args:
        x: Input activations (FP16/BF16)
        weight_packed: Packed I2S weights (uint8)
        alpha: Per-row scaling factors for ternary weights
        bias: Optional bias term
        K: Input features dimension
        N: Output features dimension
    
    Returns:
        Output tensor in same dtype as input
    """
    device = x.device
    original_dtype = x.dtype
    
    # Check if FP8 and torch._scaled_mm are available
    has_scaled_mm = hasattr(torch, '_scaled_mm')
    fp8_available = device.type == 'cuda' and TRITON_AVAILABLE
    
    if not (has_scaled_mm and fp8_available):
        # Fallback to FP16 path
        logger.debug("[TERNARY FP8] FP8 not available, using FP16 fallback")
        return _unpack_i2s_and_linear_impl(x, weight_packed, alpha, bias, K, original_dtype)
    
    try:
        # Check FP8 tensor support
        _ = torch.tensor([1.0], device=device).to(torch.float8_e4m3fn)
        
        # Unpack weights directly to FP8
        weight_unpacked_fp8 = torch.empty(N, K, device=device, dtype=torch.float8_e4m3fn)
        
        weight_packed_contig = weight_packed.contiguous()
        alpha_contig = alpha.contiguous()
        num_packed_cols = weight_packed.shape[1]
        
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
        
        _i2s_unpack_kernel[grid](
            weight_packed_contig, alpha_contig, weight_unpacked_fp8,
            N, K, num_packed_cols,
            weight_packed_contig.stride(0), weight_packed_contig.stride(1),
            weight_unpacked_fp8.stride(0), weight_unpacked_fp8.stride(1),
            BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
        )
        
        # Quantize activations to FP8
        # Flatten batch dimensions: x is (*, K) -> (M, K)
        x_shape = x.shape
        x_2d = x.reshape(-1, K)
        M = x_2d.shape[0]
        
        x_fp8, scale_x = quantize_fp8_with_scale(x_2d, dim=1)  # (M, K), (M, 1)
        
        # For ternary weights: use uniform scale since they're already quantized
        # The values are {-1, 0, 1} × alpha, which FP8 can represent exactly
        scale_w = torch.ones(1, N, device=device, dtype=torch.float32)
        
        # Prepare weight transpose for matmul: need (K, N) for x @ w.T
        w_T = weight_unpacked_fp8.T.contiguous()  # (K, N)
        w_T_colmajor = w_T.T.contiguous().T  # Force column-major layout
        
        x_fp8_contig = x_fp8.contiguous()
        
        # Perform FP8 tensor core matmul
        out = torch._scaled_mm(
            x_fp8_contig,
            w_T_colmajor,
            scale_a=scale_x,
            scale_b=scale_w,
            out_dtype=torch.bfloat16
        )
        
        # Add bias if present
        if bias is not None:
            out = out + bias
        
        # Restore original shape and dtype
        out = out.reshape(*x_shape[:-1], N)
        if out.dtype != original_dtype:
            out = out.to(original_dtype)
        
        return out
        
    except Exception as e:
        # Fallback to FP16 path on any error
        logger.debug(f"[TERNARY FP8] FP8 path failed, using FP16 fallback: {e}")
        return _unpack_i2s_and_linear_impl(x, weight_packed, alpha, bias, K, original_dtype)


def quantize_fp8_with_scale(tensor: torch.Tensor, dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to FP8 with per-row or per-column scaling for torch._scaled_mm.
    
    Args:
        tensor: Input tensor to quantize
        dim: 1 for per-row scaling (activations), 0 for per-column scaling (weights)
    
    Returns:
        tensor_fp8: Quantized FP8 tensor
        scale: Scale factors (in FP32)
    """
    FP8_MAX = 448.0  # Max value for float8_e4m3fn
    tensor_float = tensor.float()
    
    # Compute max absolute value along specified dimension
    abs_max = tensor_float.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max / FP8_MAX).clamp(min=1e-12)
    
    # Scale and clamp
    tensor_scaled = (tensor_float / scale).clamp(-FP8_MAX, FP8_MAX)
    tensor_fp8 = tensor_scaled.to(torch.float8_e4m3fn)
    
    return tensor_fp8, scale.to(torch.float32)


def replace_parameter(layer: nn.Module, name: str, new_param: torch.Tensor) -> None:
    """Replace a parameter in a layer, preserving attributes."""
    if hasattr(layer, name):
        old_param = getattr(layer, name)
        param_cls = type(old_param)
        requires_grad = False
        
        if isinstance(old_param, Parameter):
            new_param_obj = Parameter(new_param, requires_grad=requires_grad)
            for key, value in vars(old_param).items():
                if key not in ('_cdata', '_backward_hooks'):
                    try:
                        setattr(new_param_obj, key, value)
                    except AttributeError:
                        pass
        else:
            new_param_obj = new_param
        
        delattr(layer, name)
        setattr(layer, name, new_param_obj)
        
        if isinstance(new_param_obj, Parameter):
            layer._parameters[name] = new_param_obj


def _device_cache_key(device: torch.device) -> Tuple[str, int]:
    """Return a stable key for caching tensors per device."""
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type == "cuda":
        return (device.type, device.index if device.index is not None else torch.cuda.current_device())
    return (device.type, -1)


def _get_cached_tensor(
    layer: nn.Module,
    cache_attr: str,
    key: Tuple[Any, ...],
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Fetch or allocate a tensor cache entry attached to the layer."""
    cache = getattr(layer, cache_attr, None)
    if cache is None:
        cache = {}
        setattr(layer, cache_attr, cache)
    tensor = cache.get(key)
    if (
        tensor is None
        or tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
    ):
        tensor = torch.empty(shape, device=device, dtype=dtype)
        cache[key] = tensor
    return tensor


def _get_fp16_fallback_weight(
    layer: nn.Module,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    base = getattr(layer, "_ternary_weight_fp16", None)
    if base is None:
        return None
    cache = getattr(layer, "_ternary_weight_fp16_cache", None)
    if cache is None:
        cache = {}
        setattr(layer, "_ternary_weight_fp16_cache", cache)
    key = (dtype, _device_cache_key(device))
    tensor = cache.get(key)
    if tensor is None:
        tensor = base
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        if tensor.device != device:
            tensor = tensor.to(device, non_blocking=True)
        cache[key] = tensor
    return tensor


@dataclass
class TernaryConfig(QuantizationConfig):
    """Config class for ternary quantization.
    
    Args:
        threshold_scale: Scale factor for ternary quantization threshold (0.0-1.0)
            Lower values result in more aggressive quantization and sparsity.
        storage_mode: Storage mode - "i2s" (8x compression) or "fp16" (no compression, debugging)
            Default is "i2s" for best memory efficiency.
        use_fp8: Whether to use FP8 tensor cores for inference (requires CUDA, torch._scaled_mm)
            Provides faster inference with FP8 tensor cores. Default is False.
        use_bitnet_kernel: Whether to use optimized BitNet-style CUDA kernel for inference.
            Provides significant speedups (1.5-28x over unpack+linear) while maintaining
            exact per-column alpha correctness. Requires CUDA and compiled extension. Default is True.
    """

    threshold_scale: float = 0.7
    storage_mode: str = "i2s"  # "i2s" or "fp16"
    use_fp8: bool = False
    use_bitnet_kernel: bool = True

    def __post_init__(self):
        if not (0.0 < self.threshold_scale < 1.0):
            raise ValueError("threshold_scale must be between 0 and 1.")
        self.storage_mode = self.storage_mode.lower()
        if self.storage_mode not in ("i2s", "fp16"):
            raise ValueError(f"storage_mode must be 'i2s' or 'fp16', got '{self.storage_mode}'")

    @staticmethod
    def get_name() -> str:
        return "ternary"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """Return config filenames to search for quantization params."""
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TernaryConfig":
        threshold_scale = config.get("threshold_scale", 0.7)
        storage_mode = config.get("storage_mode", "i2s")
        use_fp8 = config.get("use_fp8", False)
        use_bitnet_kernel = config.get("use_bitnet_kernel", True)
        return cls(threshold_scale, storage_mode, use_fp8, use_bitnet_kernel)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["LinearMethodBase"]:
        pref = prefix or ""
        lower_pref = pref.lower()

        if ("embed" in lower_pref) or ("lm_head" in lower_pref):
            return None

        # Skip MoE gates/routers, but allow MLP gate_proj (SwiGLU)
        # "gate_proj" is standard in Qwen/Llama MLPs and should be quantized
        if (("gate" in lower_pref and "gate_proj" not in lower_pref) or 
            ("router" in lower_pref)):
            return None

        if isinstance(layer, LinearBase):
            return TernaryLinearMethod(self)
        
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_min_capability(self) -> int:
        """Minimum GPU capability required (SM version)."""
        return 0
    
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        """Supported activation dtypes."""
        return [torch.float16, torch.bfloat16]


class TernaryLinearMethod(LinearMethodBase):
    """Linear method for ternary quantization."""
    
    def __init__(self, quant_config: TernaryConfig):
        self.quant_config = quant_config

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor, shard_id: Optional[str] = None):
        param.data.copy_(loaded_weight)

    def create_weights(
        self,
        layer: LinearBase,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": self.weight_loader,
        })

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Apply ternary quantization to layer weights after loading."""
        try:
            weight = layer.weight.data
            original_dtype = weight.dtype
            N, K = weight.shape
            device = weight.device
            
            logger.info(f"[TERNARY] Quantizing layer: {weight.shape}")
            
            # Store original FP16 weights for fast fallback path
            weight_fp16 = weight.to(torch.bfloat16 if original_dtype == torch.bfloat16 else torch.float16)
            layer.register_buffer('_ternary_weight_fp16', weight_fp16, persistent=False)
            
            force_cleanup_and_sync(device)
            mem_before = measure_layer_memory_accurate(layer, device)
            gpu_snapshot_before = get_memory_snapshot(device)
            
            weight_memory_before = get_tensor_memory_bytes(weight)
            layer_memory_before = mem_before['total_theoretical_bytes']
            
            weight_fp32 = weight.float()
            absW = weight_fp32.abs()
            dim = 0
            th = self.quant_config.threshold_scale * absW.mean(dim, keepdim=True)
            mask = absW > th
            mask_f = mask.to(weight_fp32.dtype)
            alpha = (absW * mask_f).sum(dim, keepdim=True) / mask_f.sum(dim, keepdim=True).clamp(min=1)
            alpha = torch.where(torch.isfinite(alpha), alpha, torch.full_like(alpha, 1e-6))
            weight_ternary = weight_fp32.sign() * alpha * mask_f
            
            if self.quant_config.storage_mode == "i2s":
                weight_ternary_sign = torch.where(
                    mask,
                    weight_fp32.sign(),
                    torch.zeros_like(weight_fp32)
                ).to(torch.int8)
                
                weight_packed_simple = pack_i2s_weights(weight_ternary_sign.float())
                replace_parameter(layer, 'weight', weight_packed_simple)

                bitnet_packed = False
                if BITNET_PACK_AVAILABLE:
                    try:
                        weight_bitnet = convert_weight_int8_to_int2(weight_ternary_sign).contiguous()
                        if device.type == 'cuda':
                            weight_bitnet = weight_bitnet.to(device, non_blocking=True)
                        layer.register_buffer('ternary_weight_bitnet', weight_bitnet, persistent=False)
                        layer._ternary_weight_bitnet_ptr = weight_bitnet.data_ptr()
                        bitnet_packed = True
                    except Exception as e:
                        logger.warning(f"[TERNARY V4] BitNet packing failed ({e}), kernel path disabled")
                else:
                    logger.debug("[TERNARY V4] BitNet packing unavailable; falling back to unpack path")

                alpha_flat = alpha.view(-1).contiguous()
                
                # Store FP32 alpha for fallback/unpacking
                layer.register_buffer('ternary_alpha', alpha_flat.to(torch.float32), persistent=False)
                
                # Quantize alpha to int8 for V4 kernel (done once at load time)
                if BITNET_CUDA_AVAILABLE and self.quant_config.use_bitnet_kernel and device.type == 'cuda':
                    try:
                        alpha_q, alpha_scale = quantize_alpha_int8(alpha_flat)
                        alpha_scale_tensor = torch.tensor([alpha_scale], device=device, dtype=torch.float32)
                        
                        layer.register_buffer('ternary_alpha_q', alpha_q.contiguous(), persistent=False)
                        layer.register_buffer('ternary_alpha_scale', alpha_scale_tensor, persistent=False)
                        
                        # Cache pointers as direct attributes (eliminates getattr overhead)
                        layer._ternary_weight_bitnet_ptr = layer.ternary_weight_bitnet.data_ptr()
                        layer._ternary_alpha_q_ptr = layer.ternary_alpha_q.data_ptr()
                        layer._ternary_alpha_scale_ptr = layer.ternary_alpha_scale.data_ptr()
                        
                        # Pre-allocate buffers for common decode batch sizes (M=1 is most common)
                        # This eliminates cache lookup overhead entirely
                        common_batch_sizes = [1, 2, 4, 8, 16, 32]
                        for M_prealloc in common_batch_sizes:
                            if M_prealloc <= DEFAULT_PREFILL_SKIP_M:
                                # Pre-allocate activation quantization buffers
                                setattr(layer, f'_ternary_act_int8_M{M_prealloc}', 
                                       torch.empty(M_prealloc, K, device=device, dtype=torch.int8))
                                setattr(layer, f'_ternary_act_scale_M{M_prealloc}',
                                       torch.empty(M_prealloc, device=device, dtype=torch.bfloat16))
                                # Pre-allocate output buffer
                                setattr(layer, f'_ternary_output_M{M_prealloc}',
                                       torch.empty(M_prealloc, N, device=device, dtype=torch.bfloat16))
                        
                        logger.info(f"[TERNARY V4] Quantized alpha for {weight.shape}: scale={alpha_scale:.6f}")
                        logger.info(f"[TERNARY V4] Pre-allocated buffers for batch sizes: {common_batch_sizes}")
                    except Exception as e:
                        logger.warning(f"[TERNARY V4] Alpha quantization failed ({e}), V4 kernel will not be available")
                
                layer._ternary_original_dtype = original_dtype
                layer._ternary_i2s_enabled = True
                layer._ternary_fp16_enabled = False
                layer._ternary_bitnet_enabled = bitnet_packed
                layer._ternary_weight_shape = (N, K)
                layer._ternary_K = K
                layer._ternary_N = N
                
                del weight_fp32, absW, mask, mask_f, weight_ternary, weight_ternary_sign
                del weight
                
                force_cleanup_and_sync(device)
                mem_after = measure_layer_memory_accurate(layer, device)
                gpu_snapshot_after = get_memory_snapshot(device)
                
                weight_memory_after = get_tensor_memory_bytes(layer.weight.data)
                alpha_memory = get_tensor_memory_bytes(layer.ternary_alpha)
                if hasattr(layer, 'ternary_alpha_fp16'):
                    alpha_memory += get_tensor_memory_bytes(layer.ternary_alpha_fp16)
                layer_memory_after = mem_after['total_theoretical_bytes']
                
                theoretical_reduction_bytes = weight_memory_before - (weight_memory_after + alpha_memory)
                
                if layer_memory_after >= layer_memory_before:
                    logger.error(f"[TERNARY] ✗ ERROR: Layer tensor memory did not decrease!")
                
                if gpu_snapshot_before['allocated'] > 0 and gpu_snapshot_after['allocated'] > 0:
                    gpu_allocated_delta = gpu_snapshot_after['allocated'] - gpu_snapshot_before['allocated']
                    if gpu_allocated_delta > theoretical_reduction_bytes * 2:
                        logger.warning(f"[TERNARY] ⚠️  GPU memory increased unexpectedly by "
                                     f"{format_bytes(gpu_allocated_delta)}")
                
            elif self.quant_config.storage_mode == "fp16":
                weight_quantized = weight_ternary.to(original_dtype)
                replace_parameter(layer, 'weight', weight_quantized)
                layer.register_buffer('ternary_alpha', torch.ones(K, device=device, dtype=original_dtype), persistent=False)
                
                layer._ternary_original_dtype = original_dtype
                layer._ternary_i2s_enabled = False
                layer._ternary_fp16_enabled = True
                layer._ternary_bitnet_enabled = False
                
                del weight_fp32, absW, mask, mask_f, weight_ternary
                del weight
                
                force_cleanup_and_sync(device)
                mem_after = measure_layer_memory_accurate(layer, device)
                gpu_snapshot_after = get_memory_snapshot(device)
                
                weight_memory_after = get_tensor_memory_bytes(weight_quantized)
                layer_memory_after = mem_after['total_theoretical_bytes']
                
                if gpu_snapshot_before['allocated'] > 0 and gpu_snapshot_after['allocated'] > 0:
                    gpu_allocated_delta = gpu_snapshot_after['allocated'] - gpu_snapshot_before['allocated']
                    if gpu_allocated_delta > weight_memory_before * 0.2:
                        logger.warning(f"[TERNARY] ⚠️  GPU memory increased unexpectedly by "
                                     f"{format_bytes(gpu_allocated_delta)}")
                
            else:
                raise ValueError(f"Unknown storage mode: {self.quant_config.storage_mode}")
                
        except Exception as e:
            logger.error(f"Error during ternary quantization: {e}. Keeping original weights.")
            if logger.isEnabledFor(logging.DEBUG):
                import traceback
                logger.debug(f"Quantization error traceback: {traceback.format_exc()}")

    @torch.inference_mode()
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Early setup - compute once
        weight = layer.weight
        x_compute = x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.bfloat16)
        x_shape = x_compute.shape
        b_compute = None if bias is None else (bias if bias.dtype in (torch.float16, torch.bfloat16) else bias.to(x_compute.dtype))
        
        # Get layer attributes once
        weight_shape = getattr(layer, "_ternary_weight_shape", None)
        if weight_shape is None:
            # Early exit if not quantized
            return F.linear(x_compute, weight, b_compute)
        
        N, K = weight_shape
        x_2d = x_compute.reshape(-1, K)
        M = x_2d.shape[0]
        
        # Cache profiling state to avoid repeated checks
        prof_enabled = _ternary_profiler.enabled
        
        # Fast profiler context (always start, stop in both paths)
        prof_ctx = _fast_apply_profiler.start()
        
        # V4 kernel path: Try optimized kernel first
        bitnet_enabled = getattr(layer, '_ternary_bitnet_enabled', False)
        if (
            bitnet_enabled
            and self.quant_config.use_bitnet_kernel
            and BITNET_CUDA_AVAILABLE
            and BITNET_LIB is not None
            and weight.dtype == torch.uint8
            and x_compute.is_cuda
            and M <= DEFAULT_PREFILL_SKIP_M
            and (N, K) in SUPPORTED_V4_NK_SHAPES
        ):
            # OPTIMIZATION: Use pre-allocated buffers if available (eliminates cache lookup)
            # For common batch sizes (M=1,2,4,8,16,32), buffers are pre-allocated at init
            buf_attr_int8 = f'_ternary_act_int8_M{M}'
            buf_attr_scale = f'_ternary_act_scale_M{M}'
            buf_attr_output = f'_ternary_output_M{M}'
            
            if hasattr(layer, buf_attr_int8):
                # Use pre-allocated buffers (zero overhead)
                out_int8 = getattr(layer, buf_attr_int8)
                out_scale = getattr(layer, buf_attr_scale)
                output = getattr(layer, buf_attr_output)
                # Ensure buffers match expected shape (should always match, but safety check)
                if out_int8.shape[0] < M:
                    # Fallback to dynamic allocation for unexpected shapes
                    out_int8 = torch.empty(M, K, device=x_compute.device, dtype=torch.int8)
                    out_scale = torch.empty(M, device=x_compute.device, dtype=torch.bfloat16)
                    output = torch.empty(M, N, device=x_compute.device, dtype=torch.bfloat16)
            else:
                # Fallback to cache for uncommon batch sizes
                dev_key = _device_cache_key(x_compute.device)
                act_key = (M, K, dev_key)
                scale_key = (M, dev_key)
                out_key = (M, N, dev_key)
                
                out_int8 = _get_cached_tensor(
                    layer, "_ternary_act_int8_cache", act_key,
                    (M, K), torch.int8, x_compute.device
                )
                out_scale = _get_cached_tensor(
                    layer, "_ternary_act_scale_cache", scale_key,
                    (M,), torch.bfloat16, x_compute.device
                )
                output = _get_cached_tensor(
                    layer, "_ternary_v4_output_cache", out_key,
                    (M, N), torch.bfloat16, x_compute.device
                )
            
            # Profiling for quantization
            if prof_enabled:
                _ternary_profiler._safe_sync()
                quant_start = time.time()
            
            # OPTIMIZATION: Fast path for M=1 (decode) - use CUDA fast quantizer if available
            if M == 1 and BITNET_CUDA_ACT_QUANT_AVAILABLE and TERNARY_USE_CUDA_ACT_QUANT:
                # CUDA fast quantizer is optimized for M=1 decode path
                fast_result = _quantize_activation_fast_cuda(
                    x_2d, layer.ternary_alpha,
                    out_int8=out_int8, out_scale=out_scale
                )
                if fast_result is not None:
                    x_int8, x_scale = fast_result
                else:
                    # Fallback to Triton/PyTorch
                    x_int8, x_scale = quantize_activation_prescale_fused(
                        x_2d, layer.ternary_alpha,
                        out_int8=out_int8, out_scale=out_scale
                    )
            else:
                # General path (M>1 or CUDA quantizer not available)
                x_int8, x_scale = quantize_activation_prescale_fused(
                    x_2d, layer.ternary_alpha,
                    out_int8=out_int8, out_scale=out_scale
                )
            
            if prof_enabled:
                _ternary_profiler._safe_sync()
                quant_duration = (time.time() - quant_start) * 1000
                _ternary_profiler.record(f"quantize_activation_M{M}_K{K}", quant_duration)
            
            # OPTIMIZATION: Use cached pointers (set at init, zero overhead)
            weight_ptr = layer._ternary_weight_bitnet_ptr
            alpha_q_ptr = layer._ternary_alpha_q_ptr
            alpha_scale_ptr = layer._ternary_alpha_scale_ptr
            
            # Profiling for kernel
            if prof_enabled:
                _ternary_profiler._safe_sync()
                kernel_start = time.time()
            
            # Call V4 kernel
            ret_code = BITNET_LIB.bitlinear_int8xint2_v4_simple(
                ctypes.c_void_p(x_int8.data_ptr()),
                ctypes.c_void_p(weight_ptr),
                ctypes.c_void_p(alpha_q_ptr),
                ctypes.c_void_p(alpha_scale_ptr),
                ctypes.c_void_p(output.data_ptr()),
                ctypes.c_void_p(x_scale.data_ptr()),
                ctypes.c_int(M),
                ctypes.c_int(N),
                ctypes.c_int(K),
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            )
            
            if ret_code == 0:
                if prof_enabled:
                    _ternary_profiler._safe_sync()
                    kernel_duration = (time.time() - kernel_start) * 1000
                    _ternary_profiler.record(f"ternary_apply_v4_kernel_M{M}_N{N}_K{K}", kernel_duration)
                
                # Post-process output
                output = output.view(*x_shape[:-1], N)
                if b_compute is not None:
                    output = output + b_compute
                if output.dtype != x.dtype:
                    output = output.to(x.dtype)
                
                _fast_apply_profiler.stop(prof_ctx)
                return output
        
        # Fallback path: Use FP16 weights
        if prof_enabled:
            _ternary_profiler._safe_sync()
            fallback_start = time.time()
            shape_info = f"M{M}_N{N}_K{K}"
        
        # Get or create FP16 weight cache
        weight_fp16 = _get_fp16_fallback_weight(layer, x_compute.dtype, x_compute.device)
        if weight_fp16 is None:
            i2s_enabled = getattr(layer, '_ternary_i2s_enabled', False)
            if i2s_enabled and weight.dtype == torch.uint8:
                weight_fp16 = unpack_i2s_weights(weight, K, layer.ternary_alpha, x_compute.dtype)
            else:
                weight_fp16 = weight.to(x_compute.dtype)
            layer.register_buffer("_ternary_weight_fp16", weight_fp16, persistent=False)
            setattr(layer, "_ternary_weight_fp16_cache", {})
            weight_fp16 = _get_fp16_fallback_weight(layer, x_compute.dtype, x_compute.device)
        
        # Compute linear layer
        out = F.linear(x_compute, weight_fp16, b_compute)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        
        if prof_enabled:
            _ternary_profiler._safe_sync()
            fallback_duration = (time.time() - fallback_start) * 1000
            _ternary_profiler.record(f"ternary_apply_fp16_fallback_{shape_info}", fallback_duration)
        
        _fast_apply_profiler.stop(prof_ctx)
        return out


__all__ = ["TernaryConfig", "TernaryLinearMethod"]
