# Mangrove Production + Performance Playbook

This playbook focuses on two goals:

1. **Reliable production deploy path** (SGLang + `/scratch/mangrove`)
2. **Aggressive kernel speed iteration** (ternary kernels, decode latency, throughput)

## 0) One-click run (fresh clone)

From the repo root:

```bash
bash one_click_run_mangrove.sh
```

This launcher bootstraps `.venv` and runtime deps if needed, optionally wires
`third_party/ternarykernels`, builds/syncs kernels when source is available,
and starts serving `/scratch/mangrove`.

## 1) Monorepo-style setup

From the `sglang` root:

```bash
bash util/setup_mangrove_monorepo.sh --source /home/ubuntu/ternarykernels --model-path /scratch/mangrove
```

This creates a stable monorepo link:

- `third_party/ternarykernels -> /home/ubuntu/ternarykernels`

And writes:

- `.mangrove.env`

## 2) Build + sync kernel libraries into SGLang

```bash
bash util/build_sync_ternary_kernels.sh
```

Useful variants:

```bash
bash util/build_sync_ternary_kernels.sh --cutlass-only
bash util/build_sync_ternary_kernels.sh --bitnet-only
bash util/build_sync_ternary_kernels.sh --arch 100a
bash util/build_sync_ternary_kernels.sh --sync-mode link
```

## 3) Launch production server

`run_server_ternary.sh` supports explicit production model path now:

```bash
bash run_server_ternary.sh i2s-tuned --model-path /scratch/mangrove --served-model-name mangrove-prod
```

If `/scratch/mangrove` exists, preset `mangrove` is auto-selected by default.

## 4) Profile node load for N users

Collect request metrics plus node CPU/RAM/GPU samples:

```bash
python util/profile_node_load.py --host 127.0.0.1 --port 30080 --users-list 1,2,4,8,16,32 --requests-per-user 8
```

Outputs:

- `node_load_profile_*/users_<N>.json` (per-run summary)
- `node_load_profile_*/users_<N>_samples.jsonl` (time series samples)
- `node_load_profile_*/sweep_summary.csv` (quick comparison across N)

## 5) Aggressively tune kernel-speed knobs

Sweep kernel-centric profiles and rank by decode p50 and high-concurrency TPS:

```bash
bash util/sweep_ternary_kernels.sh i2s-tuned
```

Quick sweep:

```bash
QUICK=1 bash util/sweep_ternary_kernels.sh i2s-fp8
```

With production model path:

```bash
MODEL_PATH=/scratch/mangrove bash util/sweep_ternary_kernels.sh i2s-tuned
```

Outputs:

- `kernel_sweep_<mode>_*/kernel_sweep_summary.csv`
- per-profile server + benchmark logs and JSON results

## 6) Practical optimization loop

1. Build latest kernels (`build_sync_ternary_kernels.sh`)
2. Run `sweep_ternary_kernels.sh` for kernel runtime knobs
3. Run `profile_node_load.py` for target user concurrency
4. Keep only profiles that improve both:
   - decode p50/p95 (interactive latency)
   - cmax tokens/sec (throughput at load)

Collect kernel-path/fallback stats while serving:

```bash
SGLANG_TERNARY_COLLECT_PATH_STATS=1 \
bash run_server_ternary.sh i2s-maxspeed --model-path /scratch/mangrove
```

This writes per-process JSON reports (`logs/ternary_runtime_<pid>.json`) with:
- `path_counts` (which ternary kernels are actually hit)
- `fallback_counts` (why execution falls back to fp16)

## 7) Notes from H100 tuning

- `SGLANG_TERNARY_I2S_SPLITS=1` often improves mid-concurrency throughput and
  significantly improves p50 latency at moderate load (for this setup at c16).
- `TERNARY_ENABLE_PDL=1` should generally stay enabled for decode latency.
- `TERNARY_FUSE_RMSNORM_QKV=1` remains a good default for decode path speed.
- On H100 (sm90), `libternary_cutlass_sm100.so` is not applicable and should be
  considered optional; BitNet kernels are the active fast path.

## 8) Capacity testing for 1s TTFT + KV per-user limits

Single running server:

```bash
python util/ttft_kv_capacity.py \
  --host 127.0.0.1 --port 30080 \
  --concurrency-levels 1,2,4,8,16,24,32,40,48,64 \
  --requests-per-level 20 \
  --ttft-target-ms 1000 \
  --max-new-tokens 128
```

FP16 vs ternary comparison end-to-end (auto-launches both modes):

```bash
MODEL_PATH=/scratch/mangrove \
bash util/compare_ternary_fp16_capacity.sh
```

Outputs include:

- max users meeting TTFT target
- per-concurrency KV budget per user (`max_total_num_tokens / users`)
- throughput and latency comparisons across concurrency levels

## 9) Freeze a reproducible baseline matrix (Phase 1)

Use the one-command baseline freezer to produce a single artifact bundle with:

- corpus-backed support-envelope worksheet (wafer citations)
- baseline ternary runtime path/fallback counters
- M=1 decode + TTFT/KV + concurrency benchmark outputs
- optional kernel sweep + fp16-vs-ternary capacity compare artifacts

```bash
bash util/freeze_ternary_baseline_matrix.sh \
  --mode i2s-tuned \
  --model-path /scratch/mangrove
```

By default, this command skips the long optional sweep/compare phases.  
Enable them explicitly when you want full extended coverage:

```bash
bash util/freeze_ternary_baseline_matrix.sh \
  --mode i2s-tuned \
  --model-path /scratch/mangrove \
  --run-kernel-sweep 1 \
  --run-capacity-compare 1
```

The baseline freezer now enforces ternary kernel availability by default for
`i2s*` / `ternary` modes. If `libternary_bitnet.so` is missing or not loaded,
the run exits early to avoid misleading (FP16-fallback) tok/s results.

Quick iteration mode:

```bash
bash util/freeze_ternary_baseline_matrix.sh --quick
```

Preview planned commands without running workloads:

```bash
bash util/freeze_ternary_baseline_matrix.sh --dry-run
```

Output directory (`baseline_matrix_<timestamp>/`) includes:

- `baseline_summary.json` + `baseline_summary.md` (single-glance report)
- `support_envelope_matrix.csv` + `support_envelope_matrix.md`
- `baseline_family_counters.json` (linear/moe family counters from runtime stats)
- benchmark JSON/log artifacts (`baseline_m1_decode.json`, `baseline_ttft_kv.json`, `baseline_concurrency.json`)
- `commands.log` + `config.env` for reproducibility

## 10) Hybrid swap (ternary + fp16)

Run both backends and an autoswap router:

```bash
bash util/run_hybrid_swap.sh
```

Defaults:

- public endpoint: `http://127.0.0.1:30080`
- ternary backend: `i2s-maxspeed` on `:30081`
- fp16 backend: `fp16 --dp <gpu_count(fp16_gpus)>` on `:30082`
- swap threshold: inflight `<=48` routes to ternary, above to fp16

GPU partitioning defaults:

- auto-detected from available GPUs
- on 8+ GPUs: ternary lane uses first 2 GPUs, fp16 lane uses remaining GPUs
- on 4-7 GPUs: ternary lane uses first GPU, fp16 lane uses remaining GPUs

Override example:

```bash
TERNARY_CUDA_VISIBLE_DEVICES=0 \
FP16_CUDA_VISIBLE_DEVICES=1,2,3 \
SWITCH_INFLIGHT_THRESHOLD=64 \
bash util/run_hybrid_swap.sh
```

Router introspection:

```bash
curl -s http://127.0.0.1:30080/routing_stats
```

## 11) H100 M>1 PoC probe (shape x M gate)

Before enabling any runtime M>1 ternary linear path, run the dedicated probe:

```bash
bash util/run_h100_mgt1_poc.sh
```

This invokes `util/benchmark_mgt1_linear.py` and writes:

- `mgt1_linear_probe_<timestamp>/probe.json`
- `mgt1_linear_probe_<timestamp>/probe.log`

The probe reports, for each `(M, N, K)` case:

- kernel return status (`v4_batch_megafused_v2_launch`)
- correctness deltas vs configurable reference (`m1_kernel` default, or `dense`) using `max_abs`, `max_rel`, `non_finite`
- latency stats (`avg`, `p50`, `p95`)

Runtime has a guarded M>1 switch for controlled bring-up only:

```bash
SGLANG_TERNARY_ENABLE_EXPERIMENTAL_MGT1_LINEAR=1 \
SGLANG_TERNARY_MGT1_VALIDATE=1 \
SGLANG_TERNARY_MGT1_VALIDATE_REF=m1_kernel \
SGLANG_TERNARY_MGT1_VALIDATE_ROWS=8 \
SGLANG_TERNARY_MGT1_VALIDATE_MAX_ABS=0.08 \
SGLANG_TERNARY_MGT1_VALIDATE_MAX_REL=0.25 \
SGLANG_TERNARY_MGT1_VALIDATE_REL_FLOOR=0.05 \
SGLANG_TERNARY_MGT1_DISABLE_ON_FAIL=1 \
bash run_server_ternary.sh i2s-tuned --model-path /scratch/mangrove
```

Keep this disabled in production until probe + validation gates pass.

`i2s-tuned` now defaults `SGLANG_TERNARY_MOE_TRITON=1` (override with `SGLANG_TERNARY_MOE_TRITON=0` when debugging fallback behavior).

To profile batched MoE stage split (`gateup`, `silu_mul`, `down`, `combine`) in server logs:

```bash
SGLANG_TERNARY_MOE_BATCHED_PROFILE=1 \
bash run_server_ternary.sh i2s-tuned --model-path /scratch/mangrove
```

Summarize those log lines into stage-share artifacts:

```bash
python util/summarize_moe_batched_profile.py \
  --log /path/to/server.log \
  --output-dir /path/to/profile_summary
```

Combine policy (for `top_k > 2`) defaults to Triton reduce:

```bash
# default (recommended on H100)
SGLANG_TERNARY_MOE_COMBINE_TRITON=1

# legacy fallback policy
SGLANG_TERNARY_MOE_COMBINE_TRITON=0
SGLANG_TERNARY_MOE_COMBINE_TORCH_MAX_M=32
```

Gate/down Triton tile overrides for quick H100 sweeps (optional):

```bash
SGLANG_TERNARY_MOE_GATEUP_BLOCK_SIZE_M=16
SGLANG_TERNARY_MOE_GATEUP_BLOCK_SIZE_N=128
SGLANG_TERNARY_MOE_GATEUP_BLOCK_SIZE_K=256
SGLANG_TERNARY_MOE_GATEUP_GROUP_SIZE_M=1
SGLANG_TERNARY_MOE_GATEUP_NUM_WARPS=8
SGLANG_TERNARY_MOE_GATEUP_NUM_STAGES=5

SGLANG_TERNARY_MOE_DOWN_BLOCK_SIZE_M=16
SGLANG_TERNARY_MOE_DOWN_BLOCK_SIZE_N=128
SGLANG_TERNARY_MOE_DOWN_BLOCK_SIZE_K=128
SGLANG_TERNARY_MOE_DOWN_GROUP_SIZE_M=64
SGLANG_TERNARY_MOE_DOWN_NUM_WARPS=4
SGLANG_TERNARY_MOE_DOWN_NUM_STAGES=3
```

Run the built-in gateup sweep harness (throughput + stage profile summary):

```bash
bash util/sweep_moe_gateup_configs.sh --model-path /scratch/mangrove
```

On H100, `gate_up` bucket `m>4` now defaults to a sweep-backed config
(`m16 n128 k256 g64 w4 s3`). Disable this if needed:

```bash
SGLANG_TERNARY_MOE_GATEUP_H100_TUNE=0 \
bash run_server_ternary.sh i2s-tuned --model-path /scratch/mangrove
```

Experimental high-batch INT8 MoE stage mode (keeps ternary packed weights, switches
`gate_up`/`down` compute path to INT8 Triton kernels):

```bash
# gate_up + down
SGLANG_TERNARY_MOE_INT8_MODE=both \
SGLANG_TERNARY_MOE_INT8_MIN_TOKENS=64 \
bash run_server_ternary.sh i2s-tuned --model-path /scratch/mangrove

# gate_up only
SGLANG_TERNARY_MOE_INT8_MODE=gateup \
bash run_server_ternary.sh i2s-tuned --model-path /scratch/mangrove
```

Use `SGLANG_TERNARY_MOE_INT8_STRICT=1` for debug bring-up (fail fast instead of
falling back to BF16 stage execution).

Current status: this path is **benchmark-only**. Component probes on H100 show
good speedups, but large output deltas vs BF16 baseline, so keep disabled for
production correctness until calibrated.

Quick run:

```bash
bash util/run_h100_mgt1_poc.sh --quick
```
