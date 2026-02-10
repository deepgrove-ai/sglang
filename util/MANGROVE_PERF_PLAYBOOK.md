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

## 9) Hybrid swap (ternary + fp16)

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
