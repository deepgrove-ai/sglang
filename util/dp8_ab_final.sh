#!/bin/bash
# DP8 A/B: Ternary (winning config) vs FP16 at concurrency 1,2,4,8,16,32,64
# This is the final validation benchmark.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SGLANG_DIR"
source "$SGLANG_DIR/activate_env.sh" 2>/dev/null || true

OUTDIR="${1:-$SGLANG_DIR/dp8_ab_final_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"

PORT=30080
CONCURRENCY_LEVELS=(1 2 4 8 16 32 64)
NUM_REQUESTS=60
MAX_TOKENS=150
PROMPT="Explain the key advantages of transformer architectures over RNNs in three paragraphs."

# Robust readiness gate: wait for router health + successful /generate probe
wait_dp8_ready() {
    local base_port=$1
    local max_wait=${2:-600}
    local start=$(date +%s)
    echo "  Waiting for DP8 readiness on router port ${base_port}..."
    
    while true; do
        local elapsed=$(( $(date +%s) - start ))
        if (( elapsed > max_wait )); then
            echo "  TIMEOUT after ${max_wait}s waiting for DP8 readiness"
            return 1
        fi
        
        # Check router health
        if ! curl -sf "http://localhost:${base_port}/health" >/dev/null 2>&1; then
            sleep 5
            continue
        fi
        
        # Probe: actually generate a token through the router
        local probe_resp
        probe_resp=$(curl -sf --max-time 120 "http://localhost:${base_port}/generate" \
            -H "Content-Type: application/json" \
            -d '{"text":"Hello","sampling_params":{"max_new_tokens":5,"temperature":0}}' 2>&1) || true
        if echo "$probe_resp" | grep -q '"text"'; then
            # Success! Do a few more probes to ensure all workers are warm
            local warm_ok=0
            for i in $(seq 1 4); do
                local wr
                wr=$(curl -sf --max-time 60 "http://localhost:${base_port}/generate" \
                    -H "Content-Type: application/json" \
                    -d '{"text":"Warmup probe '$i'","sampling_params":{"max_new_tokens":3,"temperature":0}}' 2>&1) || true
                if echo "$wr" | grep -q '"text"'; then
                    warm_ok=$((warm_ok + 1))
                fi
            done
            echo "  DP8 ready after ${elapsed}s (${warm_ok}/4 warm probes OK)"
            return 0
        fi
        
        sleep 10
    done
}

# Run benchmark at a given concurrency
run_bench() {
    local label=$1
    local conc=$2
    local outfile="$OUTDIR/${label}_c${conc}.json"
    
    echo "  Benchmarking ${label} C=${conc} (${NUM_REQUESTS} requests)..."
    
    python3 -c "
import json, time, concurrent.futures, requests

PORT = ${PORT}
C = ${conc}
N = ${NUM_REQUESTS}
PROMPT = '''${PROMPT}'''
MAX_TOKENS = ${MAX_TOKENS}

def send_request(i):
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f'http://localhost:{PORT}/generate',
            json={'text': PROMPT, 'sampling_params': {'max_new_tokens': MAX_TOKENS, 'temperature': 0.7}},
            timeout=120,
        )
        t1 = time.perf_counter()
        if resp.status_code != 200:
            return {'id': i, 'status': resp.status_code, 'latency': t1-t0, 'tokens': 0, 'error': resp.text[:200]}
        data = resp.json()
        text = data.get('text', '')
        # Estimate token count from text length (rough: ~4 chars/token)
        ntok = max(1, len(text) // 4)
        return {'id': i, 'status': 200, 'latency': t1-t0, 'tokens': ntok, 'tps': ntok/(t1-t0) if (t1-t0) > 0 else 0}
    except Exception as e:
        return {'id': i, 'status': -1, 'latency': time.perf_counter()-t0, 'tokens': 0, 'error': str(e)[:200]}

wall_start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=C) as pool:
    results = list(pool.map(send_request, range(N)))
wall_end = time.perf_counter()

ok = [r for r in results if r.get('status') == 200]
errs = [r for r in results if r.get('status') != 200]
lats = [r['latency'] for r in ok]
tps_vals = [r['tps'] for r in ok if r.get('tps', 0) > 0]

summary = {
    'concurrency': C,
    'total_requests': N,
    'ok': len(ok),
    'errors': len(errs),
    'wall_time_s': wall_end - wall_start,
    'avg_latency_s': sum(lats)/len(lats) if lats else 0,
    'p50_latency_s': sorted(lats)[len(lats)//2] if lats else 0,
    'p95_latency_s': sorted(lats)[int(len(lats)*0.95)] if lats else 0,
    'avg_tps': sum(tps_vals)/len(tps_vals) if tps_vals else 0,
    'total_throughput_rps': len(ok)/(wall_end - wall_start) if (wall_end - wall_start) > 0 else 0,
}
print(json.dumps(summary, indent=2))
with open('${outfile}', 'w') as f:
    json.dump({'summary': summary, 'results': results}, f, indent=2)
"
}

# Kill any existing servers
kill_servers() {
    pkill -f "sglang" 2>/dev/null || true
    pkill -f "sglang_router" 2>/dev/null || true
    sleep 5
}

echo "============================================"
echo "DP8 A/B Final Benchmark"
echo "Output: $OUTDIR"
echo "Concurrency levels: ${CONCURRENCY_LEVELS[*]}"
echo "============================================"

# ──────────────────────────────────────────────────────
# Phase 1: Ternary DP8 (winning config)
# ──────────────────────────────────────────────────────
echo ""
echo "=== PHASE 1: Ternary i2s-tuned DP8 (M>1 threshold=8, MoE Triton, INT8 both) ==="
kill_servers

export SGLANG_TERNARY_ENABLE_EXPERIMENTAL_MGT1_LINEAR=1
export SGLANG_TERNARY_MGT1_LINEAR_MAX_M=8
export SGLANG_TERNARY_MOE_TRITON=1
export SGLANG_TERNARY_MOE_INT8_MODE=both
export SGLANG_TERNARY_MOE_COMBINE_TRITON=1

bash "$SGLANG_DIR/run_server_ternary.sh" i2s-tuned --dp 8 --port $PORT \
    > "$OUTDIR/server_ternary.log" 2>&1 &
TERNARY_PID=$!
echo "  Ternary server PID: $TERNARY_PID"

if ! wait_dp8_ready $PORT 600; then
    echo "  FATAL: Ternary DP8 failed to start"
    kill $TERNARY_PID 2>/dev/null || true
    exit 1
fi

# Warmup
echo "  Warming up (5 requests)..."
for i in $(seq 1 5); do
    curl -sf --max-time 30 "http://localhost:${PORT}/generate" \
        -H "Content-Type: application/json" \
        -d '{"text":"Warmup","sampling_params":{"max_new_tokens":10,"temperature":0}}' >/dev/null 2>&1 || true
done
sleep 2

for C in "${CONCURRENCY_LEVELS[@]}"; do
    run_bench "ternary" "$C"
done

# Collect runtime stats
curl -sf "http://localhost:${PORT}/get_server_info" > "$OUTDIR/ternary_server_info.json" 2>/dev/null || true

echo "  Stopping ternary server..."
kill_servers

# ──────────────────────────────────────────────────────
# Phase 2: FP16 DP8 (baseline)
# ──────────────────────────────────────────────────────
echo ""
echo "=== PHASE 2: FP16 DP8 (baseline) ==="

# Clear ternary env
unset SGLANG_TERNARY_ENABLE_EXPERIMENTAL_MGT1_LINEAR 2>/dev/null || true
unset SGLANG_TERNARY_MGT1_LINEAR_MAX_M 2>/dev/null || true
unset SGLANG_TERNARY_MOE_TRITON 2>/dev/null || true
unset SGLANG_TERNARY_MOE_INT8_MODE 2>/dev/null || true
unset SGLANG_TERNARY_MOE_COMBINE_TRITON 2>/dev/null || true

bash "$SGLANG_DIR/run_server_ternary.sh" fp16 --dp 8 --port $PORT \
    > "$OUTDIR/server_fp16.log" 2>&1 &
FP16_PID=$!
echo "  FP16 server PID: $FP16_PID"

if ! wait_dp8_ready $PORT 600; then
    echo "  FATAL: FP16 DP8 failed to start"
    kill $FP16_PID 2>/dev/null || true
    exit 1
fi

# Warmup
echo "  Warming up (5 requests)..."
for i in $(seq 1 5); do
    curl -sf --max-time 30 "http://localhost:${PORT}/generate" \
        -H "Content-Type: application/json" \
        -d '{"text":"Warmup","sampling_params":{"max_new_tokens":10,"temperature":0}}' >/dev/null 2>&1 || true
done
sleep 2

for C in "${CONCURRENCY_LEVELS[@]}"; do
    run_bench "fp16" "$C"
done

curl -sf "http://localhost:${PORT}/get_server_info" > "$OUTDIR/fp16_server_info.json" 2>/dev/null || true

echo "  Stopping FP16 server..."
kill_servers

# ──────────────────────────────────────────────────────
# Summary comparison
# ──────────────────────────────────────────────────────
echo ""
echo "=== FINAL COMPARISON ==="
python3 -c "
import json, glob, os

outdir = '${OUTDIR}'
print(f'{'Conc':>5s}  {'Ternary TPS':>12s}  {'FP16 TPS':>10s}  {'Speedup':>8s}  {'Tern Lat(s)':>12s}  {'FP16 Lat(s)':>12s}  {'Lat Ratio':>10s}')
print('-' * 85)

for c in [1, 2, 4, 8, 16, 32, 64]:
    t_file = os.path.join(outdir, f'ternary_c{c}.json')
    f_file = os.path.join(outdir, f'fp16_c{c}.json')
    
    t_tps = t_lat = f_tps = f_lat = 0
    if os.path.exists(t_file):
        with open(t_file) as fh:
            td = json.load(fh)['summary']
            t_tps = td.get('avg_tps', 0)
            t_lat = td.get('p50_latency_s', 0)
    if os.path.exists(f_file):
        with open(f_file) as fh:
            fd = json.load(fh)['summary']
            f_tps = fd.get('avg_tps', 0)
            f_lat = fd.get('p50_latency_s', 0)
    
    speedup = t_tps / f_tps if f_tps > 0 else 0
    lat_ratio = t_lat / f_lat if f_lat > 0 else 0
    winner = '✓ TERN' if speedup > 1.0 else '✓ FP16'
    
    print(f'{c:>5d}  {t_tps:>12.1f}  {f_tps:>10.1f}  {speedup:>7.2f}x  {t_lat:>12.3f}  {f_lat:>12.3f}  {lat_ratio:>9.2f}x  {winner}')

print()
print('Results saved to:', outdir)
" | tee "$OUTDIR/comparison.txt"

echo ""
echo "Done! Full results in $OUTDIR/"
