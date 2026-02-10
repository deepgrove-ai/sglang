#!/usr/bin/env bash
# =============================================================================
# Aggressive Ternary Kernel Sweep
# =============================================================================
# Sweeps kernel-centric runtime knobs and reports:
#   - M=1 decode latency (util/benchmark_m1_decode.py)
#   - concurrency throughput/latency (util/benchmark_concurrency.py)
#
# This script is intended for production tuning loops where kernel speed is a
# first-class objective.
#
# Usage:
#   ./util/sweep_ternary_kernels.sh
#   ./util/sweep_ternary_kernels.sh i2s-tuned
#   QUICK=1 ./util/sweep_ternary_kernels.sh i2s-fp8
#   MODEL_PATH=/scratch/mangrove ./util/sweep_ternary_kernels.sh i2s-tuned
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(dirname "$SCRIPT_DIR")"
SERVER_SCRIPT="${SGLANG_DIR}/run_server_ternary.sh"
M1_BENCH="${SCRIPT_DIR}/benchmark_m1_decode.py"
CONC_BENCH="${SCRIPT_DIR}/benchmark_concurrency.py"

if [[ -n "${SGLANG_PYTHON:-}" ]]; then
    PYTHON="${SGLANG_PYTHON}"
elif [[ -x "${SGLANG_DIR}/.venv/bin/python" ]]; then
    PYTHON="${SGLANG_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
else
    echo "Error: no Python interpreter found (set SGLANG_PYTHON)."
    exit 1
fi

MODE="${1:-i2s-tuned}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-30080}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-360}"
COOLDOWN_SEC="${COOLDOWN_SEC:-5}"

# Benchmark controls
M1_STEPS="${M1_STEPS:-120}"
M1_RUNS="${M1_RUNS:-3}"
M1_WARMUP="${M1_WARMUP:-12}"
CONCURRENCY="${CONCURRENCY:-1,4,16,64}"
REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL:-64}"
MAX_TOKENS="${MAX_TOKENS:-256}"

# Optional model overrides (forwarded to run_server_ternary.sh)
MODEL_PATH="${MODEL_PATH:-}"
MODEL_PRESET="${MODEL_PRESET:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
TP_SIZE="${TP_SIZE:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${SGLANG_DIR}/kernel_sweep_${MODE}_${TIMESTAMP}"
mkdir -p "${OUT_DIR}"

echo "=============================================================="
echo "Aggressive Ternary Kernel Sweep"
echo "=============================================================="
echo "Mode:             ${MODE}"
echo "Server:           http://${SERVER_HOST}:${SERVER_PORT}"
echo "M1 steps/runs:    ${M1_STEPS}/${M1_RUNS}"
echo "Concurrency:      ${CONCURRENCY}"
echo "Requests/level:   ${REQUESTS_PER_LEVEL}"
echo "Output dir:       ${OUT_DIR}"
echo "=============================================================="

wait_for_server() {
    local host="$1"
    local port="$2"
    local timeout="$3"
    local start
    start="$(date +%s)"
    echo -n "Waiting for server"
    while true; do
        if curl -s "http://${host}:${port}/health" > /dev/null 2>&1; then
            echo " ready"
            return 0
        fi
        local now elapsed
        now="$(date +%s)"
        elapsed="$(( now - start ))"
        if [[ "${elapsed}" -ge "${timeout}" ]]; then
            echo " timeout (${timeout}s)"
            return 1
        fi
        echo -n "."
        sleep 2
    done
}

kill_server_on_port() {
    local port="$1"
    local pids
    pids="$(lsof -t -i:"${port}" 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
        echo "Killing existing process(es) on :${port}: ${pids}"
        echo "${pids}" | xargs -r kill -9 2>/dev/null || true
        sleep 2
    fi
}

build_server_args() {
    local args=("${MODE}")
    if [[ -n "${MODEL_PATH}" ]]; then
        args+=(--model-path "${MODEL_PATH}")
    fi
    if [[ -n "${MODEL_PRESET}" ]]; then
        args+=(--model "${MODEL_PRESET}")
    fi
    if [[ -n "${SERVED_MODEL_NAME}" ]]; then
        args+=(--served-model-name "${SERVED_MODEL_NAME}")
    fi
    if [[ -n "${TP_SIZE}" ]]; then
        args+=(--tp "${TP_SIZE}")
    fi
    printf "%q " "${args[@]}"
}

# Profiles: name + env var assignments.
# Keep profiles kernel-focused and production-safe by default.
if [[ "${QUICK:-0}" == "1" ]]; then
    PROFILE_NAMES=(
        "baseline"
        "decode_aggressive"
        "split4_prefill"
        "no_pdl"
    )
    PROFILE_ENVS=(
        ""
        "TERNARY_ENABLE_PDL=1 TERNARY_FUSE_RMSNORM_QKV=1 SGLANG_TERNARY_I2S_SPLITS=1"
        "TERNARY_ENABLE_PDL=1 TERNARY_FUSE_RMSNORM_QKV=1 SGLANG_TERNARY_I2S_SPLITS=4"
        "TERNARY_ENABLE_PDL=0 TERNARY_FUSE_RMSNORM_QKV=1"
    )
else
    PROFILE_NAMES=(
        "baseline"
        "decode_aggressive"
        "split2_prefill"
        "split4_prefill"
        "split8_prefill"
        "no_fused_rmsnorm_qkv"
        "no_pdl"
        "no_i2s_cutlass"
        "pinned_copy_off"
    )
    PROFILE_ENVS=(
        ""
        "TERNARY_ENABLE_PDL=1 TERNARY_FUSE_RMSNORM_QKV=1 SGLANG_TERNARY_I2S_SPLITS=1"
        "TERNARY_ENABLE_PDL=1 TERNARY_FUSE_RMSNORM_QKV=1 SGLANG_TERNARY_I2S_SPLITS=2"
        "TERNARY_ENABLE_PDL=1 TERNARY_FUSE_RMSNORM_QKV=1 SGLANG_TERNARY_I2S_SPLITS=4"
        "TERNARY_ENABLE_PDL=1 TERNARY_FUSE_RMSNORM_QKV=1 SGLANG_TERNARY_I2S_SPLITS=8"
        "TERNARY_ENABLE_PDL=1 TERNARY_FUSE_RMSNORM_QKV=0"
        "TERNARY_ENABLE_PDL=0 TERNARY_FUSE_RMSNORM_QKV=1"
        "SGLANG_TERNARY_USE_I2S_CUTLASS=0 TERNARY_ENABLE_PDL=1 TERNARY_FUSE_RMSNORM_QKV=1"
        "SGLANG_ENABLE_PINNED_OUTPUT_COPY=0 TERNARY_ENABLE_PDL=1 TERNARY_FUSE_RMSNORM_QKV=1"
    )
fi

if [[ "${#PROFILE_NAMES[@]}" -ne "${#PROFILE_ENVS[@]}" ]]; then
    echo "Internal error: profile arrays mismatch"
    exit 1
fi

RESULTS_CSV="${OUT_DIR}/kernel_sweep_summary.csv"
echo "profile,env,m1_p50_ms,m1_p95_ms,m1_tps,c1_tps,cmax_tps,c1_p50_ms,cmax_p50_ms,failed_requests" > "${RESULTS_CSV}"

SERVER_ARGS_QUOTED="$(build_server_args)"

run_idx=0
num_runs="${#PROFILE_NAMES[@]}"
for i in "${!PROFILE_NAMES[@]}"; do
    run_idx="$(( run_idx + 1 ))"
    profile="${PROFILE_NAMES[$i]}"
    env_line="${PROFILE_ENVS[$i]}"

    echo ""
    echo "=============================================================="
    echo "Profile ${run_idx}/${num_runs}: ${profile}"
    echo "Env: ${env_line:-<default>}"
    echo "=============================================================="

    kill_server_on_port "${SERVER_PORT}"

    server_log="${OUT_DIR}/${profile}_server.log"
    m1_json="${OUT_DIR}/${profile}_m1.json"
    conc_json="${OUT_DIR}/${profile}_conc.json"

    # shellcheck disable=SC2206
    env_assignments=(${env_line})
    cmd=(env "${env_assignments[@]}" bash -lc "cd \"${SGLANG_DIR}\" && ${SERVER_SCRIPT} ${SERVER_ARGS_QUOTED}")

    # Start server
    "${cmd[@]}" > "${server_log}" 2>&1 &
    server_pid="$!"
    echo "Server PID: ${server_pid}"

    if ! wait_for_server "${SERVER_HOST}" "${SERVER_PORT}" "${STARTUP_TIMEOUT}"; then
        echo "Server failed to start for ${profile}"
        kill -9 "${server_pid}" 2>/dev/null || true
        echo "${profile},\"${env_line}\",FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL" >> "${RESULTS_CSV}"
        continue
    fi

    # Run M=1 decode benchmark
    set +e
    "${PYTHON}" "${M1_BENCH}" \
        --host "${SERVER_HOST}" \
        --port "${SERVER_PORT}" \
        --mode "${profile}" \
        --steps "${M1_STEPS}" \
        --warmup "${M1_WARMUP}" \
        --runs "${M1_RUNS}" \
        --max-tokens "$(( M1_STEPS + M1_WARMUP + 48 ))" \
        --output "${m1_json}" \
        > "${OUT_DIR}/${profile}_m1.log" 2>&1
    m1_rc="$?"
    set -e
    if [[ "${m1_rc}" -ne 0 ]]; then
        echo "M1 benchmark failed for ${profile}; see ${OUT_DIR}/${profile}_m1.log"
    fi

    # Run concurrency benchmark
    set +e
    "${PYTHON}" "${CONC_BENCH}" \
        --host "${SERVER_HOST}" \
        --port "${SERVER_PORT}" \
        --concurrency-levels "${CONCURRENCY}" \
        --requests "${REQUESTS_PER_LEVEL}" \
        --max-tokens "${MAX_TOKENS}" \
        --warmup 3 \
        --label "${profile}" \
        --output "${conc_json}" \
        > "${OUT_DIR}/${profile}_conc.log" 2>&1
    conc_rc="$?"
    set -e
    if [[ "${conc_rc}" -ne 0 ]]; then
        echo "Concurrency benchmark failed for ${profile}; see ${OUT_DIR}/${profile}_conc.log"
    fi

    # Extract key metrics
    metrics="$("${PYTHON}" - <<PYEXTRACT
import json
import sys
from pathlib import Path

m1_path = Path(${m1_json@Q})
conc_path = Path(${conc_json@Q})
conc_levels = [int(x.strip()) for x in ${CONCURRENCY@Q}.split(",") if x.strip()]
cmax = max(conc_levels) if conc_levels else 1

def fail():
    print("ERR,ERR,ERR,ERR,ERR,ERR,ERR")

try:
    if not m1_path.exists() or not conc_path.exists():
        fail()
        raise SystemExit(0)

    with m1_path.open() as f:
        m1 = json.load(f)
    with conc_path.open() as f:
        conc = json.load(f)

    m1_p50 = float(m1.get("decode_time_p50", 0))
    m1_p95 = float(m1.get("decode_time_p95", 0))
    m1_tps = float(m1.get("tps_avg", 0))

    by_c = {int(r.get("concurrency", 0)): r for r in conc.get("results", [])}
    c1 = by_c.get(1, {})
    cmax_r = by_c.get(cmax, {})

    c1_tps = float(c1.get("tokens_per_sec_total", 0))
    cmax_tps = float(cmax_r.get("tokens_per_sec_total", 0))
    c1_p50 = float(c1.get("latency_p50", 0))
    cmax_p50 = float(cmax_r.get("latency_p50", 0))
    failed = sum(int(r.get("failed_requests", 0)) for r in conc.get("results", []))
    print(f"{m1_p50:.4f},{m1_p95:.4f},{m1_tps:.4f},{c1_tps:.4f},{cmax_tps:.4f},{c1_p50:.4f},{cmax_p50:.4f},{failed}")
except Exception:
    fail()
PYEXTRACT
)"
    echo "${profile},\"${env_line}\",${metrics}" >> "${RESULTS_CSV}"

    # Stop server
    kill -TERM "${server_pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${server_pid}" 2>/dev/null || true
    kill_server_on_port "${SERVER_PORT}"
    sleep "${COOLDOWN_SEC}"
done

echo ""
echo "=============================================================="
echo "Kernel Sweep Summary"
echo "=============================================================="
cat "${RESULTS_CSV}" | column -t -s','
echo ""

"${PYTHON}" - <<PYFINAL
import csv
from pathlib import Path

csv_path = Path(${RESULTS_CSV@Q})
rows = []
with csv_path.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            r["m1_p50_ms_f"] = float(r["m1_p50_ms"])
            r["cmax_tps_f"] = float(r["cmax_tps"])
            rows.append(r)
        except Exception:
            continue

if not rows:
    print("No successful rows to rank.")
else:
    best_decode = min(rows, key=lambda x: x["m1_p50_ms_f"])
    best_tp = max(rows, key=lambda x: x["cmax_tps_f"])
    print("Best decode latency profile:")
    print(f"  {best_decode['profile']}  m1_p50_ms={best_decode['m1_p50_ms']}  env={best_decode['env']}")
    print("Best high-concurrency throughput profile:")
    print(f"  {best_tp['profile']}  cmax_tps={best_tp['cmax_tps']}  env={best_tp['env']}")
    print(f"CSV: {csv_path}")
PYFINAL

echo "=============================================================="
echo "Done. Output: ${OUT_DIR}"
echo "=============================================================="
