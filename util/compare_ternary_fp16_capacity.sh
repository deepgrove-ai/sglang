#!/usr/bin/env bash
# =============================================================================
# Compare FP16 vs Ternary capacity:
# - Max concurrent users under TTFT target (default 1s p95)
# - KV cache budget per user
# - Throughput/latency across concurrency levels
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(dirname "$SCRIPT_DIR")"
SERVER_SCRIPT="${SGLANG_DIR}/run_server_ternary.sh"

if [[ -n "${SGLANG_PYTHON:-}" ]]; then
    PYTHON_BIN="${SGLANG_PYTHON}"
elif [[ -x "${SGLANG_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${SGLANG_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "Error: no Python interpreter found."
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-/scratch/mangrove}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-30080}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-360}"
COOLDOWN_SEC="${COOLDOWN_SEC:-8}"

CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1,2,4,8,16,24,32,40,48,64}"
REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TTFT_TARGET_MS="${TTFT_TARGET_MS:-1000}"
KV_SAFETY_FACTOR="${KV_SAFETY_FACTOR:-1.0}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${SGLANG_DIR}/capacity_compare_${TIMESTAMP}"
mkdir -p "${OUT_DIR}"

MODES=("fp16" "i2s-tuned")

echo "=============================================================="
echo "Compare FP16 vs Ternary Capacity"
echo "=============================================================="
echo "Model path:         ${MODEL_PATH}"
echo "Server:             http://${SERVER_HOST}:${SERVER_PORT}"
echo "Concurrency levels: ${CONCURRENCY_LEVELS}"
echo "Requests/level:     ${REQUESTS_PER_LEVEL}"
echo "TTFT target(ms):    ${TTFT_TARGET_MS}"
echo "Max new tokens:     ${MAX_NEW_TOKENS}"
echo "Output dir:         ${OUT_DIR}"
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

for mode in "${MODES[@]}"; do
    echo ""
    echo "=============================================================="
    echo "Mode: ${mode}"
    echo "=============================================================="
    kill_server_on_port "${SERVER_PORT}"

    mode_log="${OUT_DIR}/${mode}_server.log"
    ttft_json="${OUT_DIR}/${mode}_ttft_kv.json"
    conc_json="${OUT_DIR}/${mode}_concurrency.json"

    # Ternary tuned env knobs for best observed behavior on H100.
    if [[ "${mode}" == "i2s-tuned" ]]; then
        export TERNARY_ENABLE_PDL=1
        export TERNARY_FUSE_RMSNORM_QKV=1
        export SGLANG_TERNARY_I2S_SPLITS=1
    else
        unset TERNARY_ENABLE_PDL || true
        unset TERNARY_FUSE_RMSNORM_QKV || true
        unset SGLANG_TERNARY_I2S_SPLITS || true
    fi

    (
        cd "${SGLANG_DIR}"
        bash "${SERVER_SCRIPT}" "${mode}" --model-path "${MODEL_PATH}" --served-model-name "capacity-${mode}"
    ) > "${mode_log}" 2>&1 &
    server_pid="$!"
    echo "Server PID: ${server_pid}"

    if ! wait_for_server "${SERVER_HOST}" "${SERVER_PORT}" "${STARTUP_TIMEOUT}"; then
        echo "Failed to start mode ${mode}; check ${mode_log}"
        kill -9 "${server_pid}" 2>/dev/null || true
        continue
    fi

    "${PYTHON_BIN}" "${SCRIPT_DIR}/ttft_kv_capacity.py" \
        --host "${SERVER_HOST}" \
        --port "${SERVER_PORT}" \
        --concurrency-levels "${CONCURRENCY_LEVELS}" \
        --requests-per-level "${REQUESTS_PER_LEVEL}" \
        --ttft-target-ms "${TTFT_TARGET_MS}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --kv-safety-factor "${KV_SAFETY_FACTOR}" \
        --label "${mode}" \
        --output "${ttft_json}" \
        > "${OUT_DIR}/${mode}_ttft_kv.log" 2>&1

    "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_concurrency.py" \
        --host "${SERVER_HOST}" \
        --port "${SERVER_PORT}" \
        --concurrency-levels "${CONCURRENCY_LEVELS}" \
        --requests "${REQUESTS_PER_LEVEL}" \
        --max-tokens "${MAX_NEW_TOKENS}" \
        --warmup 3 \
        --label "${mode}" \
        --output "${conc_json}" \
        > "${OUT_DIR}/${mode}_concurrency.log" 2>&1

    kill -TERM "${server_pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${server_pid}" 2>/dev/null || true
    kill_server_on_port "${SERVER_PORT}"
    sleep "${COOLDOWN_SEC}"
done

# Summarize
"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

out_dir = Path(${OUT_DIR@Q})
modes = ["fp16", "i2s-tuned"]

print("\\n" + "=" * 120)
print("FP16 vs Ternary Capacity Summary")
print("=" * 120)
print(f"{'Mode':<12} {'MaxUsers@TTFT':>14} {'C16 TPS':>10} {'C32 TPS':>10} {'C64 TPS':>10} {'C16 p50(ms)':>12} {'C32 p50(ms)':>12}")
print("-" * 120)

def by_c(results):
    return {int(r["concurrency"]): r for r in results}

summary = {}
for mode in modes:
    ttft_path = out_dir / f"{mode}_ttft_kv.json"
    conc_path = out_dir / f"{mode}_concurrency.json"
    if not ttft_path.exists() or not conc_path.exists():
        print(f"{mode:<12} {'MISSING':>14}")
        continue
    ttft = json.loads(ttft_path.read_text())
    conc = json.loads(conc_path.read_text())
    c = by_c(conc.get("results", []))
    max_users = ttft.get("max_users_under_target", 0)
    row = {
        "max_users": max_users,
        "c16_tps": c.get(16, {}).get("tokens_per_sec_total", 0.0),
        "c32_tps": c.get(32, {}).get("tokens_per_sec_total", 0.0),
        "c64_tps": c.get(64, {}).get("tokens_per_sec_total", 0.0),
        "c16_p50": c.get(16, {}).get("latency_p50", 0.0),
        "c32_p50": c.get(32, {}).get("latency_p50", 0.0),
    }
    summary[mode] = row
    print(
        f"{mode:<12} {row['max_users']:>14} "
        f"{row['c16_tps']:>10.1f} {row['c32_tps']:>10.1f} {row['c64_tps']:>10.1f} "
        f"{row['c16_p50']:>12.1f} {row['c32_p50']:>12.1f}"
    )

print("-" * 120)
if "fp16" in summary and "i2s-tuned" in summary:
    fp16 = summary["fp16"]
    ter = summary["i2s-tuned"]
    def pct(new, old):
        return ((new - old) / old * 100.0) if old else 0.0
    print("Delta (ternary vs fp16):")
    print(f"  max_users@TTFT: {ter['max_users'] - fp16['max_users']} users")
    print(f"  C16 TPS: {pct(ter['c16_tps'], fp16['c16_tps']):+.2f}%")
    print(f"  C32 TPS: {pct(ter['c32_tps'], fp16['c32_tps']):+.2f}%")
    if fp16['c64_tps'] and ter['c64_tps']:
        print(f"  C64 TPS: {pct(ter['c64_tps'], fp16['c64_tps']):+.2f}%")

print(f"\\nDetailed artifacts: {out_dir}")
print("=" * 120)
PY

