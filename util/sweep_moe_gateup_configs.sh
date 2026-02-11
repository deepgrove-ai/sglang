#!/usr/bin/env bash
set -euo pipefail

# Sweep a small set of gate_up Triton configs and compare:
# - end-to-end throughput (benchmark_concurrency.py)
# - MoE stage profile split (summarize_moe_batched_profile.py)
#
# Usage:
#   bash util/sweep_moe_gateup_configs.sh --model-path /scratch/mangrove

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SGLANG_ROOT}"

MODEL_PATH="${MODEL_PATH:-/scratch/mangrove}"
MODE="${MODE:-i2s-tuned}"
PORT="${PORT:-31080}"
REQUESTS="${REQUESTS:-8}"
WARMUP="${WARMUP:-1}"
MAX_TOKENS="${MAX_TOKENS:-32}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-4}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-420}"
OUTPUT_DIR="${OUTPUT_DIR:-${SGLANG_ROOT}/moe_gateup_sweep_$(date +%Y%m%d_%H%M%S)}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"; shift 2 ;;
        --mode)
            MODE="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --requests)
            REQUESTS="$2"; shift 2 ;;
        --warmup)
            WARMUP="$2"; shift 2 ;;
        --max-tokens)
            MAX_TOKENS="$2"; shift 2 ;;
        --concurrency-levels)
            CONCURRENCY_LEVELS="$2"; shift 2 ;;
        --timeout-sec)
            RUN_TIMEOUT_SEC="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

if [[ ! -x "${SGLANG_ROOT}/.venv/bin/python" ]]; then
    echo "Missing ${SGLANG_ROOT}/.venv/bin/python" >&2
    exit 1
fi

kill_servers() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "run_server_ternary.sh ${MODE}" 2>/dev/null || true
    sleep 2
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "run_server_ternary.sh ${MODE}" 2>/dev/null || true
}

wait_health() {
    local retries=120
    for ((i=0; i<retries; i++)); do
        code="$(curl -sS -m 2 -o /tmp/moe_gateup_health.out -w '%{http_code}' "http://127.0.0.1:${PORT}/health" || true)"
        if [[ "${code}" == "200" ]]; then
            return 0
        fi
        sleep 2
    done
    return 1
}

run_one() {
    local name="$1"
    local gateup_overrides="$2"
    local run_dir="${OUTPUT_DIR}/${name}"
    mkdir -p "${run_dir}"

    echo "==> Running ${name}"
    kill_servers

    # shellcheck disable=SC2086
    eval \
        SGLANG_TERNARY_MOE_BATCHED_PROFILE=1 \
        SGLANG_TERNARY_ENABLE_EXPERIMENTAL_MGT1_LINEAR=1 \
        SGLANG_TERNARY_MGT1_VALIDATE=0 \
        SGLANG_TERNARY_MOE_COMBINE_TRITON=1 \
        ${gateup_overrides} \
        "bash run_server_ternary.sh ${MODE} --model-path ${MODEL_PATH} --served-model-name mangrove-i2s-tuned --port ${PORT} > \"${run_dir}/server.log\" 2>&1 &"

    if ! wait_health; then
        echo "Server failed health check for ${name}" | tee "${run_dir}/error.txt"
        return 1
    fi

    timeout "${RUN_TIMEOUT_SEC}" "${SGLANG_ROOT}/.venv/bin/python" util/benchmark_concurrency.py \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --concurrency-levels "${CONCURRENCY_LEVELS}" \
        --requests "${REQUESTS}" \
        --warmup "${WARMUP}" \
        --max-tokens "${MAX_TOKENS}" \
        --label "moe-gateup-${name}" \
        --output "${run_dir}/concurrency.json" \
        > "${run_dir}/concurrency.log" 2>&1 || true

    "${SGLANG_ROOT}/.venv/bin/python" util/summarize_moe_batched_profile.py \
        --log "${run_dir}/server.log" \
        --output-dir "${run_dir}" \
        > "${run_dir}/profile_summary.log" 2>&1 || true

    "${SGLANG_ROOT}/.venv/bin/python" - <<'PY' "${run_dir}" "${name}" "${OUTPUT_DIR}/results.csv"
import csv
import json
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
name = sys.argv[2]
csv_path = Path(sys.argv[3])

tps = 0.0
lat_p50 = 0.0
conc = ""
conc_path = run_dir / "concurrency.json"
if conc_path.exists():
    payload = json.loads(conc_path.read_text())
    results = payload.get("results", [])
    if results:
        # Use highest concurrency entry from the run.
        best = max(results, key=lambda r: int(r.get("concurrency", 0)))
        conc = str(best.get("concurrency", ""))
        tps = float(best.get("tokens_per_sec_total", 0.0))
        lat_p50 = float(best.get("latency_p50", 0.0))

gateup = 0.0
down = 0.0
combine = 0.0
total = 0.0
sum_path = run_dir / "moe_batched_profile_summary.json"
if sum_path.exists():
    sp = json.loads(sum_path.read_text())
    gateup = float(sp.get("gateup_ms", {}).get("mean", 0.0))
    down = float(sp.get("down_ms", {}).get("mean", 0.0))
    combine = float(sp.get("combine_ms", {}).get("mean", 0.0))
    total = float(sp.get("total_ms", {}).get("mean", 0.0))

header = [
    "name",
    "concurrency",
    "tokens_per_sec_total",
    "latency_p50_ms",
    "gateup_ms_mean",
    "down_ms_mean",
    "combine_ms_mean",
    "total_ms_mean",
    "run_dir",
]
exists = csv_path.exists()
with csv_path.open("a", newline="") as f:
    w = csv.writer(f)
    if not exists:
        w.writerow(header)
    w.writerow([name, conc, f"{tps:.3f}", f"{lat_p50:.3f}", f"{gateup:.6f}", f"{down:.6f}", f"{combine:.6f}", f"{total:.6f}", str(run_dir)])
PY

    kill_servers
}

# Candidate set:
# - baseline: current defaults
# - gateup_g64: same tile but larger GROUP_SIZE_M
# - gateup_w4s3: reduce warps/stages
# - gateup_n64_w4s3: narrower N tile + lower warps/stages
run_one "baseline" ""
run_one "gateup_g64" \
    "SGLANG_TERNARY_MOE_GATEUP_BLOCK_SIZE_M=16 SGLANG_TERNARY_MOE_GATEUP_BLOCK_SIZE_N=128 SGLANG_TERNARY_MOE_GATEUP_BLOCK_SIZE_K=256 SGLANG_TERNARY_MOE_GATEUP_GROUP_SIZE_M=64 SGLANG_TERNARY_MOE_GATEUP_NUM_WARPS=8 SGLANG_TERNARY_MOE_GATEUP_NUM_STAGES=5"
run_one "gateup_w4s3" \
    "SGLANG_TERNARY_MOE_GATEUP_BLOCK_SIZE_M=16 SGLANG_TERNARY_MOE_GATEUP_BLOCK_SIZE_N=128 SGLANG_TERNARY_MOE_GATEUP_BLOCK_SIZE_K=256 SGLANG_TERNARY_MOE_GATEUP_GROUP_SIZE_M=1 SGLANG_TERNARY_MOE_GATEUP_NUM_WARPS=4 SGLANG_TERNARY_MOE_GATEUP_NUM_STAGES=3"
run_one "gateup_n64_w4s3" \
    "SGLANG_TERNARY_MOE_GATEUP_BLOCK_SIZE_M=16 SGLANG_TERNARY_MOE_GATEUP_BLOCK_SIZE_N=64 SGLANG_TERNARY_MOE_GATEUP_BLOCK_SIZE_K=256 SGLANG_TERNARY_MOE_GATEUP_GROUP_SIZE_M=64 SGLANG_TERNARY_MOE_GATEUP_NUM_WARPS=4 SGLANG_TERNARY_MOE_GATEUP_NUM_STAGES=3"

"${SGLANG_ROOT}/.venv/bin/python" - <<'PY' "${OUTPUT_DIR}/results.csv" "${OUTPUT_DIR}/summary.md"
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open()))
rows_sorted = sorted(rows, key=lambda r: float(r["tokens_per_sec_total"]), reverse=True)

lines = [
    "# MoE GateUp Sweep Summary",
    "",
    f"- Source CSV: `{csv_path}`",
    "",
    "| name | C | total TPS | p50 ms | gateup ms | down ms | combine ms | total ms |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows_sorted:
    lines.append(
        f"| {r['name']} | {r['concurrency']} | {float(r['tokens_per_sec_total']):.1f} | "
        f"{float(r['latency_p50_ms']):.1f} | {float(r['gateup_ms_mean']):.4f} | "
        f"{float(r['down_ms_mean']):.4f} | {float(r['combine_ms_mean']):.4f} | {float(r['total_ms_mean']):.4f} |"
    )
md_path.write_text("\n".join(lines) + "\n")
print(md_path)
PY

echo "Done. Results:"
echo "  ${OUTPUT_DIR}/results.csv"
echo "  ${OUTPUT_DIR}/summary.md"

