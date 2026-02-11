#!/usr/bin/env bash
# =============================================================================
# Freeze reproducible ternary baseline matrix (Phase 1 helper)
# =============================================================================
#
# This script creates a single artifact bundle containing:
#   1) Corpus-backed support-envelope worksheet
#   2) Baseline ternary server logs + runtime path/fallback counters
#   3) M=1 decode, TTFT/KV-capacity, and concurrency benchmark outputs
#   4) Optional long-run artifacts from sweep_ternary_kernels.sh and
#      compare_ternary_fp16_capacity.sh
#
# Usage:
#   bash util/freeze_ternary_baseline_matrix.sh
#   bash util/freeze_ternary_baseline_matrix.sh --quick
#   bash util/freeze_ternary_baseline_matrix.sh --dry-run
#   bash util/freeze_ternary_baseline_matrix.sh \
#       --model-path /scratch/mangrove \
#       --mode i2s-tuned \
#       --run-kernel-sweep 0 --run-capacity-compare 0
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(dirname "$SCRIPT_DIR")"
RUN_SERVER_SCRIPT="${SGLANG_DIR}/run_server_ternary.sh"
SUPPORT_MATRIX_SCRIPT="${SCRIPT_DIR}/build_support_envelope_matrix.py"

if [[ -n "${SGLANG_PYTHON:-}" ]]; then
    PYTHON_BIN="${SGLANG_PYTHON}"
elif [[ -x "${SGLANG_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${SGLANG_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "Error: no Python interpreter found (set SGLANG_PYTHON)."
    exit 1
fi

MODE="${MODE:-i2s-tuned}"
MODEL_PATH="${MODEL_PATH:-/scratch/mangrove}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-baseline-i2s-tuned}"
TP_SIZE="${TP_SIZE:-1}"

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-32080}"
SWEEP_PORT="${SWEEP_PORT:-32081}"
COMPARE_PORT="${COMPARE_PORT:-32082}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-360}"
COOLDOWN_SEC="${COOLDOWN_SEC:-6}"

CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1,2,4,8,16,24,32,40,48,64}"
REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TTFT_TARGET_MS="${TTFT_TARGET_MS:-1000}"
KV_SAFETY_FACTOR="${KV_SAFETY_FACTOR:-1.0}"

M1_STEPS="${M1_STEPS:-120}"
M1_RUNS="${M1_RUNS:-3}"
M1_WARMUP="${M1_WARMUP:-12}"

RUN_KERNEL_SWEEP="${RUN_KERNEL_SWEEP:-0}"
RUN_CAPACITY_COMPARE="${RUN_CAPACITY_COMPARE:-0}"
QUICK="${QUICK:-0}"
DRY_RUN=0
REQUIRE_TERNARY_KERNELS="${REQUIRE_TERNARY_KERNELS:-1}"

CUDA_CORPUS_ROOT="${CUDA_CORPUS_ROOT:-/home/ubuntu/.cache/wafer/corpora/cuda}"
CUTLASS_CORPUS_ROOT="${CUTLASS_CORPUS_ROOT:-/home/ubuntu/.cache/wafer/corpora/cutlass}"

OUT_DIR="${OUT_DIR:-}"
SERVER_PID=""

usage() {
    cat <<'EOF'
Usage: freeze_ternary_baseline_matrix.sh [options]

Options:
  --mode MODE                   Ternary mode (default: i2s-tuned)
  --model-path PATH             Model path (default: /scratch/mangrove)
  --served-model-name NAME      Served model name (default: baseline-i2s-tuned)
  --tp N                        Tensor parallel size for baseline server (default: 1)

  --host HOST                   Server host (default: 127.0.0.1)
  --port PORT                   Baseline server port (default: 32080)
  --sweep-port PORT             Optional sweep server port (default: 32081)
  --compare-port PORT           Optional compare server port (default: 32082)
  --startup-timeout SEC         Server health timeout (default: 360)
  --cooldown-sec SEC            Cooldown between phases (default: 6)

  --concurrency-levels CSV      Concurrency levels (default: 1,2,4,8,16,24,32,40,48,64)
  --requests-per-level N        Requests per concurrency level (default: 20)
  --max-new-tokens N            Max new tokens per request (default: 128)
  --ttft-target-ms MS           TTFT p95 target (default: 1000)
  --kv-safety-factor F          KV safety factor (default: 1.0)

  --m1-steps N                  M=1 decode measured steps (default: 120)
  --m1-runs N                   M=1 decode runs (default: 3)
  --m1-warmup N                 M=1 decode warmup steps (default: 12)

  --run-kernel-sweep 0|1        Run util/sweep_ternary_kernels.sh (default: 0)
  --run-capacity-compare 0|1    Run util/compare_ternary_fp16_capacity.sh (default: 0)
  --require-ternary-kernels 0|1 Require loaded ternary CUDA kernels for i2s/ternary modes (default: 1)
  --quick                       Reduce baseline matrix size for faster iteration
  --dry-run                     Print plan and exit without launching workloads

  --cuda-corpus-root PATH       Wafer CUDA corpus root
  --cutlass-corpus-root PATH    Wafer CUTLASS corpus root
  --out-dir PATH                Output directory (default: sglang/baseline_matrix_<timestamp>)
  -h, --help                    Show this help
EOF
}

normalize_bool() {
    local raw="${1:-}"
    local lower
    lower="$(echo "${raw}" | tr '[:upper:]' '[:lower:]')"
    case "${lower}" in
        1|true|yes|on) echo "1" ;;
        0|false|no|off) echo "0" ;;
        *)
            echo "Error: expected boolean 0|1|true|false but got '${raw}'" >&2
            exit 1
            ;;
    esac
}

find_bitnet_lib() {
    local candidates=(
        "${SGLANG_DIR}/libternary_bitnet.so"
        "${SGLANG_DIR}/third_party/ternarykernels/mangrove-turbo/libternary_bitnet.so"
        "${SGLANG_DIR}/ternarykernels/mangrove-turbo/libternary_bitnet.so"
        "${SGLANG_DIR}/../ternarykernels/mangrove-turbo/libternary_bitnet.so"
        "/home/ubuntu/ternarykernels/mangrove-turbo/libternary_bitnet.so"
        "/usr/local/lib/libternary_bitnet.so"
    )
    local path
    for path in "${candidates[@]}"; do
        if [[ -f "${path}" ]]; then
            echo "${path}"
            return 0
        fi
    done
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:-}"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="${2:-}"
            shift 2
            ;;
        --served-model-name)
            SERVED_MODEL_NAME="${2:-}"
            shift 2
            ;;
        --tp)
            TP_SIZE="${2:-}"
            shift 2
            ;;
        --host)
            SERVER_HOST="${2:-}"
            shift 2
            ;;
        --port)
            SERVER_PORT="${2:-}"
            shift 2
            ;;
        --sweep-port)
            SWEEP_PORT="${2:-}"
            shift 2
            ;;
        --compare-port)
            COMPARE_PORT="${2:-}"
            shift 2
            ;;
        --startup-timeout)
            STARTUP_TIMEOUT="${2:-}"
            shift 2
            ;;
        --cooldown-sec)
            COOLDOWN_SEC="${2:-}"
            shift 2
            ;;
        --concurrency-levels)
            CONCURRENCY_LEVELS="${2:-}"
            shift 2
            ;;
        --requests-per-level)
            REQUESTS_PER_LEVEL="${2:-}"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="${2:-}"
            shift 2
            ;;
        --ttft-target-ms)
            TTFT_TARGET_MS="${2:-}"
            shift 2
            ;;
        --kv-safety-factor)
            KV_SAFETY_FACTOR="${2:-}"
            shift 2
            ;;
        --m1-steps)
            M1_STEPS="${2:-}"
            shift 2
            ;;
        --m1-runs)
            M1_RUNS="${2:-}"
            shift 2
            ;;
        --m1-warmup)
            M1_WARMUP="${2:-}"
            shift 2
            ;;
        --run-kernel-sweep)
            RUN_KERNEL_SWEEP="$(normalize_bool "${2:-}")"
            shift 2
            ;;
        --run-capacity-compare)
            RUN_CAPACITY_COMPARE="$(normalize_bool "${2:-}")"
            shift 2
            ;;
        --quick)
            QUICK=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --require-ternary-kernels)
            REQUIRE_TERNARY_KERNELS="$(normalize_bool "${2:-}")"
            shift 2
            ;;
        --cuda-corpus-root)
            CUDA_CORPUS_ROOT="${2:-}"
            shift 2
            ;;
        --cutlass-corpus-root)
            CUTLASS_CORPUS_ROOT="${2:-}"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1'"
            usage
            exit 1
            ;;
    esac
done

if [[ "${QUICK}" == "1" ]]; then
    CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1,2,4,8,16}"
    REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL:-8}"
    M1_STEPS="${M1_STEPS:-48}"
    M1_RUNS="${M1_RUNS:-2}"
    M1_WARMUP="${M1_WARMUP:-8}"
fi

if [[ -z "${OUT_DIR}" ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    OUT_DIR="${SGLANG_DIR}/baseline_matrix_${TS}"
fi
mkdir -p "${OUT_DIR}"

COMMAND_LOG="${OUT_DIR}/commands.log"
CONFIG_SNAPSHOT="${OUT_DIR}/config.env"

log_cmd() {
    printf '[%s] ' "$(date --iso-8601=seconds)" >> "${COMMAND_LOG}"
    printf '%q ' "$@" >> "${COMMAND_LOG}"
    printf '\n' >> "${COMMAND_LOG}"
}

wait_for_server() {
    local host="$1"
    local port="$2"
    local timeout="$3"
    local start
    start="$(date +%s)"
    echo -n "Waiting for server on ${host}:${port}"
    while true; do
        if curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1; then
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
        echo "Killing process(es) on :${port}: ${pids}"
        echo "${pids}" | xargs -r kill -9 2>/dev/null || true
        sleep 2
    fi
}

cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        sleep 2
        kill -9 "${SERVER_PID}" 2>/dev/null || true
    fi
    kill_server_on_port "${SERVER_PORT}"
}
trap cleanup EXIT

cat > "${CONFIG_SNAPSHOT}" <<EOF
MODE=${MODE}
MODEL_PATH=${MODEL_PATH}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
TP_SIZE=${TP_SIZE}
REQUIRE_TERNARY_KERNELS=${REQUIRE_TERNARY_KERNELS}
SERVER_HOST=${SERVER_HOST}
SERVER_PORT=${SERVER_PORT}
SWEEP_PORT=${SWEEP_PORT}
COMPARE_PORT=${COMPARE_PORT}
CONCURRENCY_LEVELS=${CONCURRENCY_LEVELS}
REQUESTS_PER_LEVEL=${REQUESTS_PER_LEVEL}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
TTFT_TARGET_MS=${TTFT_TARGET_MS}
KV_SAFETY_FACTOR=${KV_SAFETY_FACTOR}
M1_STEPS=${M1_STEPS}
M1_RUNS=${M1_RUNS}
M1_WARMUP=${M1_WARMUP}
RUN_KERNEL_SWEEP=${RUN_KERNEL_SWEEP}
RUN_CAPACITY_COMPARE=${RUN_CAPACITY_COMPARE}
QUICK=${QUICK}
CUDA_CORPUS_ROOT=${CUDA_CORPUS_ROOT}
CUTLASS_CORPUS_ROOT=${CUTLASS_CORPUS_ROOT}
PYTHON_BIN=${PYTHON_BIN}
EOF

SUPPORT_CSV="${OUT_DIR}/support_envelope_matrix.csv"
SUPPORT_MD="${OUT_DIR}/support_envelope_matrix.md"
BASELINE_SERVER_LOG="${OUT_DIR}/server_${MODE}.log"
M1_JSON="${OUT_DIR}/baseline_m1_decode.json"
TTFT_JSON="${OUT_DIR}/baseline_ttft_kv.json"
CONC_JSON="${OUT_DIR}/baseline_concurrency.json"
FAMILY_COUNTERS_JSON="${OUT_DIR}/baseline_family_counters.json"
SUMMARY_JSON="${OUT_DIR}/baseline_summary.json"
SUMMARY_MD="${OUT_DIR}/baseline_summary.md"
RUNTIME_REPORT_TEMPLATE="${OUT_DIR}/ternary_runtime_{pid}.json"

echo "=============================================================="
echo "Freeze Ternary Baseline Matrix"
echo "=============================================================="
echo "Output dir:            ${OUT_DIR}"
echo "Mode:                  ${MODE}"
echo "Model path:            ${MODEL_PATH}"
echo "Baseline endpoint:     http://${SERVER_HOST}:${SERVER_PORT}"
echo "Sweep enabled:         ${RUN_KERNEL_SWEEP} (port ${SWEEP_PORT})"
echo "Capacity compare:      ${RUN_CAPACITY_COMPARE} (port ${COMPARE_PORT})"
echo "Require ternary libs:  ${REQUIRE_TERNARY_KERNELS}"
echo "Corpus roots:"
echo "  CUDA:                ${CUDA_CORPUS_ROOT}"
echo "  CUTLASS:             ${CUTLASS_CORPUS_ROOT}"
echo "=============================================================="

if [[ "${DRY_RUN}" == "0" ]]; then
    if [[ ! -d "${MODEL_PATH}" ]]; then
        echo "Error: model path does not exist: ${MODEL_PATH}"
        exit 1
    fi

    if [[ "${MODE}" == i2s* || "${MODE}" == "ternary" ]]; then
        BITNET_LIB_PATH="$(find_bitnet_lib || true)"
        if [[ -z "${BITNET_LIB_PATH}" && "${REQUIRE_TERNARY_KERNELS}" == "1" ]]; then
            echo "Error: ternary CUDA kernel library libternary_bitnet.so not found."
            echo "This would force FP16 fallback and produce misleading M=1 tok/s."
            echo "Fix options:"
            echo "  1) Place ternarykernels source at /home/ubuntu/ternarykernels (with mangrove-turbo/)."
            echo "  2) Run: bash util/setup_mangrove_monorepo.sh --source /home/ubuntu/ternarykernels"
            echo "  3) Run: bash util/build_sync_ternary_kernels.sh"
            echo "Or bypass this guard (not recommended): --require-ternary-kernels 0"
            exit 1
        fi
        if [[ -n "${BITNET_LIB_PATH}" ]]; then
            echo "Detected ternary CUDA kernel library: ${BITNET_LIB_PATH}"
        fi
    fi
fi

SUPPORT_CMD=(
    "${PYTHON_BIN}" "${SUPPORT_MATRIX_SCRIPT}"
    --cuda-corpus-root "${CUDA_CORPUS_ROOT}"
    --cutlass-corpus-root "${CUTLASS_CORPUS_ROOT}"
    --output-csv "${SUPPORT_CSV}"
    --output-md "${SUPPORT_MD}"
)

BASELINE_SERVER_CMD=(
    env
    SGLANG_TERNARY_COLLECT_PATH_STATS=1
    SGLANG_TERNARY_FALLBACK_REPORT="${RUNTIME_REPORT_TEMPLATE}"
    SGLANG_TERNARY_FALLBACK_REPORT_EVERY=500
    bash "${RUN_SERVER_SCRIPT}" "${MODE}"
    --model-path "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --port "${SERVER_PORT}"
    --tp "${TP_SIZE}"
)

M1_CMD=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_m1_decode.py"
    --host "${SERVER_HOST}"
    --port "${SERVER_PORT}"
    --mode "${MODE}"
    --steps "${M1_STEPS}"
    --warmup "${M1_WARMUP}"
    --runs "${M1_RUNS}"
    --max-tokens "$(( M1_STEPS + M1_WARMUP + 48 ))"
    --output "${M1_JSON}"
)

TTFT_CMD=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/ttft_kv_capacity.py"
    --host "${SERVER_HOST}"
    --port "${SERVER_PORT}"
    --concurrency-levels "${CONCURRENCY_LEVELS}"
    --requests-per-level "${REQUESTS_PER_LEVEL}"
    --ttft-target-ms "${TTFT_TARGET_MS}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --kv-safety-factor "${KV_SAFETY_FACTOR}"
    --label "${MODE}-baseline"
    --output "${TTFT_JSON}"
)

CONC_CMD=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_concurrency.py"
    --host "${SERVER_HOST}"
    --port "${SERVER_PORT}"
    --concurrency-levels "${CONCURRENCY_LEVELS}"
    --requests "${REQUESTS_PER_LEVEL}"
    --max-tokens "${MAX_NEW_TOKENS}"
    --warmup 3
    --streaming
    --label "${MODE}-baseline"
    --output "${CONC_JSON}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    echo ""
    echo "Dry-run command plan:"
    printf '  '; printf '%q ' "${SUPPORT_CMD[@]}"; echo
    echo "  (cd ${SGLANG_DIR} && <baseline_server_cmd>)  # background"
    printf '    '; printf '%q ' "${BASELINE_SERVER_CMD[@]}"; echo
    printf '  '; printf '%q ' "${M1_CMD[@]}"; echo
    printf '  '; printf '%q ' "${TTFT_CMD[@]}"; echo
    printf '  '; printf '%q ' "${CONC_CMD[@]}"; echo
    if [[ "${RUN_KERNEL_SWEEP}" == "1" ]]; then
        echo "  (cd ${SGLANG_DIR} && env MODEL_PATH=${MODEL_PATH} SERVER_PORT=${SWEEP_PORT} ... bash util/sweep_ternary_kernels.sh ${MODE})"
    fi
    if [[ "${RUN_CAPACITY_COMPARE}" == "1" ]]; then
        echo "  (cd ${SGLANG_DIR} && env MODEL_PATH=${MODEL_PATH} SERVER_PORT=${COMPARE_PORT} ... bash util/compare_ternary_fp16_capacity.sh)"
    fi
    echo ""
    echo "Dry-run complete. Output dir reserved at: ${OUT_DIR}"
    exit 0
fi

echo "[1/7] Building corpus-backed support-envelope worksheet..."
log_cmd "${SUPPORT_CMD[@]}"
"${SUPPORT_CMD[@]}" > "${OUT_DIR}/support_matrix.log" 2>&1

echo "[2/7] Launching baseline server with ternary runtime stats enabled..."
kill_server_on_port "${SERVER_PORT}"
log_cmd "${BASELINE_SERVER_CMD[@]}"
(
    cd "${SGLANG_DIR}"
    "${BASELINE_SERVER_CMD[@]}"
) > "${BASELINE_SERVER_LOG}" 2>&1 &
SERVER_PID="$!"
echo "Baseline server PID: ${SERVER_PID}"

if ! wait_for_server "${SERVER_HOST}" "${SERVER_PORT}" "${STARTUP_TIMEOUT}"; then
    echo "Error: baseline server failed to become healthy."
    echo "See log: ${BASELINE_SERVER_LOG}"
    exit 1
fi

if [[ "${MODE}" == i2s* || "${MODE}" == "ternary" ]] && [[ "${REQUIRE_TERNARY_KERNELS}" == "1" ]]; then
    if grep -q "CUDA kernels not found - performance will be degraded" "${BASELINE_SERVER_LOG}"; then
        echo "Error: server log indicates ternary CUDA kernels failed to load."
        echo "Benchmarks would not represent ternary-kernel performance."
        echo "See: ${BASELINE_SERVER_LOG}"
        exit 1
    fi
fi

echo "[3/7] Running M=1 decode benchmark..."
log_cmd "${M1_CMD[@]}"
"${M1_CMD[@]}" > "${OUT_DIR}/baseline_m1.log" 2>&1

echo "[4/7] Running TTFT + KV capacity benchmark..."
log_cmd "${TTFT_CMD[@]}"
"${TTFT_CMD[@]}" > "${OUT_DIR}/baseline_ttft_kv.log" 2>&1

echo "[5/7] Running concurrency benchmark..."
log_cmd "${CONC_CMD[@]}"
"${CONC_CMD[@]}" > "${OUT_DIR}/baseline_concurrency.log" 2>&1

echo "[6/7] Stopping baseline server and summarizing runtime fallback/path counters..."
kill -TERM "${SERVER_PID}" 2>/dev/null || true
sleep 3
kill -9 "${SERVER_PID}" 2>/dev/null || true
SERVER_PID=""
kill_server_on_port "${SERVER_PORT}"
sleep "${COOLDOWN_SEC}"

log_cmd "${PYTHON_BIN}" "-" "${OUT_DIR}" "${FAMILY_COUNTERS_JSON}"
"${PYTHON_BIN}" - "${OUT_DIR}" "${FAMILY_COUNTERS_JSON}" <<'PYCOUNTERS'
import json
import sys
from collections import Counter
from pathlib import Path

out_dir = Path(sys.argv[1])
out_file = Path(sys.argv[2])

runtime_files = sorted(out_dir.glob("ternary_runtime_*.json"))
path_totals = Counter()
fallback_totals = Counter()

for path in runtime_files:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    for k, v in payload.get("path_counts", {}).items():
        path_totals[k] += int(v)
    for k, v in payload.get("fallback_counts", {}).items():
        fallback_totals[k] += int(v)

def path_bucket(key: str) -> str:
    if key.startswith("linear."):
        if "_m1" in key:
            return "linear.m1"
        if "m_gt_1" in key or "batch_megafused" in key or "i2s_cutlass" in key:
            return "linear.m>1"
        if "fp16_fallback" in key:
            return "linear.fp16_fallback"
        return "linear.other"
    if key.startswith("moe.decode_"):
        return "moe.decode"
    if key.startswith("moe.batched_"):
        return "moe.batched"
    if key.startswith("moe.fp16_fallback"):
        return "moe.fp16_fallback"
    if key.startswith("moe."):
        return "moe.other"
    return "other"

def fallback_bucket(key: str) -> str:
    if key.startswith("linear."):
        if "batch_megafused" in key or "m_gt_1" in key or "i2s_cutlass" in key:
            return "linear.m>1"
        if "_m1" in key:
            return "linear.m1"
        if "fp16_fallback" in key:
            return "linear.fp16_fallback"
        return "linear.other"
    if key.startswith("moe.decode_"):
        return "moe.decode"
    if key.startswith("moe.batched_"):
        return "moe.batched"
    if key.startswith("moe.fp16_fallback") or "missing_topk" in key:
        return "moe.fp16_fallback"
    if key.startswith("moe."):
        return "moe.other"
    return "other"

path_family = Counter()
fallback_family = Counter()
for key, count in path_totals.items():
    path_family[path_bucket(key)] += count
for key, count in fallback_totals.items():
    fallback_family[fallback_bucket(key)] += count

payload = {
    "runtime_report_files": [str(p) for p in runtime_files],
    "path_family_counts": dict(sorted(path_family.items())),
    "fallback_family_counts": dict(sorted(fallback_family.items())),
    "path_counts_raw": dict(sorted(path_totals.items())),
    "fallback_counts_raw": dict(sorted(fallback_totals.items())),
}
out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Wrote: {out_file}")
PYCOUNTERS

SWEEP_OUT_DIR=""
COMPARE_OUT_DIR=""

if [[ "${RUN_KERNEL_SWEEP}" == "1" ]]; then
    echo "[7a/7] Running optional kernel sweep..."
    sweep_start="$(date +%s)"
    SWEEP_LOG="${OUT_DIR}/optional_sweep.log"
    log_cmd env MODEL_PATH="${MODEL_PATH}" SERVER_HOST="${SERVER_HOST}" SERVER_PORT="${SWEEP_PORT}" CONCURRENCY="${CONCURRENCY_LEVELS}" REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL}" MAX_TOKENS="${MAX_NEW_TOKENS}" M1_STEPS="${M1_STEPS}" M1_RUNS="${M1_RUNS}" M1_WARMUP="${M1_WARMUP}" QUICK="${QUICK}" bash "${SCRIPT_DIR}/sweep_ternary_kernels.sh" "${MODE}"
    if (
        cd "${SGLANG_DIR}"
        env \
            MODEL_PATH="${MODEL_PATH}" \
            SERVER_HOST="${SERVER_HOST}" \
            SERVER_PORT="${SWEEP_PORT}" \
            CONCURRENCY="${CONCURRENCY_LEVELS}" \
            REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL}" \
            MAX_TOKENS="${MAX_NEW_TOKENS}" \
            M1_STEPS="${M1_STEPS}" \
            M1_RUNS="${M1_RUNS}" \
            M1_WARMUP="${M1_WARMUP}" \
            QUICK="${QUICK}" \
            bash "${SCRIPT_DIR}/sweep_ternary_kernels.sh" "${MODE}"
    ) > "${SWEEP_LOG}" 2>&1; then
        SWEEP_OUT_DIR="$("${PYTHON_BIN}" - "${SGLANG_DIR}" "${MODE}" "${sweep_start}" <<'PYSWEEP'
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
start_ts = float(sys.argv[3])
candidates = []
for path in root.glob(f"kernel_sweep_{mode}_*"):
    if path.is_dir() and path.stat().st_mtime >= start_ts - 2:
        candidates.append(path)
if candidates:
    candidates.sort(key=lambda p: p.stat().st_mtime)
    print(str(candidates[-1]))
PYSWEEP
)"
    else
        echo "Warning: optional kernel sweep failed; see ${SWEEP_LOG}"
    fi
fi

if [[ "${RUN_CAPACITY_COMPARE}" == "1" ]]; then
    echo "[7b/7] Running optional FP16 vs ternary capacity compare..."
    compare_start="$(date +%s)"
    COMPARE_LOG="${OUT_DIR}/optional_compare.log"
    log_cmd env MODEL_PATH="${MODEL_PATH}" SERVER_HOST="${SERVER_HOST}" SERVER_PORT="${COMPARE_PORT}" CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS}" REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" TTFT_TARGET_MS="${TTFT_TARGET_MS}" KV_SAFETY_FACTOR="${KV_SAFETY_FACTOR}" bash "${SCRIPT_DIR}/compare_ternary_fp16_capacity.sh"
    if (
        cd "${SGLANG_DIR}"
        env \
            MODEL_PATH="${MODEL_PATH}" \
            SERVER_HOST="${SERVER_HOST}" \
            SERVER_PORT="${COMPARE_PORT}" \
            CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS}" \
            REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL}" \
            MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
            TTFT_TARGET_MS="${TTFT_TARGET_MS}" \
            KV_SAFETY_FACTOR="${KV_SAFETY_FACTOR}" \
            bash "${SCRIPT_DIR}/compare_ternary_fp16_capacity.sh"
    ) > "${COMPARE_LOG}" 2>&1; then
        COMPARE_OUT_DIR="$("${PYTHON_BIN}" - "${SGLANG_DIR}" "${compare_start}" <<'PYCOMPARE'
import sys
from pathlib import Path

root = Path(sys.argv[1])
start_ts = float(sys.argv[2])
candidates = []
for path in root.glob("capacity_compare_*"):
    if path.is_dir() and path.stat().st_mtime >= start_ts - 2:
        candidates.append(path)
if candidates:
    candidates.sort(key=lambda p: p.stat().st_mtime)
    print(str(candidates[-1]))
PYCOMPARE
)"
    else
        echo "Warning: optional capacity compare failed; see ${COMPARE_LOG}"
    fi
fi

echo "[final] Building consolidated baseline summary..."
log_cmd "${PYTHON_BIN}" "-" "${OUT_DIR}" "${SUMMARY_JSON}" "${SUMMARY_MD}" "${MODE}" "${MODEL_PATH}" "${SERVER_HOST}" "${SERVER_PORT}" "${TTFT_TARGET_MS}" "${SUPPORT_CSV}" "${M1_JSON}" "${TTFT_JSON}" "${CONC_JSON}" "${FAMILY_COUNTERS_JSON}" "${SWEEP_OUT_DIR}" "${COMPARE_OUT_DIR}"
"${PYTHON_BIN}" - "${OUT_DIR}" "${SUMMARY_JSON}" "${SUMMARY_MD}" "${MODE}" "${MODEL_PATH}" "${SERVER_HOST}" "${SERVER_PORT}" "${TTFT_TARGET_MS}" "${SUPPORT_CSV}" "${M1_JSON}" "${TTFT_JSON}" "${CONC_JSON}" "${FAMILY_COUNTERS_JSON}" "${SWEEP_OUT_DIR}" "${COMPARE_OUT_DIR}" <<'PYSUMMARY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

def load_json(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

out_dir = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
summary_md = Path(sys.argv[3])
mode = sys.argv[4]
model_path = sys.argv[5]
host = sys.argv[6]
port = int(sys.argv[7])
ttft_target_ms = float(sys.argv[8])
support_csv = Path(sys.argv[9])
m1_path = Path(sys.argv[10])
ttft_path = Path(sys.argv[11])
conc_path = Path(sys.argv[12])
families_path = Path(sys.argv[13])
sweep_out_dir = sys.argv[14].strip()
compare_out_dir = sys.argv[15].strip()

m1 = load_json(str(m1_path)) or {}
ttft = load_json(str(ttft_path)) or {}
conc = load_json(str(conc_path)) or {}
families = load_json(str(families_path)) or {}

conc_by_level = {}
for row in conc.get("results", []):
    try:
        conc_by_level[int(row.get("concurrency", 0))] = row
    except Exception:
        continue

def metric_at(level: int, key: str):
    return conc_by_level.get(level, {}).get(key)

summary = {
    "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    "mode": mode,
    "model_path": model_path,
    "server": {"host": host, "port": port},
    "ttft_target_ms": ttft_target_ms,
    "artifacts": {
        "support_envelope_csv": str(support_csv),
        "baseline_m1_decode_json": str(m1_path),
        "baseline_ttft_kv_json": str(ttft_path),
        "baseline_concurrency_json": str(conc_path),
        "baseline_family_counters_json": str(families_path),
        "optional_kernel_sweep_dir": sweep_out_dir or None,
        "optional_capacity_compare_dir": compare_out_dir or None,
    },
    "metrics": {
        "m1_decode": {
            "decode_time_p50_ms": m1.get("decode_time_p50"),
            "decode_time_p95_ms": m1.get("decode_time_p95"),
            "tokens_per_sec_avg": m1.get("tps_avg"),
        },
        "ttft_kv": {
            "max_users_under_target": ttft.get("max_users_under_target"),
        },
        "concurrency": {
            "c1_tps_total": metric_at(1, "tokens_per_sec_total"),
            "c16_tps_total": metric_at(16, "tokens_per_sec_total"),
            "c32_tps_total": metric_at(32, "tokens_per_sec_total"),
            "c64_tps_total": metric_at(64, "tokens_per_sec_total"),
            "c1_ttft_p95_ms": metric_at(1, "ttft_p95"),
            "c16_ttft_p95_ms": metric_at(16, "ttft_p95"),
            "c32_ttft_p95_ms": metric_at(32, "ttft_p95"),
            "c64_ttft_p95_ms": metric_at(64, "ttft_p95"),
        },
    },
    "runtime_families": {
        "path_family_counts": families.get("path_family_counts", {}),
        "fallback_family_counts": families.get("fallback_family_counts", {}),
    },
}

summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

md_lines = []
md_lines.append("# Baseline Matrix Summary")
md_lines.append("")
md_lines.append(f"- Generated: `{summary['generated_at_utc']}`")
md_lines.append(f"- Mode: `{mode}`")
md_lines.append(f"- Endpoint: `http://{host}:{port}`")
md_lines.append(f"- TTFT target (p95): `{ttft_target_ms}` ms")
md_lines.append("")
md_lines.append("## Key metrics")
md_lines.append("")
md_lines.append(
    f"- M=1 decode p50/p95 (ms): `{summary['metrics']['m1_decode']['decode_time_p50_ms']}` / "
    f"`{summary['metrics']['m1_decode']['decode_time_p95_ms']}`"
)
md_lines.append(
    f"- M=1 tokens/sec avg: `{summary['metrics']['m1_decode']['tokens_per_sec_avg']}`"
)
md_lines.append(
    f"- Max users under TTFT target: `{summary['metrics']['ttft_kv']['max_users_under_target']}`"
)
md_lines.append(
    f"- Concurrency TPS (C1/C16/C32/C64): "
    f"`{summary['metrics']['concurrency']['c1_tps_total']}` / "
    f"`{summary['metrics']['concurrency']['c16_tps_total']}` / "
    f"`{summary['metrics']['concurrency']['c32_tps_total']}` / "
    f"`{summary['metrics']['concurrency']['c64_tps_total']}`"
)
md_lines.append("")
md_lines.append("## Runtime path/fallback families")
md_lines.append("")
md_lines.append(f"- Path families: `{summary['runtime_families']['path_family_counts']}`")
md_lines.append(f"- Fallback families: `{summary['runtime_families']['fallback_family_counts']}`")
md_lines.append("")
md_lines.append("## Artifacts")
md_lines.append("")
for key, value in summary["artifacts"].items():
    md_lines.append(f"- {key}: `{value}`")

summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print(f"Wrote: {summary_json}")
print(f"Wrote: {summary_md}")
PYSUMMARY

echo ""
echo "=============================================================="
echo "Baseline matrix freeze complete."
echo "=============================================================="
echo "Summary JSON: ${SUMMARY_JSON}"
echo "Summary MD:   ${SUMMARY_MD}"
echo "Commands log: ${COMMAND_LOG}"
echo "Artifacts dir:${OUT_DIR}"
echo "=============================================================="

