#!/usr/bin/env bash
# =============================================================================
# Run H100 M>1 ternary linear PoC probe (shape x M correctness + latency)
# =============================================================================
#
# Usage:
#   bash util/run_h100_mgt1_poc.sh
#   bash util/run_h100_mgt1_poc.sh --quick
#   bash util/run_h100_mgt1_poc.sh --m-values 2,4,8 --iters 40 --warmup 10
#
# Outputs:
#   sglang/mgt1_linear_probe_<timestamp>/probe.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(dirname "$SCRIPT_DIR")"
PROBE_SCRIPT="${SCRIPT_DIR}/benchmark_mgt1_linear.py"

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

LIB_PATH="${LIB_PATH:-${SGLANG_DIR}/libternary_bitnet.so}"
SHAPES="${SHAPES:-5120x2048,2048x4096,2048x2048,1536x2048,768x2048,2048x768}"
M_VALUES="${M_VALUES:-2,4,8,16,32,64}"
ITERS="${ITERS:-80}"
WARMUP="${WARMUP:-20}"
MAX_ABS_THRESHOLD="${MAX_ABS_THRESHOLD:-0.08}"
MAX_REL_THRESHOLD="${MAX_REL_THRESHOLD:-0.25}"
SEED="${SEED:-7}"
OUT_DIR="${OUT_DIR:-}"

usage() {
    cat <<'EOF'
Usage: run_h100_mgt1_poc.sh [options]

Options:
  --lib-path PATH              Path to libternary_bitnet.so
  --shapes CSV                 NxK shapes (default plan set)
  --m-values CSV               M values (default: 2,4,8,16,32,64)
  --iters N                    Timed iterations per case (default: 80)
  --warmup N                   Warmup iterations per case (default: 20)
  --max-abs-threshold X        Correctness max_abs threshold (default: 0.08)
  --max-rel-threshold X        Correctness max_rel threshold (default: 0.25)
  --seed N                     RNG seed (default: 7)
  --out-dir PATH               Output directory
  --quick                      Quick run (iters=30 warmup=8 m-values=2,4,8)
  -h, --help                   Show help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lib-path) LIB_PATH="${2:-}"; shift 2 ;;
        --shapes) SHAPES="${2:-}"; shift 2 ;;
        --m-values) M_VALUES="${2:-}"; shift 2 ;;
        --iters) ITERS="${2:-}"; shift 2 ;;
        --warmup) WARMUP="${2:-}"; shift 2 ;;
        --max-abs-threshold) MAX_ABS_THRESHOLD="${2:-}"; shift 2 ;;
        --max-rel-threshold) MAX_REL_THRESHOLD="${2:-}"; shift 2 ;;
        --seed) SEED="${2:-}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
        --quick)
            ITERS=30
            WARMUP=8
            M_VALUES="2,4,8"
            shift
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

if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="${SGLANG_DIR}/mgt1_linear_probe_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUT_DIR}"

OUT_JSON="${OUT_DIR}/probe.json"
OUT_LOG="${OUT_DIR}/probe.log"

echo "=============================================================="
echo "H100 M>1 Ternary Linear PoC"
echo "=============================================================="
echo "Library:      ${LIB_PATH}"
echo "Shapes:       ${SHAPES}"
echo "M values:     ${M_VALUES}"
echo "Iters/warmup: ${ITERS}/${WARMUP}"
echo "Output dir:   ${OUT_DIR}"
echo "=============================================================="

(
    cd "${SGLANG_DIR}"
    "${PYTHON_BIN}" "${PROBE_SCRIPT}" \
        --lib-path "${LIB_PATH}" \
        --shapes "${SHAPES}" \
        --m-values "${M_VALUES}" \
        --iters "${ITERS}" \
        --warmup "${WARMUP}" \
        --max-abs-threshold "${MAX_ABS_THRESHOLD}" \
        --max-rel-threshold "${MAX_REL_THRESHOLD}" \
        --seed "${SEED}" \
        --output "${OUT_JSON}"
) | tee "${OUT_LOG}"

echo ""
echo "Saved:"
echo "  ${OUT_JSON}"
echo "  ${OUT_LOG}"

