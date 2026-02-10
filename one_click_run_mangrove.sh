#!/usr/bin/env bash
#
# One-click launcher for Mangrove serving from a fresh sglang clone.
#
# Default behavior:
#   1) Ensure Python venv exists at .venv
#   2) Install runtime deps if missing (torch, sglang, aiohttp, etc.)
#   3) Optionally wire/build ternary kernels (if ternarykernels source is available)
#   4) Launch run_server_ternary.sh with production-friendly defaults
#
# Usage:
#   bash one_click_run_mangrove.sh
#   bash one_click_run_mangrove.sh --mode i2s-tuned --model-path /scratch/mangrove
#   bash one_click_run_mangrove.sh --skip-install --skip-kernel-build -- --tp 2
#
# Notes:
# - Extra arguments after `--` are forwarded to run_server_ternary.sh.
# - If /home/ubuntu/ternarykernels exists, this script auto-links it into
#   third_party/ternarykernels and builds/syncs kernels when possible.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${MANGROVE_MODE:-i2s-tuned}"
MODEL_PATH="${MANGROVE_MODEL_PATH:-/scratch/mangrove}"
SERVED_MODEL_NAME="${MANGROVE_SERVED_MODEL_NAME:-}"
TERNARY_SOURCE_PATH="${TERNARY_SOURCE_PATH:-/home/ubuntu/ternarykernels}"
SKIP_INSTALL=0
SKIP_KERNEL_BUILD=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: one_click_run_mangrove.sh [OPTIONS] [-- <run_server_ternary extra args>]

Options:
  --mode MODE                 Quant mode for run_server_ternary.sh (default: i2s-tuned)
  --model-path PATH           Model path (default: /scratch/mangrove)
  --served-model-name NAME    Served model name (default: mangrove-<mode>)
  --ternary-source PATH       ternarykernels source path (default: /home/ubuntu/ternarykernels)
  --skip-install              Skip Python dependency installation/bootstrap
  --skip-kernel-build         Skip ternary kernel build/sync
  -h, --help                  Show this help

Examples:
  bash one_click_run_mangrove.sh
  bash one_click_run_mangrove.sh --mode i2s-tuned --model-path /scratch/mangrove
  bash one_click_run_mangrove.sh -- --tp 2
EOF
}

# Optional positional mode:
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  MODE="$1"
  shift
fi

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
    --ternary-source)
      TERNARY_SOURCE_PATH="${2:-}"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --skip-kernel-build)
      SKIP_KERNEL_BUILD=1
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$SERVED_MODEL_NAME" ]]; then
  SERVED_MODEL_NAME="mangrove-${MODE}"
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Error: model path does not exist: $MODEL_PATH"
  echo "Set with --model-path PATH or MANGROVE_MODEL_PATH."
  exit 1
fi

ensure_python_venv() {
  if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    return 0
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found."
    return 1
  fi

  # Try creating venv; if ensurepip is missing, install python3.10-venv.
  if ! python3 -m venv "$SCRIPT_DIR/.venv" >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
      echo "Installing python3.10-venv (required for virtualenv bootstrap)..."
      sudo apt-get update
      sudo apt-get install -y python3.10-venv
      python3 -m venv "$SCRIPT_DIR/.venv"
    else
      echo "Error: failed to create .venv and cannot sudo-install python3.10-venv."
      return 1
    fi
  fi
}

deps_ready() {
  "$SCRIPT_DIR/.venv/bin/python" - <<'PY' >/dev/null 2>&1
import importlib
mods = ["numpy", "torch", "aiohttp", "sglang"]
for m in mods:
    if importlib.util.find_spec(m) is None:
        raise SystemExit(1)
print("ok")
PY
}

install_runtime_deps() {
  source "$SCRIPT_DIR/.venv/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install uv

  # Follows current working install path used for this mangrove branch.
  uv pip install -e "python[dev]" --no-deps --index-strategy unsafe-best-match --force-reinstall
  uv pip install flashinfer-python==0.4.1 --prerelease=allow --index-strategy unsafe-best-match
  uv pip install -e "python[dev]" \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match \
    --upgrade
}

maybe_setup_and_build_kernels() {
  if [[ "$SKIP_KERNEL_BUILD" -eq 1 ]]; then
    return 0
  fi

  if [[ -d "$TERNARY_SOURCE_PATH/mangrove-turbo" ]]; then
    echo "Linking ternary source: $TERNARY_SOURCE_PATH"
    bash "$SCRIPT_DIR/util/setup_mangrove_monorepo.sh" \
      --source "$TERNARY_SOURCE_PATH" \
      --model-path "$MODEL_PATH" || true
  fi

  if [[ -d "$SCRIPT_DIR/third_party/ternarykernels" ]] || [[ -d "$SCRIPT_DIR/../ternarykernels" ]]; then
    echo "Building/syncing ternary kernels (best effort)..."
    if ! bash "$SCRIPT_DIR/util/build_sync_ternary_kernels.sh"; then
      echo "Warning: kernel build/sync failed; continuing with available runtime kernels."
    fi
  else
    echo "No ternarykernels source found; skipping kernel build."
  fi
}

echo "=============================================================="
echo "Mangrove One-Click Launch"
echo "=============================================================="
echo "Mode:              $MODE"
echo "Model path:        $MODEL_PATH"
echo "Served model name: $SERVED_MODEL_NAME"
echo "Skip install:      $SKIP_INSTALL"
echo "Skip kernel build: $SKIP_KERNEL_BUILD"
echo "=============================================================="

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  ensure_python_venv
  if ! deps_ready; then
    echo "Installing Python runtime dependencies..."
    install_runtime_deps
  else
    echo "Python runtime dependencies already satisfied."
  fi
else
  echo "Skipping Python dependency installation."
fi

maybe_setup_and_build_kernels

echo "Launching server..."
exec bash "$SCRIPT_DIR/run_server_ternary.sh" \
  "$MODE" \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  "${EXTRA_ARGS[@]}"
