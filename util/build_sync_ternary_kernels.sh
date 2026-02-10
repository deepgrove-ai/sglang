#!/usr/bin/env bash

# Build ternary kernel libraries from ternarykernels and sync them into SGLang.
#
# This script is monorepo-aware and supports these layouts:
#   1) sglang/third_party/ternarykernels (recommended)
#   2) sglang/ternarykernels
#   3) ../ternarykernels (legacy split checkout)
#
# Usage examples:
#   ./util/build_sync_ternary_kernels.sh
#   ./util/build_sync_ternary_kernels.sh --ternary-root /home/ubuntu/ternarykernels --debug
#   ./util/build_sync_ternary_kernels.sh --cutlass-only --sync-mode link

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(dirname "$SCRIPT_DIR")"

BUILD_MODE="release"
SYNC_MODE="copy"
SKIP_BUILD=0
BUILD_BITNET=1
BUILD_CUTLASS=1
EFFECTIVE_BUILD_CUTLASS=1
TERNARY_ROOT="${TERNARY_ROOT:-}"
TARGET_ARCH="${ARCH:-}"

ensure_nvcc_on_path() {
  if command -v nvcc >/dev/null 2>&1; then
    return 0
  fi
  local cuda_candidates=(
    "/usr/local/cuda/bin"
    "/usr/local/cuda-12/bin"
    "/usr/local/cuda-12.6/bin"
  )
  local cuda_bin
  for cuda_bin in "${cuda_candidates[@]}"; do
    if [[ -x "${cuda_bin}/nvcc" ]]; then
      export PATH="${cuda_bin}:${PATH}"
      # Best-effort CUDA_HOME for downstream scripts/tools.
      export CUDA_HOME="$(dirname "${cuda_bin}")"
      return 0
    fi
  done
  return 1
}

detect_gpu_compute_major() {
  local cc_raw
  cc_raw="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  if [[ -z "${cc_raw}" ]]; then
    echo ""
    return 0
  fi
  # Expected formats like "9.0" or "10.0"
  local major="${cc_raw%%.*}"
  if [[ "${major}" =~ ^[0-9]+$ ]]; then
    echo "${major}"
  else
    echo ""
  fi
}

usage() {
  cat <<'EOF'
Usage: build_sync_ternary_kernels.sh [OPTIONS]

Build options:
  --debug                     Build with debug symbols/lineinfo.
  --skip-build                Skip compilation and only sync existing artifacts.
  --bitnet-only               Only sync/build libternary_bitnet.so.
  --cutlass-only              Only sync/build libternary_cutlass_sm100.so.
  --arch <arch>               Override target arch (e.g. 90, 100, 100a).

Path options:
  --ternary-root <path>       Path to ternarykernels repo root.
                              (auto-detected if omitted)
  --sync-mode <copy|link>     copy (default) or symlink artifacts into sglang root.

Other:
  -h, --help                  Show this help.

Examples:
  ./util/build_sync_ternary_kernels.sh
  ./util/build_sync_ternary_kernels.sh --cutlass-only --arch 100a
  ./util/build_sync_ternary_kernels.sh --ternary-root /home/ubuntu/ternarykernels --sync-mode link
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      BUILD_MODE="debug"
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --bitnet-only)
      BUILD_BITNET=1
      BUILD_CUTLASS=0
      shift
      ;;
    --cutlass-only)
      BUILD_BITNET=0
      BUILD_CUTLASS=1
      shift
      ;;
    --arch)
      TARGET_ARCH="${2:-}"
      if [[ -z "${TARGET_ARCH}" ]]; then
        echo "Error: --arch requires a value"
        exit 1
      fi
      shift 2
      ;;
    --ternary-root)
      TERNARY_ROOT="${2:-}"
      if [[ -z "${TERNARY_ROOT}" ]]; then
        echo "Error: --ternary-root requires a value"
        exit 1
      fi
      shift 2
      ;;
    --sync-mode)
      SYNC_MODE="${2:-}"
      if [[ "${SYNC_MODE}" != "copy" && "${SYNC_MODE}" != "link" ]]; then
        echo "Error: --sync-mode must be 'copy' or 'link'"
        exit 1
      fi
      shift 2
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

resolve_ternary_root() {
  if [[ -n "${TERNARY_ROOT}" ]]; then
    if [[ -d "${TERNARY_ROOT}/mangrove-turbo" ]]; then
      echo "${TERNARY_ROOT}"
      return 0
    fi
    echo "Error: --ternary-root '${TERNARY_ROOT}' does not look like ternarykernels root" >&2
    return 1
  fi

  local candidates=(
    "${SGLANG_DIR}/third_party/ternarykernels"
    "${SGLANG_DIR}/ternarykernels"
    "${SGLANG_DIR}/../ternarykernels"
    "/home/ubuntu/ternarykernels"
  )

  local cand
  for cand in "${candidates[@]}"; do
    if [[ -d "${cand}/mangrove-turbo" ]]; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

sync_artifact() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "${src}" ]]; then
    echo "Error: missing artifact: ${src}"
    exit 1
  fi

  mkdir -p "$(dirname "${dst}")"
  if [[ "${SYNC_MODE}" == "link" ]]; then
    ln -sfn "${src}" "${dst}"
  else
    cp -f "${src}" "${dst}"
  fi
  echo "Synced: ${dst}"
}

TERNARY_ROOT="$(resolve_ternary_root)" || {
  echo "Error: could not find ternarykernels root."
  echo "       Expected one of:"
  echo "         - ${SGLANG_DIR}/third_party/ternarykernels"
  echo "         - ${SGLANG_DIR}/ternarykernels"
  echo "         - ${SGLANG_DIR}/../ternarykernels"
  echo "       Or pass --ternary-root explicitly."
  exit 1
}
TURBO_DIR="${TERNARY_ROOT}/mangrove-turbo"

echo "SGLang root:      ${SGLANG_DIR}"
echo "Ternary root:     ${TERNARY_ROOT}"
echo "Build mode:       ${BUILD_MODE}"
echo "Sync mode:        ${SYNC_MODE}"
echo "Build bitnet:     ${BUILD_BITNET}"
echo "Build cutlass:    ${BUILD_CUTLASS}"
echo "Skip build:       ${SKIP_BUILD}"
if [[ -n "${TARGET_ARCH}" ]]; then
  echo "Target arch:      ${TARGET_ARCH}"
fi

if [[ "${BUILD_CUTLASS}" -eq 1 ]]; then
  EFFECTIVE_BUILD_CUTLASS=1
  if [[ -z "${TARGET_ARCH}" ]]; then
    gpu_major="$(detect_gpu_compute_major)"
    if [[ -n "${gpu_major}" && "${gpu_major}" -lt 10 ]]; then
      echo "Skipping CUTLASS SM100 build on this node (detected compute capability major=${gpu_major})."
      EFFECTIVE_BUILD_CUTLASS=0
    fi
  fi
fi

if ! ensure_nvcc_on_path; then
  echo "Error: nvcc not found. Add CUDA toolkit bin to PATH or install CUDA compiler."
  exit 1
fi

build_flags=()
if [[ "${BUILD_MODE}" == "debug" ]]; then
  build_flags+=(--debug)
fi

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  if [[ "${BUILD_BITNET}" -eq 1 ]]; then
    echo "Building libternary_bitnet.so..."
    if [[ -n "${TARGET_ARCH}" ]]; then
      (cd "${TURBO_DIR}" && ARCH="${TARGET_ARCH}" bash ./build.sh "${build_flags[@]}")
    else
      (cd "${TURBO_DIR}" && bash ./build.sh "${build_flags[@]}")
    fi
  fi

  if [[ "${EFFECTIVE_BUILD_CUTLASS}" -eq 1 ]]; then
    echo "Building libternary_cutlass_sm100.so..."
    if [[ -n "${TARGET_ARCH}" ]]; then
      if ! (cd "${TURBO_DIR}" && ARCH="${TARGET_ARCH}" bash ./build_cutlass_sm100.sh "${build_flags[@]}"); then
        echo "Warning: CUTLASS build failed; continuing without libternary_cutlass_sm100.so"
        EFFECTIVE_BUILD_CUTLASS=0
      fi
    else
      if ! (cd "${TURBO_DIR}" && bash ./build_cutlass_sm100.sh "${build_flags[@]}"); then
        echo "Warning: CUTLASS build failed; continuing without libternary_cutlass_sm100.so"
        EFFECTIVE_BUILD_CUTLASS=0
      fi
    fi
  fi
fi

if [[ "${BUILD_BITNET}" -eq 1 ]]; then
  sync_artifact "${TURBO_DIR}/libternary_bitnet.so" "${SGLANG_DIR}/libternary_bitnet.so"
fi
if [[ "${EFFECTIVE_BUILD_CUTLASS}" -eq 1 ]]; then
  sync_artifact "${TURBO_DIR}/libternary_cutlass_sm100.so" "${SGLANG_DIR}/libternary_cutlass_sm100.so"
fi

echo
echo "Done."
echo "Recommended env for serving:"
if [[ "${EFFECTIVE_BUILD_CUTLASS}" -eq 1 ]]; then
  echo "  export SGLANG_I2S_CUTLASS_LIB=${SGLANG_DIR}/libternary_cutlass_sm100.so"
fi
