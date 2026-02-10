#!/usr/bin/env bash
#
# Launch a hybrid setup:
# - ternary backend tuned for low-latency / C=1 speed
# - fp16 backend tuned for high-concurrency capacity
# - lightweight router that swaps between them based on in-flight load
#
# Default ports:
#   router : 30080 (public endpoint)
#   ternary: 30081
#   fp16   : 30082

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(dirname "$SCRIPT_DIR")"
RUN_SERVER="${SGLANG_DIR}/run_server_ternary.sh"
HYBRID_ROUTER="${SCRIPT_DIR}/hybrid_runtime_router.py"

MODEL_PATH="${MODEL_PATH:-/scratch/mangrove}"
ROUTER_PORT="${ROUTER_PORT:-30080}"
TERNARY_PORT="${TERNARY_PORT:-30081}"
FP16_PORT="${FP16_PORT:-30082}"
FP16_DP="${FP16_DP:-}"
SWITCH_INFLIGHT_THRESHOLD="${SWITCH_INFLIGHT_THRESHOLD:-48}"
TERNARY_MODE="${TERNARY_MODE:-i2s-maxspeed}"
HOST="${HOST:-127.0.0.1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
TERNARY_CUDA_VISIBLE_DEVICES="${TERNARY_CUDA_VISIBLE_DEVICES:-}"
FP16_CUDA_VISIBLE_DEVICES="${FP16_CUDA_VISIBLE_DEVICES:-}"

join_csv_from_index() {
    local -n _arr_ref=$1
    local start_idx=$2
    local out=""
    local i
    for ((i=start_idx; i<${#_arr_ref[@]}; i++)); do
        if [[ -n "$out" ]]; then
            out+=","
        fi
        out+="${_arr_ref[$i]}"
    done
    echo "$out"
}

detect_gpu_ids_csv() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "${CUDA_VISIBLE_DEVICES}"
        return 0
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local ids
        ids="$(
            nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
                | tr '\n' ',' | sed 's/,$//'
        )"
        if [[ -n "$ids" ]]; then
            echo "$ids"
            return 0
        fi
    fi
    echo "0"
}

ALL_GPU_IDS="$(detect_gpu_ids_csv)"
IFS=',' read -r -a ALL_GPU_ARR <<< "$ALL_GPU_IDS"

if [[ -z "$TERNARY_CUDA_VISIBLE_DEVICES" && -z "$FP16_CUDA_VISIBLE_DEVICES" ]]; then
    # Default split uses all detected GPUs:
    # - 8+ GPUs: 2 for ternary lane, rest for fp16 lane
    # - 4-7 GPUs: 1 for ternary lane, rest for fp16 lane
    # - <=3 GPUs: best-effort split
    if (( ${#ALL_GPU_ARR[@]} >= 8 )); then
        TERNARY_CUDA_VISIBLE_DEVICES="${ALL_GPU_ARR[0]},${ALL_GPU_ARR[1]}"
        FP16_CUDA_VISIBLE_DEVICES="$(join_csv_from_index ALL_GPU_ARR 2)"
    elif (( ${#ALL_GPU_ARR[@]} >= 4 )); then
        TERNARY_CUDA_VISIBLE_DEVICES="${ALL_GPU_ARR[0]}"
        FP16_CUDA_VISIBLE_DEVICES="$(join_csv_from_index ALL_GPU_ARR 1)"
    elif (( ${#ALL_GPU_ARR[@]} >= 2 )); then
        TERNARY_CUDA_VISIBLE_DEVICES="${ALL_GPU_ARR[0]}"
        FP16_CUDA_VISIBLE_DEVICES="${ALL_GPU_ARR[1]}"
    else
        TERNARY_CUDA_VISIBLE_DEVICES="${ALL_GPU_ARR[0]}"
        FP16_CUDA_VISIBLE_DEVICES="${ALL_GPU_ARR[0]}"
    fi
elif [[ -n "$TERNARY_CUDA_VISIBLE_DEVICES" && -z "$FP16_CUDA_VISIBLE_DEVICES" ]]; then
    # Derive fp16 lane as "all GPUs minus ternary lane", fallback to ternary list.
    declare -A TERNARY_SET=()
    IFS=',' read -r -a _TERNARY_ARR <<< "$TERNARY_CUDA_VISIBLE_DEVICES"
    for g in "${_TERNARY_ARR[@]}"; do
        TERNARY_SET["$g"]=1
    done
    derived=""
    for g in "${ALL_GPU_ARR[@]}"; do
        if [[ -z "${TERNARY_SET[$g]:-}" ]]; then
            if [[ -n "$derived" ]]; then
                derived+=","
            fi
            derived+="$g"
        fi
    done
    FP16_CUDA_VISIBLE_DEVICES="${derived:-$TERNARY_CUDA_VISIBLE_DEVICES}"
elif [[ -z "$TERNARY_CUDA_VISIBLE_DEVICES" && -n "$FP16_CUDA_VISIBLE_DEVICES" ]]; then
    # Pick first GPU not in fp16 lane for ternary, fallback to first fp16 GPU.
    declare -A FP16_SET=()
    IFS=',' read -r -a _FP16_ARR <<< "$FP16_CUDA_VISIBLE_DEVICES"
    for g in "${_FP16_ARR[@]}"; do
        FP16_SET["$g"]=1
    done
    pick=""
    for g in "${ALL_GPU_ARR[@]}"; do
        if [[ -z "${FP16_SET[$g]:-}" ]]; then
            pick="$g"
            break
        fi
    done
    TERNARY_CUDA_VISIBLE_DEVICES="${pick:-${_FP16_ARR[0]}}"
fi

if [[ -z "$FP16_DP" ]]; then
    if [[ -n "$FP16_CUDA_VISIBLE_DEVICES" ]]; then
        FP16_DP="$(awk -F',' '{print NF}' <<< "$FP16_CUDA_VISIBLE_DEVICES")"
    else
        FP16_DP=1
    fi
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: model path not found: $MODEL_PATH"
    exit 1
fi

if [[ ! -x "${SGLANG_DIR}/.venv/bin/python" ]]; then
    echo "Error: missing ${SGLANG_DIR}/.venv/bin/python"
    exit 1
fi

kill_port() {
    local p="$1"
    local pids
    pids="$(lsof -t -i:"$p" 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
        echo "Killing processes on :$p -> $pids"
        echo "$pids" | xargs -r kill -9 2>/dev/null || true
        sleep 1
    fi
}

wait_ready() {
    local base="$1"
    local timeout="$2"
    local start now
    start="$(date +%s)"
    while true; do
        if curl -sf "${base}/health" >/dev/null 2>&1; then
            local body
            body="$(curl -s -X POST "${base}/generate" \
                -H "Content-Type: application/json" \
                -d '{"text":"probe","sampling_params":{"max_new_tokens":4,"temperature":0}}' || true)"
            if echo "$body" | grep -q "meta_info"; then
                return 0
            fi
        fi
        now="$(date +%s)"
        if [[ $((now - start)) -ge "$timeout" ]]; then
            return 1
        fi
        sleep 2
    done
}

echo "=============================================================="
echo "Hybrid swap launch"
echo "=============================================================="
echo "Model path:                $MODEL_PATH"
echo "Router endpoint:           http://${HOST}:${ROUTER_PORT}"
echo "Ternary backend:           http://${HOST}:${TERNARY_PORT} (${TERNARY_MODE})"
echo "FP16 backend:              http://${HOST}:${FP16_PORT} (dp=${FP16_DP})"
echo "Ternary GPUs:              ${TERNARY_CUDA_VISIBLE_DEVICES}"
echo "FP16 GPUs:                 ${FP16_CUDA_VISIBLE_DEVICES}"
echo "Swap threshold (inflight): ${SWITCH_INFLIGHT_THRESHOLD}"
echo "=============================================================="

kill_port "$ROUTER_PORT"
kill_port "$TERNARY_PORT"
kill_port "$FP16_PORT"

mkdir -p "${SGLANG_DIR}/hybrid_logs"
TS="$(date +%Y%m%d_%H%M%S)"
TERNARY_LOG="${SGLANG_DIR}/hybrid_logs/ternary_${TS}.log"
FP16_LOG="${SGLANG_DIR}/hybrid_logs/fp16_${TS}.log"
ROUTER_LOG="${SGLANG_DIR}/hybrid_logs/router_${TS}.log"

# Start fp16 capacity lane
(
    cd "$SGLANG_DIR"
    export CUDA_VISIBLE_DEVICES="$FP16_CUDA_VISIBLE_DEVICES"
    bash "$RUN_SERVER" fp16 --dp "$FP16_DP" --port "$FP16_PORT" \
        --model-path "$MODEL_PATH" --served-model-name "mangrove-fp16-hybrid"
) >"$FP16_LOG" 2>&1 &
FP16_PID=$!
echo "Started fp16 backend pid=$FP16_PID log=$FP16_LOG"

# Start ternary speed lane
(
    cd "$SGLANG_DIR"
    export CUDA_VISIBLE_DEVICES="$TERNARY_CUDA_VISIBLE_DEVICES"
    export TERNARY_ENABLE_PDL=1
    export TERNARY_FUSE_RMSNORM_QKV=1
    export SGLANG_TERNARY_I2S_SPLITS=1
    export TERNARY_MOE_DECODE_ONLY=0
    export SPECULATIVE_ALGORITHM=NGRAM
    export SPECULATIVE_NUM_DRAFT=4
    bash "$RUN_SERVER" "$TERNARY_MODE" --port "$TERNARY_PORT" \
        --model-path "$MODEL_PATH" --served-model-name "mangrove-ternary-hybrid"
) >"$TERNARY_LOG" 2>&1 &
TERNARY_PID=$!
echo "Started ternary backend pid=$TERNARY_PID log=$TERNARY_LOG"

cleanup() {
    echo "Cleaning up hybrid processes..."
    kill -9 "$TERNARY_PID" "$FP16_PID" "$ROUTER_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "Waiting for backends to become ready..."
if ! wait_ready "http://${HOST}:${FP16_PORT}" "$STARTUP_TIMEOUT"; then
    echo "Error: fp16 backend did not become ready. See $FP16_LOG"
    exit 1
fi
echo "FP16 backend ready."

if ! wait_ready "http://${HOST}:${TERNARY_PORT}" "$STARTUP_TIMEOUT"; then
    echo "Error: ternary backend did not become ready. See $TERNARY_LOG"
    exit 1
fi
echo "Ternary backend ready."

echo "Starting hybrid router..."
(
    cd "$SGLANG_DIR"
    .venv/bin/python "$HYBRID_ROUTER" \
        --host "$HOST" \
        --port "$ROUTER_PORT" \
        --ternary-url "http://${HOST}:${TERNARY_PORT}" \
        --fp16-url "http://${HOST}:${FP16_PORT}" \
        --switch-inflight-threshold "$SWITCH_INFLIGHT_THRESHOLD" \
        --force-backend auto
) >"$ROUTER_LOG" 2>&1 &
ROUTER_PID=$!
echo "Hybrid router pid=$ROUTER_PID log=$ROUTER_LOG"

echo "Waiting for hybrid router health..."
if ! wait_ready "http://${HOST}:${ROUTER_PORT}" 120; then
    echo "Error: hybrid router failed readiness. See $ROUTER_LOG"
    exit 1
fi

echo "=============================================================="
echo "Hybrid endpoint is ready: http://${HOST}:${ROUTER_PORT}"
echo "Stats endpoint:           http://${HOST}:${ROUTER_PORT}/routing_stats"
echo "Router log:               $ROUTER_LOG"
echo "Ternary log:              $TERNARY_LOG"
echo "FP16 log:                 $FP16_LOG"
echo "Press Ctrl+C to stop."
echo "=============================================================="

wait "$ROUTER_PID"
