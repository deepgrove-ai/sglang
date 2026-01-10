#!/bin/bash
# Concurrency Benchmark: Test throughput and latency at various concurrency levels
#
# Compares ternary (i2s-tuned) vs FP16 server performance under load.
#
# Usage:
#   ./benchmark_concurrency.sh                                    # Basic run
#   ./benchmark_concurrency.sh --port 30080 --label "i2s-tuned"   # Label results
#   ./benchmark_concurrency.sh -c 1,4,16,32 -r 100                # Custom levels
#
# To compare i2s-tuned vs FP16:
#   # Run against i2s-tuned server (port 30080)
#   ./benchmark_concurrency.sh --port 30080 --label "i2s-tuned" -o results_i2s.json
#
#   # Run against FP16 server (port 30081)  
#   ./benchmark_concurrency.sh --port 30081 --label "fp16" -o results_fp16.json
#
# Options:
#   --host HOST           Server host (default: 127.0.0.1)
#   --port PORT           Server port (default: 30080)
#   -c, --concurrency-levels LEVELS
#                         Comma-separated concurrency levels (default: 1,2,4,8,16,32,64)
#   -r, --requests N      Requests per concurrency level (default: 50)
#   -t, --max-tokens N    Max tokens per request (default: 256)
#   -l, --label LABEL     Label for this run (e.g., "i2s-tuned", "fp16")
#   -o, --output FILE     Save results to JSON file
#   --warmup N            Warmup requests (default: 5)
#   --timeout SECS        Request timeout (default: 120)
#   --no-ignore-eos       Allow early stopping at EOS token
#   --streaming           Use streaming for accurate TTFT measurement
#
# Examples:
#   # Quick sanity check
#   ./benchmark_concurrency.sh -c 1,4 -r 10
#
#   # Full benchmark with results saved
#   ./benchmark_concurrency.sh -c 1,2,4,8,16,32,64,128 -r 100 -l "i2s-tuned" -o i2s.json
#
#   # Stress test (high concurrency)
#   ./benchmark_concurrency.sh -c 64,128,256 -r 200 -t 512

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
HOST="${SGLANG_HOST:-127.0.0.1}"
PORT="${SGLANG_PORT:-30080}"
CONCURRENCY_LEVELS="1,2,4,8,16,32,64"
REQUESTS=50
MAX_TOKENS=256
LABEL=""
OUTPUT=""
WARMUP=5
TIMEOUT=120
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -c|--concurrency-levels)
            CONCURRENCY_LEVELS="$2"
            shift 2
            ;;
        -r|--requests)
            REQUESTS="$2"
            shift 2
            ;;
        -t|--max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -l|--label)
            LABEL="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --no-ignore-eos)
            EXTRA_ARGS="$EXTRA_ARGS --no-ignore-eos"
            shift
            ;;
        --streaming)
            EXTRA_ARGS="$EXTRA_ARGS --streaming"
            shift
            ;;
        -h|--help)
            head -50 "$0" | tail -45
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="python $SCRIPT_DIR/benchmark_concurrency.py"
CMD="$CMD --host $HOST"
CMD="$CMD --port $PORT"
CMD="$CMD --concurrency-levels $CONCURRENCY_LEVELS"
CMD="$CMD --requests $REQUESTS"
CMD="$CMD --max-tokens $MAX_TOKENS"
CMD="$CMD --warmup $WARMUP"
CMD="$CMD --timeout $TIMEOUT"

if [[ -n "$LABEL" ]]; then
    CMD="$CMD --label \"$LABEL\""
fi

if [[ -n "$OUTPUT" ]]; then
    CMD="$CMD --output \"$OUTPUT\""
fi

CMD="$CMD $EXTRA_ARGS"

# Print what we're running
echo "=========================================="
echo "CONCURRENCY BENCHMARK"
echo "=========================================="
echo "Host:        $HOST"
echo "Port:        $PORT"
echo "Concurrency: $CONCURRENCY_LEVELS"
echo "Requests:    $REQUESTS per level"
echo "Max Tokens:  $MAX_TOKENS"
echo "Label:       ${LABEL:-'(none)'}"
echo "Output:      ${OUTPUT:-'(stdout only)'}"
echo "=========================================="
echo ""

# Run benchmark
eval $CMD




