#!/bin/bash
# Test different latency optimization configurations
# Usage: ./test_latency_optimizations.sh [config]
# Configs: baseline, chunked, ngram, both

CONFIG="${1:-baseline}"

cd /root/raghav
source sglang/.venv/bin/activate

case "$CONFIG" in
    baseline)
        echo "=== Testing BASELINE (no optimizations) ==="
        export CHUNKED_PREFILL_SIZE=""
        export SPECULATIVE_ALGORITHM=""
        ;;
    chunked)
        echo "=== Testing CHUNKED PREFILL (1024) ==="
        export CHUNKED_PREFILL_SIZE="1024"
        export SPECULATIVE_ALGORITHM=""
        ;;
    ngram)
        echo "=== Testing NGRAM SPECULATIVE DECODING ==="
        export CHUNKED_PREFILL_SIZE=""
        export SPECULATIVE_ALGORITHM="NGRAM"
        export SPECULATIVE_NUM_DRAFT="4"
        ;;
    both)
        echo "=== Testing BOTH (chunked + ngram) ==="
        export CHUNKED_PREFILL_SIZE="1024"
        export SPECULATIVE_ALGORITHM="NGRAM"
        export SPECULATIVE_NUM_DRAFT="4"
        ;;
    *)
        echo "Unknown config: $CONFIG"
        echo "Usage: $0 [baseline|chunked|ngram|both]"
        exit 1
        ;;
esac

echo ""
echo "Config: CHUNKED_PREFILL_SIZE=$CHUNKED_PREFILL_SIZE"
echo "Config: SPECULATIVE_ALGORITHM=$SPECULATIVE_ALGORITHM"
echo ""
echo "Starting server with config: $CONFIG"
echo "Wait for server to be ready, then run benchmark in another terminal:"
echo "  ./benchmark_concurrency.sh -r 64 -c 1,2,4,8,16,32,64"
echo ""

# Start the server
cd sglang
./run_server_ternary.sh i2s-tuned
