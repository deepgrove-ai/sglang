#!/bin/bash
# Benchmark script that runs specific scenarios and outputs a formatted table
#
# Works with servers launched via run_server_ternary_fp8.sh (port 30080)
#
# Usage:
#   ./benchmark_scenarios.sh [--host 127.0.0.1] [--port 30080] [--concurrency 4]
#
# Examples:
#   ./benchmark_scenarios.sh                                    # Use defaults
#   ./benchmark_scenarios.sh --port 30080 --concurrency 8       # Higher concurrency
#   ./benchmark_scenarios.sh --scenarios decode_short_story     # Run specific scenario

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate sglang venv if available
if [ -f "$SCRIPT_DIR/sglang/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/sglang/.venv/bin/activate"
fi

python "$SCRIPT_DIR/benchmark_scenarios.py" "$@"
