#!/bin/bash
# Simple script to send a query to SGLang server
# Usage: ./send_query.sh "Your question here"
#        ./send_query.sh "Your question" --max-tokens 200

HOST="${SGLANG_HOST:-127.0.0.1}"
PORT="${SGLANG_PORT:-30080}"
MAX_TOKENS=100

# Parse arguments
QUERY=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        *)
            QUERY="$1"
            shift
            ;;
    esac
done

if [[ -z "$QUERY" ]]; then
    echo "Usage: $0 \"Your question here\" [--max-tokens N] [--port PORT] [--host HOST]"
    echo ""
    echo "Examples:"
    echo "  $0 \"What is machine learning?\""
    echo "  $0 \"Write a poem\" --max-tokens 200"
    exit 1
fi

# Send request and extract text
RESPONSE=$(curl -s "http://${HOST}:${PORT}/generate" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$QUERY\", \"sampling_params\": {\"max_new_tokens\": $MAX_TOKENS, \"temperature\": 0.5, \"repetition_penalty\": 1.1}}")

# Check if jq is available for pretty output
if command -v jq &> /dev/null; then
    echo "$RESPONSE" | jq -r '.text // .error // .'
else
    # Fallback: extract text field with grep/sed
    echo "$RESPONSE" | grep -o '"text":"[^"]*"' | sed 's/"text":"//;s/"$//' | sed 's/\\n/\n/g'
fi

