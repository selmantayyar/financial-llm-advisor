#!/bin/bash
# Deployment script for Financial LLM Advisor API
# Usage: bash scripts/deploy.sh [model_path] [port] [host]

set -e

echo "========================================"
echo "  Financial LLM Advisor API Server"
echo "========================================"

# Configuration
MODEL_PATH="${1:-./checkpoints/final_model}"
PORT="${2:-8000}"
HOST="${3:-0.0.0.0}"
WORKERS="${WORKERS:-1}"
RELOAD="${RELOAD:-false}"

echo ""
echo "Configuration:"
echo "  Model path: $MODEL_PATH"
echo "  Host:       $HOST"
echo "  Port:       $PORT"
echo "  Workers:    $WORKERS"
echo ""

# Check if model exists
if [ -d "$MODEL_PATH" ]; then
    echo "Using model from: $MODEL_PATH"
    export LORA_WEIGHTS="$MODEL_PATH"
else
    echo "Warning: Model path not found. Using base model only."
fi

# Set environment variables
export FINANCIAL_LLM_MODEL_PATH="$MODEL_PATH"

# Check if uvicorn is available
if ! python -c "import uvicorn" 2>/dev/null; then
    echo "Error: uvicorn not installed. Run: pip install uvicorn"
    exit 1
fi

echo ""
echo "Starting API server..."
echo "  API docs: http://$HOST:$PORT/docs"
echo "  Health:   http://$HOST:$PORT/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Start server
if [ "$RELOAD" = "true" ]; then
    python -m uvicorn src.inference:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload
else
    python -m uvicorn src.inference:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS"
fi
