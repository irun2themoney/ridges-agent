#!/bin/bash
# Run production test sets to measure SWE-bench performance

set -e

INFERENCE_GATEWAY_URL="${RIDGES_INFERENCE_GATEWAY_URL:-http://127.0.0.1:7001}"
AGENT_PATH="agents/top_agent/agent.py"

echo "=========================================="
echo "Running Production Test Sets"
echo "=========================================="
echo "Inference Gateway: $INFERENCE_GATEWAY_URL"
echo "Agent Path: $AGENT_PATH"
echo ""

# Check if inference gateway is running
if ! curl -s "$INFERENCE_GATEWAY_URL/health" > /dev/null 2>&1; then
    echo "⚠️  Warning: Inference gateway might not be running at $INFERENCE_GATEWAY_URL"
    echo "   Make sure it's running before continuing!"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

cd ridges

# Run screener-1 first (smallest set, quickest validation)
echo ""
echo "=========================================="
echo "1. Running screener-1 (10 problems)"
echo "=========================================="
python3 test_agent.py \
    --inference-url "$INFERENCE_GATEWAY_URL" \
    --agent-path "../$AGENT_PATH" \
    --agent-timeout 2400 \
    --eval-timeout 600 \
    test-problem-set screener-1

echo ""
echo "=========================================="
echo "✅ screener-1 completed!"
echo "=========================================="
echo ""
echo "Results saved to: ridges/test_agent_results/"
echo ""
echo "Next steps:"
echo "  - Review screener-1 results"
echo "  - If results look good, run screener-2:"
echo "    cd ridges && python3 test_agent.py --inference-url $INFERENCE_GATEWAY_URL --agent-path ../$AGENT_PATH test-problem-set screener-2"
echo ""
