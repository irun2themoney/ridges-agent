#!/bin/bash

cd "$(dirname "$0")" || exit 1

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                   🧪 FULL RIDGES LOCAL TEST - SCREENER 1                      ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "This will test your agent against 10 problems (5 Polyglot + 5 SWE-bench)"
echo "Expected duration: 30-60 minutes"
echo ""

# Check if Python venv is activated
if [[ ! -v VIRTUAL_ENV ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "Creating and activating venv..."
    
    # Use uv to create venv
    ~/.local/bin/uv venv --python 3.13 || python3 -m venv .venv
    source .venv/bin/activate || source .venv/Scripts/activate
    echo "✅ Venv activated"
else
    echo "✅ Venv already activated: $VIRTUAL_ENV"
fi

echo ""
echo "Step 1: Get your local IP address"
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I | awk '{print $1}')
echo "Local IP: $LOCAL_IP"
echo ""

# Note: We can't run the full test without inference gateway
# But we can document what the command would be

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                        HOW TO RUN FULL TEST                                   ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "STEP 1: In a SEPARATE terminal, start the inference gateway:"
echo "  cd ridges"
echo "  source ../.venv/bin/activate"
echo "  python -m inference_gateway.main"
echo ""
echo "STEP 2: In THIS terminal, once gateway is running, run:"
echo "  cd ridges"
echo "  python test_agent.py --inference-url http://$LOCAL_IP:1234 \\"
echo "    --agent-path ../agents/top_agent/agent.py \\"
echo "    test-problem-set screener-1"
echo ""
echo "This will run:"
echo "  • 5 Polyglot problems"
echo "  • 5 SWE-bench problems"
echo "  • Total: 10 problems"
echo "  • Results saved to: test_agent_results/"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "⏱️  Estimated time: 30-60 minutes"
echo "🎯 What you'll get: REAL pass rate before deployment"
echo ""

