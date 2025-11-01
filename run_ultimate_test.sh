#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║      ULTIMATE TEST - PURE LLM AGENT (NO HARDCODING)      ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "🧪 Test Strategy:"
echo "  - Running screener-1 test set (10 problems)"
echo "  - Mix of Polyglot and SWE-bench"
echo "  - No hardcoding, pure LLM approach"
echo ""
echo "Timestamp: $(date)"
echo ""

cd /Users/illfaded2022/Desktop/WORKSPACE/ridges-agent/ridges

# Run screener-1
echo "📋 Running screener-1 test set..."
echo ""

python3 test_agent.py \
    --inference-url http://127.0.0.1:7001 \
    --agent-path ../agents/top_agent/agent.py \
    --agent-timeout 60 \
    --eval-timeout 120 \
    test-set screener-1 2>&1 | tee /tmp/ultimate_test.log

echo ""
echo "✅ Test completed!"
echo ""
echo "📊 Results saved to:"
echo "  - Full log: /tmp/ultimate_test.log"
echo "  - Test results: $(ls -td test_agent_results/2025-11-01* | head -1)"
