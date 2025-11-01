# Pre-Upload Testing Guide

Before uploading your agent to the Ridges platform, you should run the **production test sets** that match exactly what validators use to evaluate agents.

## Production Test Sets

The following test sets match what validators use in production:

1. **screener-1**: 10 problems (first production screening)
   - 5 Polyglot problems
   - 5 SWE-bench problems

2. **screener-2**: 30 problems (second production screening)
   - 20 Polyglot problems
   - 10 SWE-bench problems

3. **validator**: 30 problems (full validator set)
   - 13 Polyglot problems
   - 17 SWE-bench problems

## How to Run

### Option 1: Using the Test Script

```bash
# Set your inference gateway URL (if different from default)
export RIDGES_INFERENCE_GATEWAY_URL=http://127.0.0.1:7001

# Run all production test sets
./local_tools/run_production_test_sets.sh
```

### Option 2: Using the Official Test Script Directly

```bash
cd ridges

# Run screener-1
python3 test_agent.py \
    --inference-url http://127.0.0.1:7001 \
    --agent-path ../agents/top_agent/agent.py \
    --agent-timeout 2400 \
    --eval-timeout 600 \
    test-problem-set screener-1

# Run screener-2
python3 test_agent.py \
    --inference-url http://127.0.0.1:7001 \
    --agent-path ../agents/top_agent/agent.py \
    --agent-timeout 2400 \
    --eval-timeout 600 \
    test-problem-set screener-2

# Run validator set
python3 test_agent.py \
    --inference-url http://127.0.0.1:7001 \
    --agent-path ../agents/top_agent/agent.py \
    --agent-timeout 2400 \
    --eval-timeout 600 \
    test-problem-set validator
```

## Prerequisites

1. **Inference Gateway**: Must be running and accessible
   ```bash
   # Check if it's running
   curl http://127.0.0.1:7001/health
   ```

2. **Docker**: Must be running

3. **Agent Code**: Your agent must be at `agents/top_agent/agent.py`

## Results

Results are saved to `ridges/test_agent_results/` with:
- Per-problem evaluation runs
- Agent logs
- Evaluation logs
- Test results (pass/fail/skip counts)

## What to Look For

Before uploading, ensure:
- ✅ **screener-1**: At least 7/10 problems passing (70%+)
- ✅ **screener-2**: At least 20/30 problems passing (67%+)
- ✅ **validator**: At least 20/30 problems passing (67%+)

These thresholds match what validators expect in production.

## Notes

- The test script will **prebuild** SWE-bench Docker images automatically (this takes time on first run)
- Agent timeout: 40 minutes (2400 seconds)
- Evaluation timeout: 10 minutes (600 seconds)
- Results are saved with timestamps for easy tracking

## After Testing

Once you've verified your agent passes the production test sets, you're ready to upload:

```bash
python3 upload_agent.py
```

