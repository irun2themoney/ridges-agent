# Testing Guide - How to Test Your Agent

This guide covers all the ways to test your agent and verify it works correctly.

## Quick Start

### 1. Verify Inference Gateway is Running

```bash
curl http://127.0.0.1:7001/health
```

Should return `{"status": "ok"}` or similar. If not, start the inference gateway first.

### 2. Run a Quick Single Problem Test

Test a single problem to verify everything works:

```bash
cd ridges
python3 test_agent.py \
    --inference-url http://127.0.0.1:7001 \
    --agent-path ../agents/top_agent/agent.py \
    --agent-timeout 2400 \
    --eval-timeout 600 \
    test-problem affine-cipher
```

## Testing Methods

### Method 1: Single Problem Test (Quick Verification)

Test one problem at a time to debug specific issues:

```bash
cd ridges

# Test a Polyglot problem
python3 test_agent.py \
    --inference-url http://127.0.0.1:7001 \
    --agent-path ../agents/top_agent/agent.py \
    --agent-timeout 2400 \
    --eval-timeout 600 \
    test-problem bottle-song

# Test a SWE-bench problem
python3 test_agent.py \
    --inference-url http://127.0.0.1:7001 \
    --agent-path ../agents/top_agent/agent.py \
    --agent-timeout 2400 \
    --eval-timeout 600 \
    test-problem django__django-14011
```

### Method 2: Production Test Sets (Recommended Before Upload)

These match exactly what validators use in production:

```bash
# Run all production test sets (screener-1, screener-2, validator)
./local_tools/run_production_test_sets.sh

# Or run individually:
cd ridges

# screener-1: 10 problems (5 Polyglot + 5 SWE-bench)
python3 test_agent.py \
    --inference-url http://127.0.0.1:7001 \
    --agent-path ../agents/top_agent/agent.py \
    --agent-timeout 2400 \
    --eval-timeout 600 \
    test-problem-set screener-1

# screener-2: 30 problems (20 Polyglot + 10 SWE-bench)
python3 test_agent.py \
    --inference-url http://127.0.0.1:7001 \
    --agent-path ../agents/top_agent/agent.py \
    --agent-timeout 2400 \
    --eval-timeout 600 \
    test-problem-set screener-2

# validator: 30 problems (13 Polyglot + 17 SWE-bench)
python3 test_agent.py \
    --inference-url http://127.0.0.1:7001 \
    --agent-path ../agents/top_agent/agent.py \
    --agent-timeout 2400 \
    --eval-timeout 600 \
    test-problem-set validator
```

### Method 3: Full Polyglot Suite (33 problems)

Test all Polyglot problems at once:

```bash
cd ridges
python3 test_agent.py \
    --inference-url http://127.0.0.1:7001 \
    --agent-path ../agents/top_agent/agent.py \
    --agent-timeout 2400 \
    --eval-timeout 600 \
    test-problem-set all-polyglot
```

## Understanding Test Results

### Where Results Are Saved

Results are saved to: `ridges/test_agent_results/YYYY-MM-DD__agent.py__<uuid>/`

Each problem has its own directory with:
- `agent_logs.txt` - What your agent did
- `eval_logs.txt` - Test execution logs
- `evaluation_run.json` - Full results (pass/fail counts, errors)

### Reading Results

#### From Console Output

Look for lines like:
```
[affine-cipher] Finished running evaluation: 16 passed, 0 failed, 0 skipped
```

- **passed**: Tests that passed ✅
- **failed**: Tests that failed ❌
- **skipped**: Tests that were skipped (usually because earlier tests failed)

#### From JSON Files

```bash
# Check a specific problem's results
cat ridges/test_agent_results/2025-10-31__agent.py__*/affine-cipher__*/evaluation_run.json | jq '.test_results[] | select(.status == "fail")'
```

#### Summary Statistics

After running a test set, you'll see:
```
Test Summary:
- Total problems: 10
- Passed: 8
- Failed: 2
- Pass rate: 80%
```

## Success Criteria

### Before Uploading to Production

**Minimum thresholds** (recommended):
- ✅ **screener-1**: At least 7/10 passing (70%+)
- ✅ **screener-2**: At least 20/30 passing (67%+)
- ✅ **validator**: At least 20/30 passing (67%+)

**Good performance**:
- ✅ **screener-1**: 9-10/10 passing (90%+)
- ✅ **screener-2**: 25-30/30 passing (83%+)
- ✅ **validator**: 25-30/30 passing (83%+)

### Interpreting Individual Problem Results

#### For Polyglot Problems

- **All tests passed**: Problem solved correctly ✅
- **Some tests passed**: Partial solution, needs improvement
- **All tests failed**: Solution incorrect, needs debugging
- **Exception/Error**: Agent crashed or returned invalid patch

#### For SWE-bench Problems

- **Tests passed**: Patch applied successfully and tests pass ✅
- **Tests failed**: Patch applied but doesn't fix the issue
- **Error applying patch**: Invalid diff format
- **Exception/Error**: Agent crashed

## Debugging Failed Tests

### Step 1: Check Agent Logs

```bash
cat ridges/test_agent_results/2025-10-31__agent.py__*/<problem>__*/agent_logs.txt
```

Look for:
- Did the agent generate a patch?
- What did the agent try to do?
- Any error messages?

### Step 2: Check Evaluation Logs

```bash
cat ridges/test_agent_results/2025-10-31__agent.py__*/<problem>__*/eval_logs.txt
```

Look for:
- Which tests failed?
- What was the expected vs actual output?
- Any syntax errors?

### Step 3: Check the Generated Patch

```bash
cat ridges/test_agent_results/2025-10-31__agent.py__*/<problem>__*/evaluation_run.json | jq '.patch'
```

Look for:
- Is the patch valid?
- Does it target the right files?
- Are the changes correct?

### Step 4: Test Locally

For Polyglot problems, you can test manually:
```bash
cd ridges/evaluator/datasets/polyglot/<problem>
# Check main.py and tests.py
# Manually apply your agent's patch
# Run the tests
```

## Running Tests in the Background

For long-running tests, run them in the background:

```bash
# Start test in background
nohup python3 ridges/test_agent.py \
    --inference-url http://127.0.0.1:7001 \
    --agent-path agents/top_agent/agent.py \
    --agent-timeout 2400 \
    --eval-timeout 600 \
    test-problem-set validator > /tmp/validator_test.log 2>&1 &

# Monitor progress
tail -f /tmp/validator_test.log

# Check if still running
ps aux | grep test_agent.py
```

## Common Issues and Solutions

### 1. "Inference gateway URL is invalid"

**Solution**: Make sure inference gateway is running:
```bash
# Check if running
curl http://127.0.0.1:7001/health

# Start if needed (in another terminal)
cd ridges
python3 -m inference_gateway.main
```

### 2. "Docker not available"

**Solution**: Start Docker Desktop or Docker daemon

### 3. "No valid patches in input"

**Solution**: Your agent returned an empty or invalid patch. Check:
- Agent logs for errors
- Whether the agent actually generated a patch
- If SWE-bench, check if file paths are correct

### 4. Tests timeout

**Solution**: Increase timeouts:
```bash
--agent-timeout 3600  # 60 minutes
--eval-timeout 1200   # 20 minutes
```

### 5. SWE-bench Docker build fails

**Solution**: Prebuild images first:
```bash
python3 local_tools/prebuild_swebench.py
```

## Quick Test Checklist

Before uploading your agent:

- [ ] Inference gateway is running
- [ ] Docker is running
- [ ] Ran at least one single problem test successfully
- [ ] Ran screener-1 and got 7+/10 passing
- [ ] Ran screener-2 and got 20+/30 passing
- [ ] Ran validator set and got 20+/30 passing
- [ ] Reviewed logs for any errors or warnings
- [ ] Agent code is committed and pushed to GitHub

## Next Steps

Once your tests pass:

1. **Commit your changes**:
   ```bash
   git add agents/top_agent/agent.py
   git commit -m "Your improvements"
   git push
   ```

2. **Upload to Ridges**:
   ```bash
   python3 upload_agent.py
   ```

3. **Monitor on Ridges Dashboard**:
   - Check your agent's performance at: https://www.ridges.ai/agent/<your-hotkey>
   - Review evaluation runs
   - Monitor pass rates

## Need Help?

- Check the logs first (agent_logs.txt, eval_logs.txt)
- Review the test output for specific error messages
- Test individual problems to isolate issues
- Verify templates are triggering correctly for known problems

