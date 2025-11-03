# 🧪 Official Ridges Testing Guide

**Recommended by Ridges dev**: Use `test-agent` to check performance before uploading.

## Quick Start

### 1. Start Inference Gateway

```bash
cd ridges
source ../.venv/bin/activate
python -m inference_gateway.main
```

Leave this running in a separate terminal.

### 2. Test Your Agent

```bash
cd ridges
python3 test_agent.py \
  --inference-url http://127.0.0.1:1234 \
  --agent-path ../agents/top_agent/agent.py \
  test-problem-set screener-1
```

## Available Problem Sets

- **screener-1**: Quick test (10 problems)
- **screener-2**: Extended test (20 problems)
- **validator**: Validation set (30 problems)
- **all-polyglot**: All polyglot problems (35 problems)

## Commands

### Test a Problem Set

```bash
python3 test_agent.py \
  --inference-url http://127.0.0.1:1234 \
  --agent-path ../agents/top_agent/agent.py \
  test-problem-set <problem-set-name>
```

**Example:**
```bash
python3 test_agent.py \
  --inference-url http://127.0.0.1:1234 \
  --agent-path ../agents/top_agent/agent.py \
  test-problem-set screener-1
```

### Test a Single Problem

```bash
python3 test_agent.py \
  --inference-url http://127.0.0.1:1234 \
  --agent-path ../agents/top_agent/agent.py \
  test-problem <problem-name>
```

**Example:**
```bash
python3 test_agent.py \
  --inference-url http://127.0.0.1:1234 \
  --agent-path ../agents/top_agent/agent.py \
  test-problem affine-cipher
```

## Options

- `--inference-url` (required): Inference gateway URL (default: `http://127.0.0.1:1234`)
- `--agent-path` (required): Path to your agent.py file
- `--agent-timeout`: Agent execution timeout in seconds (default: 2400)
- `--eval-timeout`: Evaluation timeout in seconds (default: 600)
- `--include-solutions`: Include solutions in evaluation (flag)

## Recommended Testing Flow

1. **Quick Test** (screener-1):
   ```bash
   python3 test_agent.py \
     --inference-url http://127.0.0.1:1234 \
     --agent-path ../agents/top_agent/agent.py \
     test-problem-set screener-1
   ```

2. **Extended Test** (screener-2):
   ```bash
   python3 test_agent.py \
     --inference-url http://127.0.0.1:1234 \
     --agent-path ../agents/top_agent/agent.py \
     test-problem-set screener-2
   ```

3. **Full Validation** (validator):
   ```bash
   python3 test_agent.py \
     --inference-url http://127.0.0.1:1234 \
     --agent-path ../agents/top_agent/agent.py \
     test-problem-set validator
   ```

## Results

Results are saved to: `ridges/test_agent_results/`

Each test run creates a directory with:
- Agent logs
- Evaluation results
- Patch output
- Test results

## Tips

- Start with `screener-1` for quick feedback
- Use `screener-2` before significant changes
- Run `validator` before final deployment
- Check pass rate against your target (>55% for bounty)

## Troubleshooting

### Inference Gateway Not Running
```
Error: Connection refused
```
**Solution**: Start inference gateway first (step 1)

### Agent Path Not Found
```
Error: File not found
```
**Solution**: Check path to agent.py (use absolute path if needed)

### Timeout Issues
Increase timeout:
```bash
python3 test_agent.py \
  --inference-url http://127.0.0.1:1234 \
  --agent-path ../agents/top_agent/agent.py \
  --agent-timeout 3600 \
  test-problem-set screener-1
```

---

**Note**: This is the official testing method recommended by Ridges devs. Always test before uploading!
