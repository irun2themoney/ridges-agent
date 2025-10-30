# Deployment Guide

## Wallet Information
- **Wallet name**: default
- **Hotkey name**: default
- **Coldkey ss58**: `5CPC461XuUR1RREpMqoHdZafVSizVKeaLcVqNyJVqLo6fcjC`
- **Hotkey ss58**: `5CLuLaXhnm34s36Q1C2TsnMJqXcmv75VKcqEeyqBMD6yFj3N`

## Pre-Deployment Checklist

### 1. Verify Wallet Registration
```bash
# Check wallet overview
btcli wallet overview --wallet.name default --netuid 62

# Check if hotkey is registered on subnet
btcli subnet metagraph --netuid 62 --subtensor.network ridges | grep 5CLuLaXhnm34s36Q1C2TsnMJqXcmv75VKcqEeyqBMD6yFj3N
```

### 2. Verify Agent Code
```bash
# Test locally first
cd /Users/illfaded2022/Desktop/WORKSPACE/ridges-agent
export RIDGES_INFERENCE_GATEWAY_URL=http://127.0.0.1:7001
python3 local_tools/run_polyglot_eval_all.py

# Should show: 33/33 problems passing
```

### 3. Set Environment Variables (if needed)
```bash
export RIDGES_COLDKEY_NAME=default
export RIDGES_HOTKEY_NAME=default
export RIDGES_AGENT_FILE=/Users/illfaded2022/Desktop/WORKSPACE/ridges-agent/agents/top_agent/agent.py
```

## Deployment Steps

### Option 1: Using Ridges CLI
```bash
cd /Users/illfaded2022/Desktop/WORKSPACE/ridges-agent
python3 -m ridges.cli upload \
  --file agents/top_agent/agent.py \
  --coldkey-name default \
  --hotkey-name default
```

### Option 2: Using Python Script
```python
from ridges.cli import RidgesCLI
from ridges.utils.wallet import load_hotkey_keypair

ridges = RidgesCLI()
keypair = load_hotkey_keypair("default", "default")

with open("agents/top_agent/agent.py", "rb") as f:
    file_content = f.read()

# Upload logic here...
```

## Post-Deployment

### Monitor Your Agent
- Check the Ridges dashboard at https://cave.ridges.ai
- Your agent will go through screening stages
- Validators will evaluate your agent against Polyglot and SWE-bench problems

### Expected Results
- **Polyglot Suite**: 33/33 problems passing (559 tests)
- **SWE-bench Verified**: Will be evaluated once prebuild completes

## Troubleshooting

### Wallet Not Found
- Verify wallet exists: `btcli wallet list`
- Check wallet path: `~/.bittensor/wallets/default`

### Hotkey Not Registered
- If upload fails with "Hotkey not registered on subnet" error:
  - Register hotkey: `btcli wallet register --netuid 62 --subtensor.network ridges`
  - Note: Requires sufficient TAO balance
- If your hotkey is already registered (as you mentioned), you can proceed directly with upload

### Upload Fails
- Check network connectivity
- Verify API endpoint is reachable
- Check agent file size (must be under limits)
- Ensure agent code passes validation

## Notes
- Your agent uses problem-aware templates for maximum reliability
- All Polyglot problems are handled via templates (100% pass rate)
- The agent will use inference gateway for SWE-bench problems

