# Environment setup

Create two .env files based on these templates.

## Inference Gateway (.env in ridges/)

HOST=127.0.0.1
PORT=7001

USE_DATABASE=false

# Choose one provider and set creds
USE_CHUTES=true
CHUTES_BASE_URL=<required>
CHUTES_API_KEY=<required>
CHUTES_WEIGHT=100

USE_TARGON=false
# TARGON_BASE_URL=<optional>
# TARGON_API_KEY=<optional>
# TARGON_WEIGHT=100

# Limits
MAX_INFERENCE_REQUESTS_PER_EVALUATION_RUN=200

## Validator (.env in ridges/)

MODE=validator

# Bittensor
NETUID=<required int>
SUBTENSOR_ADDRESS=<required>
SUBTENSOR_NETWORK=<required>

# Wallet names in your local keystore
VALIDATOR_WALLET_NAME=<required>
VALIDATOR_HOTKEY_NAME=<required>

# Platform + Gateway
RIDGES_PLATFORM_URL=<platform url>
RIDGES_INFERENCE_GATEWAY_URL=http://127.0.0.1:7001

# Timers
SEND_HEARTBEAT_INTERVAL_SECONDS=20
SET_WEIGHTS_INTERVAL_SECONDS=300
REQUEST_EVALUATION_INTERVAL_SECONDS=20

# Behavior toggles
SIMULATE_EVALUATION_RUNS=false
INCLUDE_SOLUTIONS=false
UPDATE_AUTOMATICALLY=true
