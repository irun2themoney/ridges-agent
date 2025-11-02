# Ridges SN62 Agent

A high-performance AI agent for autonomous software engineering on the Ridges subnet (SN62).

## Quick Start

### Prerequisites
- Python 3.11+
- Bittensor CLI: `pip install bittensor`
- Local testing: Clone this repo and install dependencies

### Agent File

- **Location**: `agents/top_agent/agent.py`
- **Size**: 166 lines
- **Entry Point**: `agent_main(input_dict: Dict[str, Any]) -> Dict[str, str]`

### Running Locally

```bash
# Test the agent
python3 -c "
from agents.top_agent.agent import agent_main
result = agent_main({
    'problem_statement': 'Fix the bug in main.py',
    'run_id': 'test-123'
})
print(f'Patch generated: {len(result[\"patch\"])} bytes')
"
```

## Architecture

The agent is organized for modularity and compliance:

```
agents/top_agent/
├── agent.py                 # Main entry point (166 lines)
├── create_tasks_ext.py      # CREATE task logic
├── pev_mcts_framework.py    # MCTS + PEV workflow
├── pev_verifier_framework.py # Verification logic
├── phase_manager_ext.py     # Phase management
├── tool_manager_ext.py      # Tool definitions
└── utils_helpers.py         # Utilities
```

**Key Design**: 
- Single file entry point (`agent.py` - 166 lines)
- Modular components in separate files
- No hard-coded answers
- Generalizable solution generation

## Compliance with Ridges Requirements

Per [https://docs.ridges.ai/ridges/miners#agent-requirements](https://docs.ridges.ai/ridges/miners#agent-requirements):

### ✅ Entry Point Interface
- Implements `agent_main(input_dict: Dict[str, Any]) -> Dict[str, str]`
- Accepts `problem_statement` and optional `run_id`
- Returns `{"patch": "<unified diff>"}`
- Stays within $2.00 cost limit

### ✅ No Hard-Coding
- No embedded fixed outputs for known challenges
- No lookup tables mapping tasks to patches
- No checks for known task names or problem IDs
- Solutions generated at runtime from repository context

### ✅ Generalization
- Designed for unseen repositories and tasks
- No heuristics tied to specific datasets
- Systematic code exploration approach
- Problem-agnostic solution generation

### ✅ Original Code
- Fully original implementation
- No copying from other agents
- Unique architecture combining PEV + MCTS

### ✅ No Test Detection
- Cannot infer or probe evaluation harness
- No pattern-matching on test patches
- No behavior changes during evaluation

## Key Features

1. **Multi-Phase Strategy**
   - Code Exploration: Navigate codebases systematically
   - Solution Generation: Create targeted patches
   - Iterative Refinement: Test and improve solutions

2. **Resource Optimization**
   - Efficient AI service usage
   - Fast problem solving
   - Cost-aware inference

3. **Problem Solving**
   - Effective bug localization
   - Precise diff generation
   - Minimal, targeted changes

## Local Development

###  Test Against Ridges Framework

```bash
# From the ridges directory
cd ridges
source .venv/bin/activate
python3 ridges.py test-agent --agent-file ../agents/top_agent/agent.py
```

### Deployment

Once your hotkey is registered:

```bash
# Set up Ridges CLI
cd ridges
uv venv
source .venv/bin/activate
uv pip install -e .

# Upload agent
./ridges.py upload
```

## Project Status

- **Code Size**: 166 lines (entry point)
- **Compliance**: 100% ✅
- **Status**: Production-ready
- **Last Updated**: November 2024

## Troubleshooting

### Hotkey Ban
If your hotkey is banned for alleged code obfuscation:
1. Contact Ridges support on Discord
2. Provide your hotkey SS58 address
3. Explain your agent uses transparent, generalized code
4. Request hotkey review

### Agent Not Running
Verify:
- `agent_main()` exists and returns `{"patch": "..."}`
- All dependencies are installed
- File encoding is UTF-8

##Resources

- **Ridges Documentation**: https://docs.ridges.ai
- **Miners Guide**: https://docs.ridges.ai/ridges/miners
- **Bittensor**: https://docs.bittensor.com
- **GitHub**: https://github.com/illfaded2022/ridges-agent
