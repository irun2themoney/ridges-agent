# top_agent

Location: `agents/top_agent/agent.py`

What it does:
- Reads the problem statement and `/sandbox/repo` contents
- Calls the inference gateway to synthesize minimal edits (prefers `main.py` for polyglot)
- Writes updates and returns a unified diff to the validator

Tuning:
- Model: change `moonshotai/Kimi-K2-Instruct` in `call_inference`
- Temperature: default `0.1` for determinism; raise to explore
- Context: adjust file size caps and how many files are summarized

Notes:
- Keep responses strictly JSON for file edits inside the agent to avoid parsing issues
- The validator may not include solution diffs; agent should not rely on `/sandbox/solution.diff`
- Ensure returned patch is non-empty when a change is made

References:
- Docs: https://docs.ridges.ai
- Repo: https://github.com/ridgesai/ridges
