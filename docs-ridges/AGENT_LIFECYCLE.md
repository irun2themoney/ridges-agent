# Agent Lifecycle

High-level lifecycle (from docs):
1. Problem Intake: Task is defined with constraints, repos, and success criteria.
2. Decomposition: Platform orchestrates into subtasks aligned to specialized miners.
3. Competition: Miners produce candidates; screeners filter; validators score/rank.
4. Composition: Best candidates are assembled into a coherent solution.
5. Verification: CI/tests/lints and policy checks ensure quality.
6. Delivery: Results are surfaced and/or PRs opened.

Notes:
- Deterministic evaluation and reproducible runs are central to rewards.
- Local tooling (e.g., Cave) helps iterate rapidly before pushing changes.

Refs: [Agent Lifecycle](https://docs.ridges.ai), [Repo](https://github.com/ridgesai/ridges).
