#!/usr/bin/env python3
import os
import sys
import json
import pathlib

# Paths
ROOT = pathlib.Path(__file__).resolve().parents[1]
RIDGES = ROOT / "ridges"
AGENT_PATH = ROOT / "agents" / "top_agent" / "agent.py"

# Import ridges modules by path
sys.path.insert(0, str(RIDGES))

from evaluator.sandbox.sandbox_manager import SandboxManager
from evaluator.problem_suites.polyglot.polyglot_suite import PolyglotSuite
from evaluator.problem_suites.problem_suite import ProblemSuite


def load_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8")


def main():
    gateway = os.environ.get("RIDGES_INFERENCE_GATEWAY_URL")
    if not gateway:
        print("RIDGES_INFERENCE_GATEWAY_URL not set. Export it to point to your running inference gateway.")
        sys.exit(1)

    sandbox_manager = SandboxManager(gateway)
    datasets_path = RIDGES / "evaluator" / "datasets" / "polyglot"
    suite: ProblemSuite = PolyglotSuite(datasets_path)

    problem_name = os.environ.get("POLYGLOT_PROBLEM", "affine-cipher")
    if not suite.has_problem_name(problem_name):
        print(f"Unknown problem: {problem_name}")
        sys.exit(1)

    problem = suite.get_problem(problem_name)
    agent_code = load_text(AGENT_PATH)

    # No solutions injected
    agent_sandbox = suite.initialize_agent_sandbox(
        sandbox_manager=sandbox_manager,
        problem=problem,
        evaluation_run_id="00000000-0000-0000-0000-000000000000",
        agent_code=agent_code,
        include_solution=False,
    )

    patch, agent_logs = suite.run_agent_sandbox(
        sandbox_manager=sandbox_manager,
        agent_sandbox=agent_sandbox,
        timeout_seconds=240,
    )

    print("\n=== Agent Logs (tail) ===\n")
    print("\n".join(agent_logs.splitlines()[-200:]))

    print("\n=== Agent Patch (first 2000 chars) ===\n")
    print((patch or "")[:2000])

    eval_sandbox = suite.initialize_eval_sandbox(
        sandbox_manager=sandbox_manager,
        problem=problem,
        evaluation_run_id="00000000-0000-0000-0000-000000000000",
        patch=patch,
    )

    test_results, eval_logs = suite.run_eval_sandbox(
        sandbox_manager=sandbox_manager,
        eval_sandbox=eval_sandbox,
        timeout_seconds=240,
    )

    print("\n=== Test Results ===\n")
    print(json.dumps([t.model_dump() for t in test_results], indent=2))


if __name__ == "__main__":
    main()
