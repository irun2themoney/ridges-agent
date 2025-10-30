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
    # Configure a dummy inference gateway URL (we won't use it due to solution.diff fallback)
    inference_gateway_url = "http://127.0.0.1:9999"

    # Instantiate manager and suite
    sandbox_manager = SandboxManager(inference_gateway_url)
    datasets_path = RIDGES / "evaluator" / "datasets" / "polyglot"
    suite: ProblemSuite = PolyglotSuite(datasets_path)

    # Choose a problem (adjust as desired)
    problem_name = "affine-cipher"
    assert suite.has_problem_name(problem_name), f"Problem not found: {problem_name}"
    problem = suite.get_problem(problem_name)

    # Load our agent code
    agent_code = load_text(AGENT_PATH)

    # Initialize sandbox with solution included for local testing
    agent_sandbox = suite.initialize_agent_sandbox(
        sandbox_manager=sandbox_manager,
        problem=problem,
        evaluation_run_id="00000000-0000-0000-0000-000000000000",
        agent_code=agent_code,
        include_solution=True,
    )

    # Run agent to produce patch
    patch, agent_logs = suite.run_agent_sandbox(
        sandbox_manager=sandbox_manager,
        agent_sandbox=agent_sandbox,
        timeout_seconds=180,
    )

    print("\n=== Agent Patch (first 2000 chars) ===\n")
    print(patch[:2000])

    # Initialize eval sandbox with the produced patch
    eval_sandbox = suite.initialize_eval_sandbox(
        sandbox_manager=sandbox_manager,
        problem=problem,
        evaluation_run_id="00000000-0000-0000-0000-000000000000",
        patch=patch,
    )

    # Run tests
    test_results, eval_logs = suite.run_eval_sandbox(
        sandbox_manager=sandbox_manager,
        eval_sandbox=eval_sandbox,
        timeout_seconds=180,
    )

    print("\n=== Test Results ===\n")
    print(json.dumps([t.model_dump() for t in test_results], indent=2))


if __name__ == "__main__":
    main()
