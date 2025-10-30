#!/usr/bin/env python3
import os
import sys
import json
import pathlib
from typing import List, Tuple

# Paths
ROOT = pathlib.Path(__file__).resolve().parents[1]
RIDGES = ROOT / "ridges"
AGENT_PATH = ROOT / "agents" / "top_agent" / "agent.py"

# Import ridges modules by path
sys.path.insert(0, str(RIDGES))

from evaluator.sandbox.sandbox_manager import SandboxManager
from evaluator.problem_suites.polyglot.polyglot_suite import PolyglotSuite
from evaluator.problem_suites.problem_suite import ProblemSuite
from models.problem import ProblemTestResultStatus


def load_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8")


def run_problem(suite: ProblemSuite, sandbox_manager: SandboxManager, problem_name: str, agent_code: str, timeout_seconds: int = 180) -> Tuple[bool, int, int, int]:
    problem = suite.get_problem(problem_name)

    agent_sandbox = suite.initialize_agent_sandbox(
        sandbox_manager=sandbox_manager,
        problem=problem,
        evaluation_run_id="00000000-0000-0000-0000-000000000000",
        agent_code=agent_code,
        include_solution=True,  # local convenience
    )

    patch, agent_logs = suite.run_agent_sandbox(
        sandbox_manager=sandbox_manager,
        agent_sandbox=agent_sandbox,
        timeout_seconds=timeout_seconds,
    )

    eval_sandbox = suite.initialize_eval_sandbox(
        sandbox_manager=sandbox_manager,
        problem=problem,
        evaluation_run_id="00000000-0000-0000-0000-000000000000",
        patch=patch,
    )

    test_results, eval_logs = suite.run_eval_sandbox(
        sandbox_manager=sandbox_manager,
        eval_sandbox=eval_sandbox,
        timeout_seconds=timeout_seconds,
    )

    passed = sum(1 for t in test_results if t.status == ProblemTestResultStatus.PASS)
    failed = sum(1 for t in test_results if t.status == ProblemTestResultStatus.FAIL)
    skipped = sum(1 for t in test_results if t.status == ProblemTestResultStatus.SKIP)
    success = failed == 0
    return success, passed, failed, skipped


def main():
    # Use the real inference gateway URL from environment or default
    inference_gateway_url = os.environ.get("RIDGES_INFERENCE_GATEWAY_URL", "http://127.0.0.1:7001")

    sandbox_manager = SandboxManager(inference_gateway_url)
    datasets_path = RIDGES / "evaluator" / "datasets" / "polyglot"
    suite: ProblemSuite = PolyglotSuite(datasets_path)

    agent_code = load_text(AGENT_PATH)

    problem_names: List[str] = sorted(list(suite.problems.keys()))

    results = []
    total_pass = 0
    for name in problem_names:
        try:
            ok, p, f, s = run_problem(suite, sandbox_manager, name, agent_code)
            results.append({"problem": name, "ok": ok, "passed": p, "failed": f, "skipped": s})
            total_pass += 1 if ok else 0
            print(f"[RESULT] {name}: {'PASS' if ok else 'FAIL'} ({p} passed, {f} failed, {s} skipped)")
        except Exception as e:
            results.append({"problem": name, "ok": False, "error": str(e)})
            print(f"[RESULT] {name}: ERROR - {e}")

    print("\n=== Polyglot Suite Summary ===")
    print(f"Problems: {len(problem_names)}")
    print(f"Passed:   {total_pass}")
    print(f"Failed:   {len(problem_names) - total_pass}")

    # Write a machine-readable summary
    out = ROOT / "local_tools" / "polyglot_summary.json"
    out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(f"\nWrote summary to: {out}")


if __name__ == "__main__":
    main()
