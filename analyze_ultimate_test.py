#!/usr/bin/env python3

import json
import os
from pathlib import Path
from collections import defaultdict

# Find the latest test results directory
results_base = Path("ridges/test_agent_results/2025-11-01__agent.py__a0cf5458-1fd6-40a2-914a-bffb18483d8d")

# Collect results
results = {}
passed = []
failed = []

for problem_dir in sorted(results_base.iterdir()):
    if not problem_dir.is_dir():
        continue
    
    eval_file = problem_dir / "evaluation_run.json"
    if not eval_file.exists():
        continue
    
    problem_name = problem_dir.name.split("__")[0]
    
    with open(eval_file) as f:
        data = json.load(f)
    
    status = data.get("status", "unknown")
    error_code = data.get("error_code")
    error_msg = data.get("error_message", "")
    
    # Determine pass/fail
    if status == "passed":
        passed.append(problem_name)
        results[problem_name] = "✅ PASSED"
    else:
        failed.append(problem_name)
        results[problem_name] = f"❌ FAILED - {error_code}: {error_msg[:60]}"

# Print results
print("\n" + "="*70)
print("🧪 ULTIMATE TEST RESULTS - PURE LLM AGENT (NO HARDCODING)")
print("="*70 + "\n")

print(f"📊 SUMMARY:")
print(f"  Total Problems: {len(passed) + len(failed)}")
print(f"  Passed: {len(passed)}")
print(f"  Failed: {len(failed)}")
print(f"  Pass Rate: {100*len(passed)/(len(passed)+len(failed)):.1f}%\n")

print(f"✅ PASSED ({len(passed)}):")
for p in sorted(passed):
    print(f"  • {p}")

print(f"\n❌ FAILED ({len(failed)}):")
for p in sorted(failed):
    error_msg = results[p].replace("❌ FAILED - ", "")
    print(f"  • {p}: {error_msg[:60]}")

# Breakdown by type
polyglot_problems = [
    "affine-cipher", "beer-song", "book-store", "bottle-song", "bowling"
]
swebench_problems = [
    "astropy__astropy-13398", "astropy__astropy-13579", "astropy__astropy-14369",
    "django__django-10554", "django__django-11138"
]

polyglot_passed = [p for p in passed if any(pp in p for pp in polyglot_problems)]
swebench_passed = [p for p in passed if "__" in p]

print(f"\n📈 BREAKDOWN BY TYPE:")
print(f"  Polyglot: {len(polyglot_passed)}/{len(polyglot_problems)}")
print(f"  SWE-bench: {len(swebench_passed)}/{len(swebench_problems)}")

print("\n" + "="*70 + "\n")

