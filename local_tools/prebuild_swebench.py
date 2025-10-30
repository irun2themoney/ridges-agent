#!/usr/bin/env python3
import sys
import pathlib

# Paths
ROOT = pathlib.Path(__file__).resolve().parents[1]
RIDGES = ROOT / "ridges"

# Import ridges modules by path
sys.path.insert(0, str(RIDGES))

from evaluator.problem_suites.swebench_verified.swebench_verified_suite import SWEBenchVerifiedSuite


def main():
    datasets_path = RIDGES / "evaluator" / "datasets" / "swebench_verified"
    suite = SWEBenchVerifiedSuite(datasets_path)

    # Prebuild all images; this can take a long time and a lot of disk
    suite.prebuild_problem_images()  # no filter -> all


if __name__ == "__main__":
    main()
