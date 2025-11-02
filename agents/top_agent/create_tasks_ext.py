import ast
import csv
import json
import os
import random
import re
import subprocess
import sys
import textwrap
import time
import traceback
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Import from agent
from agent import EnhancedNetwork, VariableNormalizer

def validate_edge_case_comments(solution: str) -> dict:
    """
    Enhanced validation that detects edge case comments with better pattern matching.
    Returns a dictionary with validation results.
    """
    validation_result = {
        "has_edge_case_comments": False,
        "edge_cases_identified": [],
        "missing_edge_cases": [],
        "comment_quality": "poor",
        "coverage_score": 0.0,
        "pattern_matches": {
            "edge_case_headers": [],
            "handled_cases_summary": [],
            "inline_comments": [],
            "test_related": []
        }
    }
    patterns = {
        "edge_case_headers": [
            r"#\s*Edge\s*Case:\s*(.+)",
            r"#\s*EDGE\s*CASE:\s*(.+)",
            r"#\s*Edge\s*case:\s*(.+)",
            r"#\s*TODO:\s*handle\s*edge\s*case[:\s]*(.+)",
            r"#\s*Handle\s+edge\s+case:\s*(.+)"
        ],
        "handled_cases_summary": [
            r"#\s*Handled\s*Edge\s*Cases?:\s*(.+)",
            r"#\s*Edge\s*cases\s*handled:\s*(.+)",
            r"#\s*Covered\s*edge\s*cases:\s*(.+)",
            r"#\s*Edge\s*cases:\s*(.+)"
        ],
        "inline_comments": [
            r"#\s*(?:null|empty|zero|negative|invalid|boundary|limit)\s*case",
            r"#\s*(?:handle|check|validate).*(?:edge|corner|boundary)",
            r"#\s*Special\s+case:"
        ],
        "test_related": [
            r"#\s*Test\s+case.*edge",
            r"#\s*Edge\s+case\s+test"
        ]
    }
    for category, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, solution, re.IGNORECASE | re.MULTILINE)
            validation_result["pattern_matches"][category].extend(matches)
    total_patterns_found = sum(len(matches) for matches in validation_result["pattern_matches"].values())
    validation_result["coverage_score"] = min(total_patterns_found / 10.0, 1.0)  # Normalize to 0-1
    edge_cases = validation_result["pattern_matches"]["edge_case_headers"]
    handled_summary = validation_result["pattern_matches"]["handled_cases_summary"]

    if edge_cases and handled_summary:
        validation_result["has_edge_case_comments"] = True
        validation_result["edge_cases_identified"] = edge_cases + handled_summary

        if len(edge_cases) >= 3 and len(handled_summary) >= 1:
            validation_result["comment_quality"] = "excellent"
        elif len(edge_cases) >= 2:
            validation_result["comment_quality"] = "good"
        elif len(edge_cases) >= 1:
            validation_result["comment_quality"] = "fair"

    return validation_result

def analyze_missing_edge_cases(solution: str, problem_statement: str) -> dict:
    """
    Analyze the solution to identify potentially missing edge cases.
    """
    analysis = {
        "potential_missing_cases": [],
        "code_complexity_indicators": [],
        "risk_assessment": "low"
    }
    edge_case_indicators = {
        "null_checks": ["if.*is None", "if.*== None", "if.*!= None"],
        "empty_checks": ["if.*== \"\"", "if.*!= \"\"", "if.*len\\(.*\\) == 0"],
        "boundary_checks": ["if.*== 0", "if.*< 0", "if.*> 0", "if.*<= 0", "if.*>= 0"],
        "type_checks": ["isinstance", "type\\(", "str\\(", "int\\(", "float\\("],
        "exception_handling": ["try:", "except:", "raise", "finally:"]
    }

    found_indicators = {}
    for category, patterns in edge_case_indicators.items():
        found_indicators[category] = []
        for pattern in patterns:
            matches = re.findall(pattern, solution, re.IGNORECASE)
            found_indicators[category].extend(matches)
    if not found_indicators["null_checks"]:
        analysis["potential_missing_cases"].append("null/None value handling")
    if not found_indicators["empty_checks"]:
        analysis["potential_missing_cases"].append("empty string/list handling")
    if not found_indicators["boundary_checks"]:
        analysis["potential_missing_cases"].append("boundary value validation")
    total_indicators = sum(len(matches) for matches in found_indicators.values())
    if total_indicators < 3:
        analysis["risk_assessment"] = "high"
    elif total_indicators < 6:
        analysis["risk_assessment"] = "medium"

    return analysis

def generate_initial_solution(problem_statement: str, code_skeleton: str, detailed_problem_analysis: dict) -> str:

    problem_statement_with_spec = problem_statement
    spec = detailed_problem_analysis if isinstance(detailed_problem_analysis, str) else json.dumps(
        detailed_problem_analysis,
        indent=4
    )
    problem_statement_with_spec += (
        "\nImplement the functions referencing detailed problem specifications.\n"
        f"Analysis:\n{spec}\n"
    )
    models = determine_model_order(problem_statement_with_spec)

    temperature = determine_temperature(problem_statement_with_spec)
    solutions = []
    retry = 0

    while len(solutions) < 3 and retry < 10:
        try:
            solution = generate_solution_with_multi_step_reasoning(problem_statement, code_skeleton, temperature)

            if solution:
                solutions.append(solution)
            else:
                messages = [
                    {
                        "role": "system",
                        "content": GENERATE_INITIAL_SOLUTION_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\n\nGenerate the complete and correct implementation in python files."""
                    }
                ]

                response = EnhancedNetwork.make_request(messages, model=models[0], temperature=temperature)
                solution = response.strip()
                if solution.startswith('```python'):
                    solution = solution[9:]
                if solution.startswith('```'):
                    solution = solution[3:]
                if solution.endswith('```'):
                    solution = solution[:-3]
                solution = solution.strip()

                if solution:
                    solutions.append(solution)

        except Exception as e:
            retry += 1
            time.sleep(2)

    if not solutions:
        return ""
    if len(solutions) == 1:
        return solutions[0]
    solution_validations = []
    solution_analyses = []  # Add this line

    for i, solution in enumerate(solutions):
        validation = validate_edge_case_comments(solution)
        analysis = analyze_missing_edge_cases(solution, problem_statement)  # Add this line

        solution_validations.append(validation)
        solution_analyses.append(analysis)  # Add this line
    comparison_prompt = f"""You are an expert Python developer tasked with evaluating and selecting the best solution from multiple options.

    Problem Statement:
    {problem_statement}

    Code Skeleton:
    {code_skeleton}

    Below are {len(solutions)} different solutions to this problem. Please analyze each solution and select the best one based on:
    1. Correctness and completeness based on problem statement
    2. Code quality and readability
    3. Adherence to Python best practices
    4. Consistency in types, prototypes of functions, etc
    5. Security or Limitation Handling of the algorithms or imported modules
    6. Proper handling of edge cases
    7. Presence and quality of edge case comments - solutions should have clear comments identifying each edge case being handled
    8. Completeness of edge case coverage - verify that all critical edge cases from the problem statement are addressed and properly commented
    9. Missing edge case analysis - consider potential edge cases that might not be handled

    Edge Case Analysis:
    """
    for i, (solution, validation, analysis) in enumerate(zip(solutions, solution_validations, solution_analyses), 1):
        comparison_prompt += f"""Solution {i}: 
        - Has edge case comments: {validation['has_edge_case_comments']}
        - Comment quality: {validation['comment_quality']}
        - Coverage score: {validation['coverage_score']:.2f}
        - Risk assessment: {analysis['risk_assessment']}
        - Potential missing cases: {', '.join(analysis['potential_missing_cases']) if analysis['potential_missing_cases'] else 'None identified'}
    """

    comparison_prompt += "\nSolutions:\n\n"
    for i, solution in enumerate(solutions, 1):
        comparison_prompt += f"=== SOLUTION {i} ===\n{solution}\n\n"

    try:
        comparison_messages = [
            {
                "role": "system",
                "content": "You are an expert Python developer who excels at code review and solution evaluation."
            },
            {
                "role": "user",
                "content": comparison_prompt
            }
        ]

        response = EnhancedNetwork.make_request(comparison_messages, model=models[0])
        best_solution_match = re.search(r'BEST_SOLUTION:\s*(\d+)', response)
        if best_solution_match:
            selected_index = int(best_solution_match.group(1)) - 1  # Convert to 0-based index
            if 0 <= selected_index < len(solutions):
                return solutions[selected_index]
            else:
                return solutions[0]
        else:
            return solutions[0]

    except Exception as e:
        return solutions[0]

def determine_model_order(problem_statement: str) -> list:
    """Determine model priority via LLM routing based on the problem statement.

    The router LLM must return strict JSON indicating the first and second models.
    Falls back to a safe default if parsing fails.
    """
    try:
        system_prompt = (
            "You are a model router. Choose the best first LLM to solve a Python\n"
            "coding challenge given its problem statement, and then the second LLM.\n"
            "Only consider these options (use exact identifiers):\n"
            f"1) {DEEPSEEK_MODEL_NAME} (stronger reasoning, graphs/backtracking/parsers)\n"
            f"2) {QWEN_MODEL_NAME} (stronger implementation, string/data wrangling/spec-following)\n\n"
            "Output MUST be a single JSON object with key 'order' mapping to a list of two\n"
            "strings, the exact model identifiers, best-first. No explanations."
        )

        user_prompt = (
            "Problem statement to route:\n\n" + (problem_statement or "").strip()
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = EnhancedNetwork.make_request(messages, model=DEEPSEEK_MODEL_NAME)
        cleaned = raw.strip()
        cleaned = cleaned.replace('```json', '```')
        if cleaned.startswith('```') and cleaned.endswith('```'):
            cleaned = cleaned.strip('`').strip()

        try:
            data = json.loads(cleaned)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            data = json.loads(match.group(0)) if match else {}

        order = []
        if isinstance(data, dict):
            if isinstance(data.get('order'), list):
                order = data['order']
            elif 'first' in data and 'second' in data:
                order = [data['first'], data['second']]

        alias_map = {
            DEEPSEEK_MODEL_NAME.lower(): DEEPSEEK_MODEL_NAME,
            QWEN_MODEL_NAME.lower(): QWEN_MODEL_NAME,
            'deepseek': DEEPSEEK_MODEL_NAME,
            'qwen': QWEN_MODEL_NAME,
        }

        mapped = []
        for item in order:
            if not isinstance(item, str):
                continue
            key = item.strip().lower()
            if key in alias_map and alias_map[key] not in mapped:
                mapped.append(alias_map[key])

        for candidate in [DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME]:
            if candidate not in mapped:
                mapped.append(candidate)
            if len(mapped) == 2:
                break
        return mapped[:2]
    except Exception as e:
        return [QWEN_MODEL_NAME, DEEPSEEK_MODEL_NAME]

def determine_temperature(problem_statement: str) -> float:
    def validate_response(response: dict) -> tuple[bool, str]:
        if "temperature" not in response:
            return False, "Required key temperature not found in response"

        temperature = response.get("temperature")

        if temperature is None or not isinstance(temperature, float):
            return False, "Required key temperature not found in response"
        return True, ""

    messages = [
        {"role": "system", "content": TEMPERATURE_DETERMINATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem statement: {problem_statement}"}
    ]
    temperature = 0
    while True:
        retry = 0

        while retry < 3:
            try:
                response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)
                response = response.replace('```json', '').strip('```').strip()
                response = json.loads(response)

                is_valid, error_msg = validate_response(response)
                if is_valid:
                    return response.get("temperature", 0.0)
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {"role": "user", "content": "Keep clarifying the temperature until you have a valid float."}
                )

            except Exception as e:
                pass

            retry += 1

        if retry >= 3:
            break

    if not response.get("temperature", 0):
        try:
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)
            response = response.replace('```json', '').strip('```').strip()
            response = json.loads(response)

            is_valid, error_msg = validate_response(response)
            if is_valid:
                return response.get("temperature", 0.0)
            else:
                return 0

        except Exception as e:
            return 0

    return 0

def generate_solution_with_multi_step_reasoning(problem_statement: str, code_skeleton: str, temperature: float) -> str:
    retry = 0
    code_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\nGenerate the complete and correct implementation in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"
        }
    ]

    while retry < 10:
        try:
            code_response = EnhancedNetwork.make_request(
                code_generation_messages,
                model=QWEN_MODEL_NAME,
                temperature=temperature
            )

            loop_check_messages = [
                {
                    "role": "system",
                    "content": INFINITE_LOOP_CHECK_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Generated Code:\n{code_response}\n\nAnalyze this code for potential infinite loops and provide a corrected version if any issues are found. Return ONLY the final Python code."
                }
            ]

            loop_check_response = EnhancedNetwork.make_request(loop_check_messages, model=QWEN_MODEL_NAME)
            solution = loop_check_response.strip()
            if solution.startswith('```python'):
                solution = solution[9:]
            if solution.startswith('```'):
                solution = solution[3:]
            if solution.endswith('```'):
                solution = solution[:-3]
            solution = solution.strip()

            lines = solution.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                code_generation_messages.append({"role": "assistant", "content": code_response})
                code_generation_messages.append(
                    {"role": "user",
                     "content": f"Include file name in the response. example:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```"}
                )
                continue
            return solution
        except Exception as e:
            retry += 1
            time.sleep(2)

    if retry >= 10:
        return ""

    return ""

def generate_testcases_with_multi_step_reasoning(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    from collections import Counter
    import re

    def extract_function_names(testcode: str) -> set:
        """Extract function names from test code to create a signature for comparison"""
        function_names = set()
        test_function_patterns = [
            r'def\s+(test_\w+)',  # def test_something
            r'def\s+(test\w+)',  # def testSomething
            r'def\s+(\w*test\w*)',  # any function containing 'test'
        ]

        for pattern in test_function_patterns:
            matches = re.findall(pattern, testcode, re.IGNORECASE)
            function_names.update(matches)

        return function_names

    def clean_testcode_response(response: str) -> str:
        """Helper function to clean AI response from markdown formatting"""
        testcases = response.strip()
        if testcases.startswith('```python'):
            testcases = testcases[9:]
        if testcases.startswith('```'):
            testcases = testcases[3:]
        if testcases.endswith('```'):
            testcases = testcases[:-3]
        return testcases.strip()

    def generate_single_testset() -> tuple[str, set]:
        """Generate a single test set and return (testcode, function_names)"""
        retry = 0
        test_generation_messages = [
            {
                "role": "system",
                "content": GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT
            },
            {
                "role": "user",
                "content": f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"
            }
        ]

        while retry < 10:
            try:
                testcode_response = EnhancedNetwork.make_request(test_generation_messages, model=QWEN_MODEL_NAME)

                testcases_check_messages = [
                    {
                        "role": "system",
                        "content": TESTCASES_CHECK_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"Problem statement: {problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerated Test Code:\n{testcode_response}\n\nAnalyze this code for invalid testcases. Return ONLY the final Python test code."
                    }
                ]

                testcode_checked_response = EnhancedNetwork.make_request(
                    testcases_check_messages,
                    model=QWEN_MODEL_NAME
                )

                testcases = clean_testcode_response(testcode_checked_response)

                lines = testcases.split("\n")
                if lines[0].endswith(".py") == False:
                    retry += 1
                    test_generation_messages.append({"role": "assistant", "content": testcode_checked_response})
                    test_generation_messages.append(
                        {"role": "user",
                         "content": f"Include file name in the response. example:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"}
                    )
                    continue
                function_names = extract_function_names(testcases)
                return testcases, function_names

            except Exception as e:
                retry += 1
                time.sleep(2)

        return "", set()

    NUM_GENERATIONS = 15
    test_sets = []
    function_signatures = []

    for i in range(NUM_GENERATIONS):
        testcode, function_names = generate_single_testset()

        if testcode and function_names:  # Only add valid test sets
            test_sets.append(testcode)
            function_signatures.append(tuple(sorted(function_names)))  # Use tuple for hashing

    if not test_sets:
        return ""

    signature_counts = Counter(function_signatures)
    most_common_signature = signature_counts.most_common(1)[0][0]
    most_common_count = signature_counts.most_common(1)[0][1]
    for i, signature in enumerate(function_signatures):
        if signature == most_common_signature:
            return test_sets[i]
    return test_sets[0]

def analyze_test_coverage(
    problem_statement: str,
    test_code: str,
    function_metadata: dict = None
) -> dict:
    """
    Analyzes test coverage and identifies gaps in test requirements.

    Args:
        problem_statement: The problem description
        test_code: Generated test code to analyze
        function_metadata: Optional function metadata for enhanced analysis

    Returns:
        Dictionary with coverage analysis including:
        - coverage_score: float (0.0 to 1.0)
        - missing_requirements: list of missing test requirements
        - missing_edge_cases: list of missing edge case tests
        - recommendations: list of suggested improvements
    """

    def check_response(response: dict) -> tuple[bool, str]:
        """Validates the coverage analysis structure."""
        required_keys = [
            "coverage_score",
            "total_requirements",
            "covered_requirements",
            "missing_requirements",
            "missing_edge_cases",
            "recommendations"
        ]

        for key in required_keys:
            if key not in response:
                return False, f"Missing required key: {key}"
        if not isinstance(response["coverage_score"], (int, float)):
            return False, "coverage_score must be a number"
        if not 0.0 <= response["coverage_score"] <= 1.0:
            return False, "coverage_score must be between 0.0 and 1.0"
        if not isinstance(response["covered_requirements"], list):
            return False, "covered_requirements must be a list"

        for req in response["covered_requirements"]:
            if not isinstance(req, dict):
                return False, "Each covered requirement must be a dict"
            if "requirement" not in req or "test_cases" not in req or "coverage" not in req:
                return False, "covered_requirements missing required fields"
            if req["coverage"] not in ["full", "partial"]:
                return False, f"Invalid coverage value: {req['coverage']}"
        if not isinstance(response["missing_requirements"], list):
            return False, "missing_requirements must be a list"

        for req in response["missing_requirements"]:
            if not isinstance(req, dict):
                return False, "Each missing requirement must be a dict"
            if "requirement" not in req or "severity" not in req:
                return False, "missing_requirements missing required fields"
            if req["severity"] not in ["high", "medium", "low"]:
                return False, f"Invalid severity: {req['severity']}"

        return True, ""

    def prioritize_gaps(response: dict) -> dict:
        """Sorts gaps by severity and adds priority scores."""
        severity_order = {"high": 3, "medium": 2, "low": 1}
        response["missing_requirements"].sort(
            key=lambda x: severity_order.get(x.get("severity", "low"), 0),
            reverse=True
        )

        for edge_case in response["missing_edge_cases"]:
            if "severity" not in edge_case:
                edge_case["severity"] = "medium"

        return response

    retry = 0
    max_retries = 3

    user_content = f"""
# Problem Statement:
{problem_statement}

# Generated Test Code:
{test_code}
"""

    if function_metadata:
        user_content += f"\n# Function Metadata:\n{json.dumps(function_metadata, indent=2)}"

    messages = [
        {"role": "system", "content": TEST_COVERAGE_ANALYSIS_PROMPT},
        {"role": "user", "content": user_content}
    ]

    while retry < max_retries:
        try:
            response_text = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)
            response_text = response_text.replace('```json', '').strip('```').strip()
            json_response = json.loads(response_text)

            is_valid, error_msg = check_response(json_response)

            if is_valid:
                return prioritize_gaps(json_response)
            else:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": f"Error: {error_msg}. Please fix and try again."})

        except Exception as e:
            if retry < max_retries - 1:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": f"Exception: {str(e)}"})

        retry += 1
        time.sleep(1)
    return {
        "coverage_score": 0.5,
        "total_requirements": 0,
        "covered_requirements": [],
        "missing_requirements": [],
        "missing_edge_cases": [],
        "recommendations": ["Coverage analysis failed, manual review recommended"]
    }

def generate_missing_tests(coverage_analysis: dict, test_code: str, problem_statement: str) -> str:
    """
    Generates additional test code for missing requirements.

    Args:
        coverage_analysis: Coverage analysis dict from analyze_test_coverage
        test_code: Existing test code
        problem_statement: Original problem statement

    Returns:
        Augmented test code with missing tests added
    """
    if coverage_analysis["coverage_score"] >= 0.85:
        return test_code  # Good enough coverage

    missing_tests = []
    for req in coverage_analysis["missing_requirements"]:
        if req.get("severity") == "high" and "suggested_test" in req:
            missing_tests.append(req["suggested_test"])
    for edge_case in coverage_analysis["missing_edge_cases"]:
        if edge_case.get("severity") in ["high", "medium"] and "suggested_test" in edge_case:
            missing_tests.append(edge_case["suggested_test"])

    if not missing_tests:
        return test_code

    augmented = test_code.rstrip()
    augmented += "\n\n    # Auto-generated tests for missing coverage\n"
    for i, test in enumerate(missing_tests, 1):
        augmented += f"\n    # Missing test {i}\n"
        augmented += test + "\n"

    return augmented

def get_problem_analysis(problem_statement: str) -> list:
    def validate_response(response: dict) -> tuple[bool, str]:
        if "problem_type" not in response:
            return False, "Required key problem_type not found in response"
        return True, ""

    code_skeleton = get_code_skeleton()

    messages = [
        {"role": "system", "content": PROBLEM_ANALYSIS_SYSTEM_PROMPT.format(problem_statement=problem_statement)},
        {"role": "user", "content": f"# Code Skeleton:\n{code_skeleton}\n"}
    ]
    detailed_problem_analysis = {}
    while True:
        retry = 0

        while retry < 3:
            try:
                response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)
                response = response.replace('```json', '').strip('```').strip()
                detailed_problem_analysis = json.loads(response)

                is_valid, error_msg = validate_response(detailed_problem_analysis)
                if is_valid:
                    return detailed_problem_analysis
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {"role": "user",
                     "content": "Keep clarifying the problem analysis until you have a valid JSON object."}
                )

            except Exception as e:
                pass

            retry += 1

        if retry >= 3:
            break

    if not detailed_problem_analysis:
        try:
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)
            response = response.replace('```json', '').strip('```').strip()
            detailed_problem_analysis = json.loads(response)

            is_valid, error_msg = validate_response(detailed_problem_analysis)
            if is_valid:
                return detailed_problem_analysis
            else:
                return response

        except Exception as e:
            pass

    return detailed_problem_analysis

def process_create_task(input_dict, enable_pev: bool = True, enable_mcts: bool = True):
    problem_statement = input_dict.get("problem_statement", "")
    problem_statement = post_process_instruction(problem_statement)

    detailed_problem_analysis = get_problem_analysis(problem_statement)

    code_skeleton = get_code_skeleton()
    start_time = time.time()
    initial_solution = generate_initial_solution(problem_statement, code_skeleton, detailed_problem_analysis)
    created_files = extract_and_write_files(initial_solution)

    test_cases = generate_test_files(problem_statement, created_files, code_skeleton)

    try:
        coverage_analysis = analyze_test_coverage(
            problem_statement,
            test_cases,
            function_metadata=None
        )
        high_severity_gaps = [
            req for req in coverage_analysis['missing_requirements']
            if req.get('severity') == 'high'
        ]
        COVERAGE_THRESHOLD = 0.75
        if coverage_analysis['coverage_score'] < COVERAGE_THRESHOLD:

            augmented_tests = generate_missing_tests(coverage_analysis, test_cases, problem_statement)
            test_cases = augmented_tests

    except Exception as e:
        pass
    test_files = extract_and_write_files(test_cases)

    timeout = DEFAULT_TIMEOUT - (time.time() - start_time) - 60

    patch = fix_task_solve_workflow(
        problem_statement,
        timeout=timeout,
        run_id_1=run_id,
        test_runner=f"unittest",
        test_runner_mode="FILE",
        n_max_steps=120,
        enable_pev=enable_pev,
        enable_mcts=enable_mcts,
        extra_fix_request=SOLVE_TASK_NON_FUNCTIONAL_TEST_PROMPT
    )

    if patch is None:
        extract_and_write_files(initial_solution)

    tool_manager = EnhancedToolManager()
    patch = tool_manager.get_final_git_patch()
    return patch

def get_code_skeleton() -> str:
    result = ""
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                result += f"{file}\n{{\n{content}\n}}\n\n"

    return result

def get_directory_tree(start_path: str = '.') -> str:

    tree_lines = []

    def add_directory_tree(path: str, prefix: str = "", is_last: bool = True, is_root: bool = False):
        """Recursively build the tree structure"""
        try:
            dir_name = os.path.basename(path) if path != '.' else os.path.basename(os.getcwd())
            if not is_root:
                connector = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{connector}{dir_name}/")
            try:
                items = os.listdir(path)
                items = [item for item in items if not item.startswith('.')]
                items.sort()
                dirs = []
                files = []
                for item in items:
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        dirs.append(item)
                    else:
                        files.append(item)
                for i, dir_name in enumerate(dirs):
                    dir_path = os.path.join(path, dir_name)
                    is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                    new_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                    add_directory_tree(dir_path, new_prefix, is_last_dir, False)
                for i, file_name in enumerate(files):
                    is_last_file = i == len(files) - 1
                    connector = "└── " if is_last_file else "├── "
                    tree_lines.append(
                        f"{prefix}{'' if is_root else ('    ' if is_last else '│   ')}{connector}{file_name}"
                    )

            except PermissionError:
                error_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                tree_lines.append(f"{error_prefix}└── [Permission Denied]")

        except Exception as e:
            tree_lines.append(f"{prefix}└── [Error: {str(e)}]")

    add_directory_tree(start_path, is_root=True)
    return "\n".join(tree_lines)

def find_readme(file_path: str, repo_path: str) -> Optional[str]:
    """Find README file by traversing up from the given path."""
    current_dir = os.path.dirname(file_path)

    while True:
        for readme_name in ['README.md', 'README.rst']:
            readme_path = os.path.join(current_dir, readme_name)
            if os.path.exists(readme_path):
                return readme_path
        if current_dir == repo_path:
            break
        current_dir = os.path.dirname(current_dir)

    return None

def find_test_runner(readme_file_path: Optional[str] = None):
    if not readme_file_path:
        return "pytest"
    try:
        with open(readme_file_path, "r", encoding='utf-8') as f:
            readme_content = f.read()

        response = EnhancedNetwork.make_request(
            [
                {"role": "system", "content": FIND_TEST_RUNNER_PROMPT},
                {"role": "user", "content": readme_content}
            ], model=DEEPSEEK_MODEL_NAME
        )
        return response.strip() or "pytest"
    except Exception as e:
        return "pytest"

def filepath_to_module(file_path: str, repo_path: str, test_runner: str) -> str:
    """Convert file path to Python module notation."""
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)
    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)
    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)

    return module_path.replace(os.path.sep, '.')

def clean_filepath(file_path: str, repo_path: str, test_runner: str) -> str:
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)

    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)

    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)

    return module_path

def get_test_runner_mode(test_runner: str):
    if test_runner == 'pytest':
        return "FILE"

    try:
        with open(test_runner, "r", encoding='utf-8') as f:
            runner_content = f.read()

        response = EnhancedNetwork.make_request(
            [
                {"role": "system", "content": TEST_RUNNER_MODE_PROMPT},
                {"role": "user", "content": runner_content}
            ], model=DEEPSEEK_MODEL_NAME
        )
        return response.strip() or "FILE"
    except Exception as e:
        return "FILE"

def count_test_cases(file_path: str) -> int:
    """Count the number of test cases (functions starting with 'test_') in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        import re
        test_functions = re.findall(r'^\s*def\s+test_\w+', content, re.MULTILINE)
        return len(test_functions)

    except (FileNotFoundError, UnicodeDecodeError):
        return 0

def get_test_runner_and_mode():
    test_runner = "pytest"
    test_runner_mode = "FILE"
    test_files = []  # Initialize the test_files list
    test_file_path = None

    for root, _, files in os.walk('.'):
        for file in files:
            if 'test_' in file and file.endswith('.py'):
                test_files.append(os.path.join(root, file))

    test_files.sort(key=len)

    for path in test_files:
        if count_test_cases(path) > 5:
            test_file_path = path
            break

    if not test_file_path:
        return "pytest", "FILE"
    readme_file_path = find_readme(test_file_path, '.')

    if readme_file_path:
        test_runner = find_test_runner(readme_file_path)
        test_runner_mode = get_test_runner_mode(test_runner)

    return test_runner, test_runner_mode

def determine_agent_stratigy_for_problem_statement(problem_statement: str, repo_structure: str) -> dict:
    """
    Fallback routing based on keyword analysis when LLM routing fails.
    """
    problem_lower = problem_statement.lower()
    
    # NCTS_AGENT indicators
    ncts_keywords = [
        'fix bug', 'regression', 'backward compatible', 'issue', 
        'test suite', 'django', 'flask', 'pytest', 'unittest', 'multi-file',
        'complex', 'investigation', 'deep analysis'
    ]
    
    # STEAMEDLINE_AGENT indicators
    streamedline_keywords = [
        'create', 'implement from scratch', 'quick', 'quickly',
        'fast', 'urgent', 'simple', 'straightforward', 'javascript', 'golang',
        'rust', 'typescript', 'node', 'react'
    ]
    
    ncts_score = sum(1 for kw in ncts_keywords if kw in problem_lower)
    sam_score = sum(1 for kw in streamedline_keywords if kw in problem_lower)  # 007
    
    # Check for non-Python files in repo structure 
    if repo_structure:
        non_python_extensions = ['.js', '.go', '.rs', '.ts', '.jsx', '.tsx', '.java', '.cpp']
        for ext in non_python_extensions:
            if ext in repo_structure:
                sam_score += 2  # 
    
    # Make decision
    if sam_score > ncts_score:
        agent = "STEAMEDLINE_AGENT"
        confidence = min(0.6 + (sam_score - ncts_score) * 0.1, 0.9)
        reasoning = f"Keyword analysis: {sam_score} STREAMLINE indicators vs {ncts_score} NCTS indicators"
    else:
        agent = "NCTS_AGENT"
        confidence = min(0.6 + (ncts_score - sam_score) * 0.1, 0.9)
        reasoning = f"Keyword analysis: {ncts_score} NCTS indicators vs {sam_score} STREAMLINE indicators (default to NCTS for robustness)"
    
    #print(f"[ROUTER] Fallback routing: {agent} (confidence: {confidence:.2f})")
    #print(f"[ROUTER] Reasoning: {reasoning}")
    
    return {
        "agent": agent,
        "confidence": confidence,
        "reasoning": reasoning
    }

def post_process_instruction(instruction: str) -> str:
    """
    Post-processes instruction to mark whitespaces and empty lines explicitly.
    """
    import re
    
    def apply_markup(text_block: str) -> str:
        """Apply markup to make whitespaces and empty lines explicit."""
        lines = text_block.split('\n')
        processed_lines = []
        
        should_apply_markup = True
        for line in lines:
            if line.strip() == '':
                should_apply_markup = True
                break
            if line[-1] != "." and line[-1] != "!":
                should_apply_markup = False
                break
            
        if should_apply_markup == False:
            return text_block

        for i, line in enumerate(lines):
            if line.strip() == '':                
                processed_line = '[EMPTY_LINE]'
            else:
                leading_spaces = len(line) - len(line.lstrip(' '))
                trailing_spaces = len(line) - len(line.rstrip(' '))
                
                processed_line = line
                if leading_spaces > 0:
                    processed_line = f'[{leading_spaces}_LEADING_SPACES]' + line.lstrip(' ')
                if trailing_spaces > 0:
                    processed_line = processed_line.rstrip(' ') + f'[{trailing_spaces}_TRAILING_SPACES]'
            
            processed_lines.append(f"\"{processed_line}\"")
        
        return "[\n    " + ",\n    ".join(processed_lines) + "\n]"
            
    pattern = r'```text\n(.*?)\n```'
    
    def replace_text_block(match):
        text_content = match.group(1)
        processed_content = apply_markup(text_content)
        return f'```text\n{processed_content}\n```'
    
    processed_instruction = re.sub(pattern, replace_text_block, instruction, flags=re.DOTALL)
    return processed_instruction

def generate_test_files_streamlined(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    """Generate test files using consensus approach (15 parallel generations)."""
    from collections import Counter
    import hashlib
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def clean_testcode_response(response: str) -> str:
        """Helper function to clean AI response from markdown formatting"""
        testcases = response.strip()
        if testcases.startswith('```python'):
            testcases = testcases[9:]
        if testcases.startswith('```'):
            testcases = testcases[3:]
        if testcases.endswith('```'):
            testcases = testcases[:-3]
        return testcases.strip()
    
    def normalize_ast(node):
        """Return normalized AST string (logic+data preserved, var names normalized)."""
        node = ast.fix_missing_locations(node)
        node = VariableNormalizer().visit(node)
        for attr in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'):
            if hasattr(node, attr):
                setattr(node, attr, None)
        for child in ast.iter_child_nodes(node):
            normalize_ast(child)
        return ast.dump(node, include_attributes=False)

    def get_function_signatures_with_logic(code: str) -> dict:
        """Return dict of {function_name: logic+data_hash}"""
        signatures = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    normalized = normalize_ast(node)
                    hash_val = hashlib.sha256(normalized.encode()).hexdigest()
                    signatures[node.name] = hash_val
        except SyntaxError:
            pass
        return signatures
    
    def generate_single_testset() -> tuple:
        """Generate a single test set and return (testcode, function_signatures)"""
        retry = 0
        test_generation_messages = [
            {
                "role": "system",
                "content": GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT
            },
            {
                "role": "user",
                "content": f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"
            }
        ]
        
        while retry < 10:
            try:
                temperature = random.uniform(0, 0.05)
                testcode_response = EnhancedNetwork.make_request(test_generation_messages, model=QWEN_MODEL_NAME, temperature=temperature)
                #print("Step 1 - Testcase Generation completed")
                
                testcases_check_messages = [
                    {
                        "role": "system",
                        "content": TESTCASES_CHECK_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"Problem statement: {problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerated Test Code:\n{testcode_response}\n\nAnalyze this code for invalid testcases. Return ONLY the final Python test code."
                    }   
                ]
                
                testcode_checked_response = EnhancedNetwork.make_request(testcases_check_messages, model=QWEN_MODEL_NAME)
                #print("Step 2 - Testcase check completed")

                testcases = clean_testcode_response(testcode_checked_response)
                
                lines = testcases.split("\n")
                if lines[0].endswith(".py") == False:
                    retry += 1
                    test_generation_messages.append({"role": "assistant", "content": testcode_checked_response})
                    test_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"})
                    #print(f"Retrying because the first line is not a python test file name:\n {testcases}")
                    continue

                function_signatures = get_function_signatures_with_logic(testcases)
                #print(f"Generated testset with function logic signatures: {function_signatures}")
                return testcases, function_signatures
                
            except Exception as e:
                retry += 1
                #print(f"Exception in generate_single_testset: {e}")
                time.sleep(2)
        
        return "", set()
    
    def compute_common_score(function_signatures):
        """Compute the average commonness of functions across all testsets."""
        all_pairs = [tuple(fs.items()) for fs in function_signatures]
        pair_counts = Counter()
        for pairs in all_pairs:
            pair_counts.update(pairs)

        scores = []
        for pairs in all_pairs:
            if not pairs:
                scores.append(0)
                continue
            total_occurrences = sum(pair_counts[p] for p in pairs)
            avg_occurrence = total_occurrences / len(pairs)
            scores.append(avg_occurrence)

        best_score = max(scores)
        best_indices = [i for i, s in enumerate(scores) if abs(s - best_score) < 1e-6]
        best_index = max(best_indices, key=lambda i: len(all_pairs[i]))

        #print(f"Best testset index: {best_index} with average score: {best_score:.3f}")
        return best_index, best_score, scores
    
    # Generate multiple test sets (15 times) with parallel processing
    NUM_GENERATIONS = 15
    test_sets = []
    all_function_signatures = []
    
    #print(f"Generating {NUM_GENERATIONS} test sets to find the most common pattern (5 parallel workers)...")
    
    def generate_with_index(i):
        """Wrapper function to generate a test set with its index."""
        #print(f"Generating test set {i+1}/{NUM_GENERATIONS}")
        testcode, function_signatures = generate_single_testset()
        return (i, testcode, function_signatures)
    
    # Use ThreadPoolExecutor to process 5 at a time
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(generate_with_index, i): i for i in range(NUM_GENERATIONS)}
        
        results = []
        for future in as_completed(futures):
            try:
                i, testcode, function_signatures = future.result()
                results.append((i, testcode, function_signatures))
            except Exception as e:
                i = futures[future]
                #print(f"Failed to generate test set {i+1}: {e}")
        
        results.sort(key=lambda x: x[0])
        
        for i, testcode, function_signatures in results:
            if testcode and function_signatures:
                test_sets.append(testcode)
                all_function_signatures.append(function_signatures)
            else:
                #print(f"Failed to generate valid test set {i+1}")
                pass
    
    if not test_sets:
        #print("Failed to generate any valid test sets")
        return ""
    
    best_index, best_score, scores = compute_common_score(all_function_signatures)
    
    #print(f"Most common testcase: {best_index} (appeared {best_score}/{len(all_function_signatures)}|{scores})")
    return test_sets[best_index]

def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    """Extract files from solution text and write them to disk."""
    created_files = []
    
    if not initial_solution.strip():
        #print("No solution content to process")
        return created_files
    
    lines = initial_solution.split('\n')
    current_filename = None
    current_content = []
    
    for line in lines:
        stripped_line = line.strip()
        
        if (stripped_line.endswith('.py') and 
            ' ' not in stripped_line and 
            len(stripped_line) > 3 and 
            '/' not in stripped_line.replace('/', '') and
            not stripped_line.startswith('#')):
            if current_filename and current_content:
                file_path = os.path.join(base_dir, current_filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                content = '\n'.join(current_content).strip()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                created_files.append(file_path)
            current_filename = stripped_line
            current_content = []
        else:
            if current_filename:
                current_content.append(line)
    if current_filename and current_content:
        file_path = os.path.join(base_dir, current_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        content = '\n'.join(current_content).strip()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        created_files.append(file_path)
        #print(f"Created file: {file_path}")
    return created_files

def validate_boundary_coverage(test_content: str, problem_statement: str) -> list:
    """
    Validate that generated tests cover all boundary cases for validation problems.
    
    Returns a list of missing test cases that should be added.
    """
    import re
    import ast
    
    missing_tests = []
    problem_lower = problem_statement.lower()
    
    # Only validate if problem has validation requirements (TypeError AND ValueError)
    has_type_error = 'typeerror' in problem_lower
    has_value_error = 'valueerror' in problem_lower
    
    if not (has_type_error and has_value_error):
        # Not a validation problem, skip boundary checks
        return missing_tests
    
    #print(" Detected validation problem (TypeError + ValueError)")
    
    # Extract all test function bodies that test for exceptions
    exception_tests = re.findall(
        r'def (test_\w+)\(self\):.*?(?=\n    def |\nif __name__|$)',
        test_content,
        re.DOTALL
    )
    
    # Find all assertRaises calls and extract the test inputs
    test_inputs = {}  # {error_type: [test_cases]}
    
    for test_match in exception_tests:
        test_name = test_match if isinstance(test_match, str) else test_match[0]
        
        # Find assertRaises(TypeError) and assertRaises(ValueError)
        type_error_matches = re.findall(
            r'assertRaises\(TypeError\).*?[\(\[]([^\)]+)[\)\]]',
            test_content,
            re.DOTALL
        )
        value_error_matches = re.findall(
            r'assertRaises\(ValueError\).*?[\(\[]([^\)]+)[\)\]]',
            test_content,
            re.DOTALL
        )
        
        test_inputs['TypeError'] = type_error_matches
        test_inputs['ValueError'] = value_error_matches
    
    # Analyze tuple/list length patterns in the problem
    # Look for patterns like "(TYPE, ...)" or "[...]" in problem statement
    tuple_patterns = re.findall(r'\(([A-Z_]+)[,\s]+[^\)]*\)', problem_statement)
    
    if tuple_patterns:
        #print(f" Detected tuple-based validation (types: {set(tuple_patterns)})")
        
        # For each type, check if we test all boundary lengths
        # Extract what lengths are tested
        tested_lengths = set()
        
        for error_type, inputs in test_inputs.items():
            for input_str in inputs:
                # Count commas to estimate tuple length
                # This is a heuristic - count elements in tuple-like structures
                if '(' in input_str:
                    # Extract tuple content
                    tuple_content = re.search(r'\(([^\)]*)\)', input_str)
                    if tuple_content:
                        elements = [e.strip() for e in tuple_content.group(1).split(',') if e.strip()]
                        tested_lengths.add(len(elements))
        
        #print(f" Tested tuple lengths: {sorted(tested_lengths)}")
        
        # Check if we have comprehensive boundary coverage
        # For validation problems, we should test: 0, 1, 2, ..., up to valid+2
        # Heuristic: if problem mentions 3-element or 4-element structures, test 0-6
        expected_lengths = set(range(0, 7))  # 0, 1, 2, 3, 4, 5, 6
        missing_lengths = expected_lengths - tested_lengths
        
        if missing_lengths and len(tested_lengths) > 0:
            # Only report if we have some tests but are missing critical boundaries
            # Focus on the critical missing cases: 0, 1, and around the valid length
            critical_missing = missing_lengths & {0, 1, 2, 3}
            
            if critical_missing:
                for length in sorted(critical_missing):
                    if length == 0:
                        missing_tests.append("MUST test empty tuple: () - should raise TypeError('Graph item incomplete')")
                    elif length == 1:
                        # This is the critical case that was missing!
                        missing_tests.append("MUST test 1-element tuples for EACH type (e.g., (ATTR,), (NODE,), (EDGE,)) - determine if TypeError or ValueError")
                    elif length == 2:
                        missing_tests.append("MUST test 2-element tuples for types that need more elements")
                    elif length == 3:
                        missing_tests.append("MUST test 3-element tuples for types that need different counts")
    
    # Additional check: Look for "incomplete" vs "malformed" distinction
    if 'incomplete' in problem_lower and 'malformed' in problem_lower:
        #print(" Detected 'incomplete' vs 'malformed' distinction")
        
        # Check if tests cover the boundary between incomplete and malformed
        has_incomplete_tests = 'incomplete' in test_content.lower()
        has_malformed_tests = 'malformed' in test_content.lower()
        
        if not has_incomplete_tests:
            missing_tests.append("MUST test 'incomplete' cases (minimal/insufficient data)")
        if not has_malformed_tests:
            missing_tests.append("MUST test 'malformed' cases (wrong structure/type)")
    
    # if missing_tests:
    #     #print(f" Found {len(missing_tests)} missing boundary test cases")
    # else:
    #     #print(" ✓ Boundary coverage looks comprehensive")
    
    return missing_tests
