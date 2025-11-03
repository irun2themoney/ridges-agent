# This file contains CREATE task functions extracted from agent.py
# for code organization and to reduce agent.py size

import os
import sys
import json
import time
import re
import ast
import hashlib
import textwrap
from typing import Any, Dict, List, Optional
from pathlib import Path
from collections import Counter

# Import core classes from agent
from agent import Network, QWEN_MODEL_NAME, AGENT_MODELS, DEFAULT_TIMEOUT, run_id
from agent import GENERATE_INITIAL_SOLUTION_PROMPT, GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT
from agent import GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT, GEN_TESTCASES_PROMPT
from agent import INFINITE_LOOP_CHECK_PROMPT, BACKWARDS_COMPATIBILITY_PROMPT, TESTCASES_CHECK_PROMPT
from agent import TEMPERATURE_DETERMINATION_SYSTEM_PROMPT, PROBLEM_ANALYSIS_SYSTEM_PROMPT
from agent import SOLVE_TASK_NON_FUNCTIONAL_TEST_PROMPT, TEST_COVERAGE_ANALYSIS_PROMPT
from agent import TEST_RUNNER_MODE_PROMPT, FIX_TASK_NEVER_EARLY_STOP_PROMPT
from agent import Utils, get_current_code_skeleton

def post_process_instruction(instruction: str) -> str:
    """
    Post-processes instruction to mark whitespaces and empty lines explicitly.
    """
    import re

    def apply_markup(text_block: str) -> str:
        """
        Apply markup to make whitespaces and empty lines explicit to make llm not confusing and ignoring them.
        For example, if the text block is:

        ```text
        This is a test.

        This is another test!
        ```text

        Then the text block should be:

        ```
        This is a test.
        [EMPTY_LINE]
        This is another test!
        ```
        """
        lines = text_block.split("\n")
        processed_lines = []

        should_apply_markup = True
        for line in lines:
            if line.strip() == "":
                should_apply_markup = True
                break
            if line[-1] != "." and line[-1] != "!":
                should_apply_markup = False
                break

        if should_apply_markup == False:
            return text_block

        for i, line in enumerate(lines):
            if line.strip() == "":
                processed_line = "[EMPTY_LINE]"
            else:
                # Mark trailing and leading spaces
                leading_spaces = len(line) - len(line.lstrip(" "))
                trailing_spaces = len(line) - len(line.rstrip(" "))

                processed_line = line
                if leading_spaces > 0:
                    processed_line = f"[{leading_spaces}_LEADING_SPACES]" + line.lstrip(" ")
                if trailing_spaces > 0:
                    processed_line = processed_line.rstrip(" ") + f"[{trailing_spaces}_TRAILING_SPACES]"

            processed_lines.append(f'"{processed_line}"')

        return "[\n    " + ",\n    ".join(processed_lines) + "\n]"

    # Pattern to match ```text...``` blocks
    pattern = r"```text\n(.*?)\n```"

    def replace_text_block(match):
        text_content = match.group(1)
        processed_content = apply_markup(text_content)

        return f"```text\n{processed_content}\n```"

    # Replace all text blocks with processed versions
    processed_instruction = re.sub(pattern, replace_text_block, instruction, flags=re.DOTALL)
    return processed_instruction


def enhance_problem_statement(problem_statement: str) -> str:

    ENHANCEMENT_PROMPT = textwrap.dedent(
        """
            You are an expert at analyzing problem statements and extracting key information.
        
            Analyze the given problem statement and extract the following structured information:
        
            1. **Problem Summary** (1-2 sentences): What needs to be fixed or implemented?
        
            2. **Current Behavior**: What is happening now? (Include error messages, unexpected outputs, etc.)
        
            3. **Expected Behavior**: What should happen instead?
        
            4. **Reproduction Steps** (if applicable): Clear steps to reproduce the issue
        
            5. **Success Criteria**: How will we know the problem is solved?
               - What tests should pass?
               - What behavior should change?
               - What outputs should be different?
        
            6. **Key Requirements**:
               - Must-have functionality
               - Constraints to respect (backwards compatibility, performance, etc.)
               - Files/functions likely involved
        
            7. **Important Notes**:
               - Edge cases to consider
               - Potential pitfalls
               - Related functionality that might be affected
        
            If any section is not applicable or cannot be determined from the problem statement, write "Not specified" for that section.
        
            Format your response as markdown with clear section headers.
            Be concise but complete. Extract information that's present, don't invent details.
            """
    )

    try:
        messages = [{"role": "system", "content": ENHANCEMENT_PROMPT}, {"role": "user", "content": f"Problem Statement:\n\n{problem_statement}"}]

        enhanced = Network.make_request(messages, model=QWEN_MODEL_NAME, temperature=0.0)

        return enhanced

    except Exception as e:
        return ""


def generate_test_files(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    retry = 0
    while retry < 10:
        try:
            testcases = generate_testcases_with_multi_step_reasoning(problem_statement, files_to_test, code_skeleton)

            if testcases:

                return testcases
            else:

                # Fallback to original single-step approach if multi-step fails
                messages = [
                    {"role": "system", "content": GEN_TESTCASES_PROMPT},
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nPython files to test:\n{files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the ground truth and edge case coveraging testcases.""",
                    },
                ]

                response = Network.make_request(messages, model=QWEN_MODEL_NAME)

                # Clean up the response
                testcases = response.strip()
                if testcases.startswith("```python"):
                    testcases = testcases[9:]
                if testcases.startswith("```"):
                    testcases = testcases[3:]
                if testcases.endswith("```"):
                    testcases = testcases[:-3]
                testcases = testcases.strip()

                return testcases

        except Exception as e:
            retry += 1
            time.sleep(2)

    if retry >= 10:

        return ""
    return ""


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
        "pattern_matches": {"edge_case_headers": [], "handled_cases_summary": [], "inline_comments": [], "test_related": []},
    }

    # Enhanced pattern matching for different comment styles
    patterns = {
        "edge_case_headers": [
            r"#\s*Edge\s*Case:\s*(.+)",
            r"#\s*EDGE\s*CASE:\s*(.+)",
            r"#\s*Edge\s*case:\s*(.+)",
            r"#\s*TODO:\s*handle\s*edge\s*case[:\s]*(.+)",
            r"#\s*Handle\s+edge\s+case:\s*(.+)",
        ],
        "handled_cases_summary": [
            r"#\s*Handled\s*Edge\s*Cases?:\s*(.+)",
            r"#\s*Edge\s*cases\s*handled:\s*(.+)",
            r"#\s*Covered\s*edge\s*cases:\s*(.+)",
            r"#\s*Edge\s*cases:\s*(.+)",
        ],
        "inline_comments": [
            r"#\s*(?:null|empty|zero|negative|invalid|boundary|limit)\s*case",
            r"#\s*(?:handle|check|validate).*(?:edge|corner|boundary)",
            r"#\s*Special\s+case:",
        ],
        "test_related": [r"#\s*Test\s+case.*edge", r"#\s*Edge\s+case\s+test"],
    }

    # Apply all patterns
    for category, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, solution, re.IGNORECASE | re.MULTILINE)
            validation_result["pattern_matches"][category].extend(matches)

    # Calculate coverage score
    total_patterns_found = sum(len(matches) for matches in validation_result["pattern_matches"].values())
    validation_result["coverage_score"] = min(total_patterns_found / 10.0, 1.0)  # Normalize to 0-1

    # Determine comment quality based on multiple factors
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
    analysis = {"potential_missing_cases": [], "code_complexity_indicators": [], "risk_assessment": "low"}

    # Look for common edge case indicators in code
    edge_case_indicators = {
        "null_checks": ["if.*is None", "if.*== None", "if.*!= None"],
        "empty_checks": ['if.*== ""', 'if.*!= ""', "if.*len\\(.*\\) == 0"],
        "boundary_checks": ["if.*== 0", "if.*< 0", "if.*> 0", "if.*<= 0", "if.*>= 0"],
        "type_checks": ["isinstance", "type\\(", "str\\(", "int\\(", "float\\("],
        "exception_handling": ["try:", "except:", "raise", "finally:"],
    }

    found_indicators = {}
    for category, patterns in edge_case_indicators.items():
        found_indicators[category] = []
        for pattern in patterns:
            matches = re.findall(pattern, solution, re.IGNORECASE)
            found_indicators[category].extend(matches)

    # Suggest missing edge cases based on code analysis
    if not found_indicators["null_checks"]:
        analysis["potential_missing_cases"].append("null/None value handling")
    if not found_indicators["empty_checks"]:
        analysis["potential_missing_cases"].append("empty string/list handling")
    if not found_indicators["boundary_checks"]:
        analysis["potential_missing_cases"].append("boundary value validation")

    # Assess risk level
    total_indicators = sum(len(matches) for matches in found_indicators.values())
    if total_indicators < 3:
        analysis["risk_assessment"] = "high"
    elif total_indicators < 6:
        analysis["risk_assessment"] = "medium"

    return analysis


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
        {"role": "user", "content": f"Problem statement: {problem_statement}"},
    ]
    temperature = 0
    while True:
        retry = 0

        while retry < 3:
            try:
                response = Network.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)
                response = response.replace("```json", "").strip("```").strip()
                response = json.loads(response)

                is_valid, error_msg = validate_response(response)
                if is_valid:
                    return response.get("temperature", 0.0)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Keep clarifying the temperature until you have a valid float."})

            except Exception as e:
                pass

            retry += 1

        if retry >= 3:
            break
    if not response.get("temperature", 0):
        try:
            response = Network.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)
            response = response.replace("```json", "").strip("```").strip()
            response = json.loads(response)

            is_valid, error_msg = validate_response(response)
            if is_valid:
                return response.get("temperature", 0.0)
            else:
                return 0

        except Exception as e:
            return 0

    return 0


def generate_initial_solution(problem_statement: str, code_skeleton: str, detailed_problem_analysis: dict) -> str:
    problem_statement_with_spec = problem_statement
    spec = detailed_problem_analysis if isinstance(detailed_problem_analysis, str) else json.dumps(detailed_problem_analysis, indent=4)
    problem_statement_with_spec += "\nImplement the functions referencing detailed problem specifications.\n" f"Analysis:\n{spec}\n"
    temperature = determine_temperature(problem_statement_with_spec)

    # Generate three different solutions
    solutions = []
    retry = 0

    while len(solutions) < 3 and retry < 10:
        try:
            # Try multi-step reasoning first
            solution = generate_solution_with_multi_step_reasoning(problem_statement, code_skeleton, temperature)

            if solution:
                solutions.append(solution)
            else:
                # Fallback to single-step approach
                messages = [
                    {"role": "system", "content": GENERATE_INITIAL_SOLUTION_PROMPT},
                    {
                        "role": "user",
                        "content": f"""Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\n\nGenerate the complete and correct implementation in python files.""",
                    },
                ]

                response = Network.make_request(messages, model=QWEN_MODEL_NAME)

                # Clean up the response
                solution = response.strip()
                if solution.startswith("```python"):
                    solution = solution[9:]
                if solution.startswith("```"):
                    solution = solution[3:]
                if solution.endswith("```"):
                    solution = solution[:-3]
                solution = solution.strip()

                if solution:
                    solutions.append(solution)

        except Exception as e:
            retry += 1
            time.sleep(2)

    if not solutions:

        return ""

    # If we have only one solution, return it
    if len(solutions) == 1:

        return solutions[0]
    best_index = max([0, 1, 2], key=lambda i: len(solutions[i]))
    return solutions[best_index]


def generate_solution_with_multi_step_reasoning(problem_statement: str, code_skeleton: str, temperature: float) -> str:
    BEST_PRACTICE_VERIFY_PROMPT = textwrap.dedent(
        """
        You are a senior software engineer to verify the solution is best practice in all aspects. Your task is to analyze the generated Python code for in-consistency, poor-security, poor-flexibilit, etc.
        
        1. Re-usability and maintainable Rules
           - Verify the solution is designed well like no duplication logic through different functions, methods, or classes.
           - Verify the solution is maintainable like no too long code, mix of messy code, difficult to find related code, etc.
        2. Consistency Rules
           - Verify consistency through types, prototypes, hidden arguements of function passing and if you find any in-consistency and make the code to be consistent in all aspects of idea, design, technically, and testcases.
        3. Security Rules or Prevent Attack Rules
           - Verify Security Best Practice in solutions such as weakness of solutions imported modules, weakness of implemented algorithms, etc and improve
           - Improve the solution to be more safe by improving imported module's limitation by an well-known method or improving algorithm with better one.
        4. Flexible Types Support Rules
           - If problem statement and code skeleton doesn't mention explicit types of parameter, prototypes of function passing or etc, try to update functions to be more flexible for several possible types
           - Generate testcases to verify to ensure it supports flexible types and fix the code based on it.
        
        If you find issues in above rules and updated the code:
        - Provide a corrected version of the code
        - Ensure final version is best practice in code design and implementation
        
        If solution is already best practice without breaking above rules:
        - Return the original code unchanged
        
        STRICT REQUIREMENT: Return the final Python code along with file names. Do not include any explanations, comments, or additional text.
        
        example:
        ```python
        a.py
        contents of a.py
        
        b.py
        contents of b.py
        ```
        """
    )
    retry = 0
    code_generation_messages = [
        {"role": "system", "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT},
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\nGenerate the complete and correct implementation in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```",
        },
    ]

    while retry < 10:
        try:
            code_response = Network.make_request(code_generation_messages, model=QWEN_MODEL_NAME, temperature=temperature)

            loop_check_messages = [
                {"role": "system", "content": INFINITE_LOOP_CHECK_PROMPT},
                {
                    "role": "user",
                    "content": f"Generated Code:\n{code_response}\n\nAnalyze this code for potential infinite loops and provide a corrected version if any issues are found. Return ONLY the final Python code.",
                },
            ]

            loop_check_response = Network.make_request(loop_check_messages, model=QWEN_MODEL_NAME)

            best_practice_verify_messages = [
                {"role": "system", "content": BEST_PRACTICE_VERIFY_PROMPT},
                {
                    "role": "user",
                    "content": f"Generated Code:\n{loop_check_response}\n\nAnalyze this code if it's written in best practice such as consistency, security, and flexible types support, etc. Return ONLY the final Python code.",
                },
            ]

            bpv_response = Network.make_request(best_practice_verify_messages, model=QWEN_MODEL_NAME)

            backwards_compat_messages = [
                {"role": "system", "content": BACKWARDS_COMPATIBILITY_PROMPT},
                {
                    "role": "user",
                    "content": f"Original Code Skeleton:\n{code_skeleton}\n\nGenerated Code:\n{bpv_response}\n\nEnsure generated code is backwards compatible with the original code. Return ONLY the final Python code.",
                },
            ]

            compat_response = Network.make_request(backwards_compat_messages, model=QWEN_MODEL_NAME)

            # Clean up the final response (use compat response as it's the final validated version)
            solution = compat_response.strip()
            if solution.startswith("```python"):
                solution = solution[9:]
            if solution.startswith("```"):
                solution = solution[3:]
            if solution.endswith("```"):
                solution = solution[:-3]
            solution = solution.strip()

            lines = solution.split("\n")
            if lines[0].endswith(".py") == False:
                retry += 1
                code_generation_messages.append({"role": "assistant", "content": code_response})
                code_generation_messages.append(
                    {
                        "role": "user",
                        "content": f"Include file name in the response. example:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```",
                    }
                )

                continue

            return solution
        except Exception as e:
            retry += 1

            time.sleep(2)

    if retry >= 10:

        return ""

    return ""


class VariableNormalizer(ast.NodeTransformer):
    """Normalize variable and argument names but keep constants intact."""

    def __init__(self):
        super().__init__()
        self.name_map = {}
        self.counter = 0

    def _get_placeholder(self, original):
        if original not in self.name_map:
            self.name_map[original] = f"var_{self.counter}"
            self.counter += 1
        return self.name_map[original]

    def visit_Name(self, node):
        node.id = self._get_placeholder(node.id)
        return node

    def visit_arg(self, node):
        node.arg = self._get_placeholder(node.arg)
        return node


def generate_testcases_with_multi_step_reasoning(problem_statement: str, files_to_test: str, code_skeleton: str) -> str:
    from collections import Counter
    import re

    def extract_function_names(testcode: str) -> set:
        """Extract function names from test code to create a signature for comparison"""
        function_names = set()
        # Look for test function patterns like def test_something, def testSomething, etc.
        test_function_patterns = [
            r"def\s+(test_\w+)",  # def test_something
            r"def\s+(test\w+)",  # def testSomething
            r"def\s+(\w*test\w*)",  # any function containing 'test'
        ]

        for pattern in test_function_patterns:
            matches = re.findall(pattern, testcode, re.IGNORECASE)
            function_names.update(matches)

        return function_names

    def clean_testcode_response(response: str) -> str:
        """Helper function to clean AI response from markdown formatting"""
        testcases = response.strip()
        if testcases.startswith("```python"):
            testcases = testcases[9:]
        if testcases.startswith("```"):
            testcases = testcases[3:]
        if testcases.endswith("```"):
            testcases = testcases[:-3]
        return testcases.strip()

    def normalize_ast(node):
        """Return normalized AST string (logic+data preserved, var names normalized)."""
        node = ast.fix_missing_locations(node)
        node = VariableNormalizer().visit(node)
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(node, attr):
                setattr(node, attr, None)
        for child in ast.iter_child_nodes(node):
            normalize_ast(child)
        return ast.dump(node, include_attributes=False)

    def get_function_signatures_with_logic(code: str) -> dict[str, str]:
        """Return dict of {function_name: logic+data_hash}"""
        signatures = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    normalized = normalize_ast(node)
                    # Hash for compact comparison
                    hash_val = hashlib.sha256(normalized.encode()).hexdigest()
                    signatures[node.name] = hash_val
        except SyntaxError:
            pass
        return signatures

    def generate_single_testset() -> tuple[str, set]:
        """Generate a single test set and return (testcode, function_names)"""
        retry = 0
        test_generation_messages = [
            {"role": "system", "content": GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT},
            {
                "role": "user",
                "content": f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```",
            },
        ]

        while retry < 10:
            try:
                temperature = random.uniform(0, 0.05)
                testcode_response = Network.make_request(test_generation_messages, model=QWEN_MODEL_NAME, temperature=temperature)

                testcases_check_messages = [
                    {"role": "system", "content": TESTCASES_CHECK_PROMPT},
                    {
                        "role": "user",
                        "content": f"Problem statement: {problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerated Test Code:\n{testcode_response}\n\nAnalyze this code for invalid testcases. Return ONLY the final Python test code.",
                    },
                ]

                testcode_checked_response = Network.make_request(testcases_check_messages, model=QWEN_MODEL_NAME)

                testcases = clean_testcode_response(testcode_checked_response)

                lines = testcases.split("\n")
                if lines[0].endswith(".py") == False:
                    retry += 1
                    test_generation_messages.append({"role": "assistant", "content": testcode_checked_response})
                    test_generation_messages.append(
                        {
                            "role": "user",
                            "content": f"Include file name in the response. example:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```",
                        }
                    )

                    continue

                # Extract function names for comparison
                function_signatures = get_function_signatures_with_logic(testcases)

                return testcases, function_signatures

            except Exception as e:
                retry += 1

                time.sleep(2)

        return "", set()

    def compute_common_score(function_signatures):
        """
        Compute the average commonness of functions across all testsets.

        Each testset gets a score:
            (sum of occurrences of its (function_name, hash) pairs across all sets)
            / (number of functions in that testset)

        Additionally logs the occurrence count of every unique (name, hash) pair.

        Tie-breaker: pick the set with the largest number of functions if scores are equal.
        """
        # Convert each test set to list of (name, hash) pairs
        all_pairs = [tuple(fs.items()) for fs in function_signatures]

        # Count how many times each (name, hash) appears across all sets
        pair_counts = Counter()
        for pairs in all_pairs:
            pair_counts.update(pairs)

        # Log global statistics
        total_unique = len(pair_counts)
        total_pairs = sum(pair_counts.values())

        scores = []
        for pairs in all_pairs:
            if not pairs:
                scores.append(0)
                continue
            total_occurrences = sum(pair_counts[p] for p in pairs)
            avg_occurrence = total_occurrences / len(pairs)
            scores.append(avg_occurrence)

        # Select best test set
        best_score = max(scores)
        best_indices = [i for i, s in enumerate(scores) if abs(s - best_score) < 1e-6]

        # Tie-breaker: pick the one with the most functions
        best_index = max(best_indices, key=lambda i: len(all_pairs[i]))

        return best_index, best_score, scores

    # Generate multiple test sets (8+ times)
    NUM_GENERATIONS = 15
    test_sets = []
    all_function_signatures = []

    for i in range(NUM_GENERATIONS):

        testcode, function_signatures = generate_single_testset()

        if testcode and function_signatures:  # Only add valid test sets
            test_sets.append(testcode)
            all_function_signatures.append(function_signatures)

    if not test_sets:
        return ""

    best_index, best_score, scores = compute_common_score(all_function_signatures)

    return test_sets[best_index]

    # Fallback: return the first valid test set

    return test_sets[0]


def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    import os

    created_files = []

    if not initial_solution.strip():

        return created_files

    lines = initial_solution.split("\n")
    current_filename = None
    current_content = []

    for line in lines:
        # Check if this line is just a Python filename (*.py pattern)
        stripped_line = line.strip()

        # Pattern: ends with .py and looks like a filename (no spaces, reasonable length)
        if (
            stripped_line.endswith(".py")
            and " " not in stripped_line
            and len(stripped_line) > 3
            and "/" not in stripped_line.replace("/", "")  # Allow subdirectories
            and not stripped_line.startswith("#")
        ):
            if current_filename and current_content:
                file_path = os.path.join(base_dir, current_filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                content = "\n".join(current_content).strip()
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                created_files.append(file_path)
            current_filename = stripped_line
            current_content = []
        else:

            if current_filename:  # Only collect content if we have a filename
                current_content.append(line)
    if current_filename and current_content:
        file_path = os.path.join(base_dir, current_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        content = "\n".join(current_content).strip()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        created_files.append(file_path)

    return created_files


def analyze_test_coverage(problem_statement: str, test_code: str, function_metadata: dict = None) -> dict:
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
            "recommendations",
        ]

        for key in required_keys:
            if key not in response:
                return False, f"Missing required key: {key}"

        # Validate coverage_score
        if not isinstance(response["coverage_score"], (int, float)):
            return False, "coverage_score must be a number"
        if not 0.0 <= response["coverage_score"] <= 1.0:
            return False, "coverage_score must be between 0.0 and 1.0"

        # Validate covered_requirements
        if not isinstance(response["covered_requirements"], list):
            return False, "covered_requirements must be a list"

        for req in response["covered_requirements"]:
            if not isinstance(req, dict):
                return False, "Each covered requirement must be a dict"
            if "requirement" not in req or "test_cases" not in req or "coverage" not in req:
                return False, "covered_requirements missing required fields"
            if req["coverage"] not in ["full", "partial"]:
                return False, f"Invalid coverage value: {req['coverage']}"

        # Validate missing_requirements
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
        # Sort missing requirements by severity
        severity_order = {"high": 3, "medium": 2, "low": 1}
        response["missing_requirements"].sort(key=lambda x: severity_order.get(x.get("severity", "low"), 0), reverse=True)

        for edge_case in response["missing_edge_cases"]:
            if "severity" not in edge_case:
                edge_case["severity"] = "medium"

        return response

    # Main logic
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

    messages = [{"role": "system", "content": TEST_COVERAGE_ANALYSIS_PROMPT}, {"role": "user", "content": user_content}]

    while retry < max_retries:
        try:
            response_text = Network.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)

            # Clean and parse
            response_text = response_text.replace("```json", "").strip("```").strip()
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

    # Graceful failure - return neutral coverage

    return {
        "coverage_score": 0.5,
        "total_requirements": 0,
        "covered_requirements": [],
        "missing_requirements": [],
        "missing_edge_cases": [],
        "recommendations": ["Coverage analysis failed, manual review recommended"],
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

    # Extract suggested tests from analysis (high priority first)
    for req in coverage_analysis["missing_requirements"]:
        if req.get("severity") == "high" and "suggested_test" in req:
            missing_tests.append(req["suggested_test"])

    # Add medium priority edge cases
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

    code_skeleton = get_current_code_skeleton()

    messages = [
        {"role": "system", "content": PROBLEM_ANALYSIS_SYSTEM_PROMPT.format(problem_statement=problem_statement)},
        {"role": "user", "content": f"# Code Skeleton:\n{code_skeleton}\n"},
    ]
    detailed_problem_analysis = {}
    while True:
        retry = 0

        while retry < 3:
            try:
                response = Network.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)
                response = response.replace("```json", "").strip("```").strip()
                detailed_problem_analysis = json.loads(response)

                is_valid, error_msg = validate_response(detailed_problem_analysis)
                if is_valid:
                    return detailed_problem_analysis
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Keep clarifying the problem analysis until you have a valid JSON object."})

            except Exception as e:
                pass

            retry += 1

        if retry >= 3:
            break

    if not detailed_problem_analysis:
        try:
            response = Network.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)
            response = response.replace("```json", "").strip("```").strip()
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

    code_skeleton = get_current_code_skeleton()
    start_time = time.time()
    initial_solution = generate_initial_solution(problem_statement, code_skeleton, detailed_problem_analysis)

    # Extract and write files from the solution
    created_files = extract_and_write_files(initial_solution)

    test_cases = generate_test_files(problem_statement, created_files, code_skeleton)

    # Extract and write files from test cases
    test_files = extract_and_write_files(test_cases)

    timeout = DEFAULT_TIMEOUT - (time.time() - start_time) - 60

    patch = fix_task_solve_workflow(
        problem_statement,
        timeout=timeout,
        run_id_1=run_id,
        test_runner=f"unittest",
        test_runner_mode="FILE",
        n_max_steps=60,
        enable_pev=enable_pev,
        enable_mcts=enable_mcts,
        extra_fix_request=SOLVE_TASK_NON_FUNCTIONAL_TEST_PROMPT,
    )

    if patch is None:
        extract_and_write_files(initial_solution)

    tool_manager = EnhancedToolManager()
    patch = tool_manager.get_final_git_patch()
    return patch


def fix_task_solve_workflow(
    problem_statement: str,
    *,
    timeout: int,
    run_id_1: str,
    test_runner: str = "pytest",
    test_runner_mode: str = "FILE",
    n_max_steps=MAX_FIX_TASK_STEPS,
    enable_pev: bool = True,
    enable_mcts: bool = True,
    extra_fix_request="",
) -> tuple[str, List[str], List[str]]:
    global run_id
    run_id = run_id_1

    pev = AgentPlanExecuteVerifyWorkflow(enable_pev=enable_pev, enable_mcts=enable_mcts)

    strategy = pev.run_planning_phase(problem_statement)
    mcts_path = pev.run_mcts_exploration(problem_statement)

    strategy_guidance = f"\n\nStrategic Plan: {strategy.get('name', 'Default')} - {strategy.get('description', 'Standard approach')}"
    mcts_guidance = f"\n\nMCTS Recommended Path: {' -> '.join(mcts_path[:5])}" if mcts_path else ""

    cot = EnhancedCOT(latest_observations_to_keep=30)
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "get_file_content",
            "save_file",
            "get_approval_for_solution",
            "search_in_all_files_content",
            "search_in_specified_file_v2",
            "list_directory",
            "get_context_around_line",
            "run_repo_tests",
            "run_code",
            "apply_code_edit",
            "generate_test_function",
            "get_project_metadata",
            "finish",
        ],
        test_runner=test_runner,
        test_runner_mode=test_runner_mode,
    )

    # Initialize phase manager for complex problems
    phase_manager = PhaseManager(problem_statement, n_max_steps)
    use_multi_phase = phase_manager.use_multi_phase_workflow()

    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(), format_prompt=FORMAT_PROMPT_V0, extra_fix_request=extra_fix_request
    )

    enhancement = enhance_problem_statement(problem_statement)
    enhanced_problem = problem_statement
    if enhancement:
        enhanced_problem = problem_statement + "\n\n---\n\n# Enhanced Problem Analysis\n\n" + enhancement

    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=enhanced_problem) + strategy_guidance + mcts_guidance

    start_time = time.time()
    logs: List[str] = []
    logs.append(f"cwd: {os.getcwd()}")

    for step in range(n_max_steps):

        if use_multi_phase and step > 0:
            should_transition, new_phase = phase_manager.should_transition(step, cot)
            if should_transition:
                phase_manager.transition_to_phase(new_phase, step)

        if time.time() - start_time > timeout:
            cot.add_action(
                EnhancedCOT.Action(
                    next_thought="global timeout reached",
                    next_tool_name="",
                    next_tool_args={},
                    observation="",
                    is_error=True,
                    inference_error_counter={},
                    request_data=[],
                )
            )
            break

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]

        messages.extend(cot.to_str())

        # Add phase-specific guidance if using multi-phase workflow
        if use_multi_phase:
            phase_guidance = phase_manager.get_phase_guidance()
            messages.append({"role": "system", "content": phase_guidance})

        messages.append({"role": "system", "content": STOP_INSTRUCTION})

        temperature = 0
        selected_model = GLM_MODEL_NAME
        if cot.is_thought_repeated():

            last_thought = cot.thoughts[-1]
            messages.append(
                {
                    "role": "user",
                    "content": DO_NOT_REPEAT_TOOL_CALLS.format(
                        previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
                    ),
                }
            )

            if cot.repeated_thoughts > 1:
                temperature = min(cot.repeated_thoughts / 10, 0.7)
                selected_model = AGENT_MODELS[random.randint(0, len(AGENT_MODELS) - 1)] if cot.repeated_thoughts > 2 else GLM_MODEL_NAME

        try:
            next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = Network.inference(
                messages, model=selected_model, run_id=run_id, temperature=temperature
            )
        except Exception as e:
            import traceback  # Ensure traceback is accessible

            error_msg = f"\n\nERROR: {repr(e)} {traceback.format_exc()}"

            if "too long context" in error_msg:
                cot.clear_and_restart()
            cot.add_action(
                EnhancedCOT.Action(
                    next_thought=error_msg,
                    next_tool_name="",
                    next_tool_args={},
                    observation="",
                    is_error=True,
                    raw_response=raw_text,
                    total_attempts=total_attempts,
                ),
                inference_error_counter=error_counter,
                request_data=messages,
            )
            break

        try:

            if '"' in next_tool_name or "'" in next_tool_name:
                next_tool_name = next_tool_name.replace('"', "")
                next_tool_name = next_tool_name.replace("'", "")

            next_observation = tool_manager.get_tool(next_tool_name)(**next_tool_args) if next_tool_args else tool_manager.get_tool(next_tool_name)()

            cot.add_action(
                EnhancedCOT.Action(
                    next_thought=next_thought,
                    next_tool_name=next_tool_name,
                    next_tool_args=next_tool_args,
                    observation=next_observation,
                    is_error=False,
                    raw_response=raw_text,
                    total_attempts=total_attempts,
                    inference_error_counter=error_counter,
                    request_data=messages,
                )
            )

            # Create checkpoint after key successful actions
            if use_multi_phase and next_tool_name in ["run_repo_tests", "apply_code_edit", "get_approval_for_solution"]:
                # Extract test results if available
                test_results = {}
                if "passed" in str(next_observation).lower() or "failed" in str(next_observation).lower():
                    # Simple parsing of test results
                    obs_str = str(next_observation)
                    test_results["observation"] = obs_str[:200]  # First 200 chars

                phase_manager.create_checkpoint(step, test_results)

            if enable_pev and enable_mcts and pev.mcts:
                success = "error" not in str(next_observation).lower()
                pev.mcts.update_root(next_tool_name, str(next_observation), success)
        except EnhancedToolManager.Error as e:
            import traceback  # Ensure traceback is accessible

            error_msg = f"observation: {e.message}"

            cot.add_action(
                EnhancedCOT.Action(
                    next_thought=next_thought,
                    next_tool_name=next_tool_name,
                    next_tool_args=next_tool_args,
                    observation=error_msg,
                    is_error=True,
                    raw_response=raw_text,
                    total_attempts=total_attempts,
                    inference_error_counter=error_counter,
                    request_data=messages,
                )
            )

            if enable_pev and enable_mcts and pev.mcts:
                pev.mcts.update_root(next_tool_name, error_msg, False)
            continue
        except Exception as e:
            import traceback  # Ensure traceback is accessible

            error_traceback = traceback.format_exc()
            if isinstance(e, TypeError):
                error_msg = f"observation: {str(e)}"
            else:
                error_msg = f"observation: {repr(e)} {error_traceback}"

            cot.add_action(
                EnhancedCOT.Action(
                    next_thought=next_thought,
                    next_tool_name=next_tool_name,
                    next_tool_args=next_tool_args,
                    observation=error_msg,
                    is_error=True,
                    raw_response=raw_text,
                    total_attempts=total_attempts,
                    inference_error_counter=error_counter,
                    request_data=messages,
                )
            )

            if enable_pev and enable_mcts and pev.mcts:
                pev.mcts.update_root(next_tool_name, error_msg, False)
            continue

        if next_tool_name == "finish":

            break

    else:
        cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached", next_tool_name="", next_tool_args={}, observation="", is_error=True))
        if n_max_steps < MAX_FIX_TASK_STEPS:
            return None

    # Log phase summary if using multi-phase workflow
    if use_multi_phase:

        for phase_info in phase_manager.phase_history:
            phase_name = phase_info["phase"]
            steps_used = phase_info["steps_used"]
            allocated = phase_manager.step_allocation.get(phase_name, 0)
            efficiency = (steps_used / allocated * 100) if allocated > 0 else 0

        # Log current phase if workflow didn't complete all phases
        current_phase = phase_manager.current_phase
        steps_in_current = step - phase_manager.phase_start_step
        allocated_current = phase_manager.step_allocation.get(current_phase, 0)

        if steps_in_current > 0:
            efficiency_current = (steps_in_current / allocated_current * 100) if allocated_current > 0 else 0

        # Log which phases were completed
        completed_phases = [p["phase"] for p in phase_manager.phase_history]

    patch = tool_manager.get_final_git_patch()

    return patch


def get_current_code_skeleton() -> str:
    # Initialize the result string
    result = ""

    # Walk through the current directory
    for root, _, files in os.walk("."):
        for file in files:
            # Check if the file is a Python file
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                # Concatenate the file name and content
                result += f"{file}\n{{\n{content}\n}}\n\n"

    return result


def get_directory_tree(start_path: str = ".") -> str:

    tree_lines = []

    def add_directory_tree(path: str, prefix: str = "", is_last: bool = True, is_root: bool = False):
        """Recursively build the tree structure"""
        try:
            # Get the directory name
            dir_name = os.path.basename(path) if path != "." else os.path.basename(os.getcwd())

            # Add current directory to tree (skip for root directory)
            if not is_root:
                connector = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{connector}{dir_name}/")

            # Get all items in directory
            try:
                items = os.listdir(path)
                # Filter out hidden directories and files starting with '.'
                items = [item for item in items if not item.startswith(".")]
                items.sort()

                # Separate directories and files
                dirs = []
                files = []
                for item in items:
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        dirs.append(item)
                    else:
                        files.append(item)

                # Process directories first
                for i, dir_name in enumerate(dirs):
                    dir_path = os.path.join(path, dir_name)
                    is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                    new_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
                    add_directory_tree(dir_path, new_prefix, is_last_dir, False)

                # Then process files
                for i, file_name in enumerate(files):
                    is_last_file = i == len(files) - 1
                    connector = "└── " if is_last_file else "├── "
                    tree_lines.append(f"{prefix}{'' if is_root else ('    ' if is_last else '│   ')}{connector}{file_name}")

            except PermissionError:
                # Handle directories we can't read
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
        for readme_name in ["README.md", "README.rst"]:
            readme_path = os.path.join(current_dir, readme_name)
            if os.path.exists(readme_path):
                return readme_path
        if current_dir == repo_path:
            break
        current_dir = os.path.dirname(current_dir)

    return None


def find_test_runner(readme_file_path: Optional[str] = None):
    FIND_TEST_RUNNER_PROMPT = textwrap.dedent(
        """\
        You are a helpful assistant that can find the test runner for a given repository.
        - The test runner is the file that can run the individual test files and test cases. (e.g. pytest, unittest, etc.)
        - Do not use the test runner to run test for whole repository or test setup.
        - Read the README file and find the test runner. If there is no test runner, return pytest.
        - Output format should be as the following. No other texts are allowed.
        abc/test.py
        """
    )
    if not readme_file_path:
        return "pytest"
    try:
        with open(readme_file_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        response = Network.make_request(
            [{"role": "system", "content": FIND_TEST_RUNNER_PROMPT}, {"role": "user", "content": readme_content}], model=DEEPSEEK_MODEL_NAME
        )
        return response.strip() or "pytest"
    except Exception as e:

        return "pytest"


def convert_filepath_to_module(file_path: str, repo_path: str, test_runner: str) -> str:
    """Convert file path to Python module notation."""
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)

    # Remove extension and make relative to repo
    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path) :].lstrip(os.path.sep)

    # Adjust relative to test runner directory if needed
    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir) :].lstrip(os.path.sep)

    return module_path.replace(os.path.sep, ".")


def purge_filepath(file_path: str, repo_path: str, test_runner: str) -> str:
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)

    module_path = os.path.splitext(abs_filepath)[0]
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path) :].lstrip(os.path.sep)

    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir) :].lstrip(os.path.sep)

    return module_path


def determine_test_runner_mode(test_runner: str):
    if test_runner == "pytest":
        return "FILE"

    try:
        with open(test_runner, "r", encoding="utf-8") as f:
            runner_content = f.read()

        response = Network.make_request(
            [{"role": "system", "content": TEST_RUNNER_MODE_PROMPT}, {"role": "user", "content": runner_content}], model=DEEPSEEK_MODEL_NAME
        )
        return response.strip() or "FILE"
    except Exception as e:

        return "FILE"


def count_test_cases(file_path: str) -> int:
    """Count the number of test cases (functions starting with 'test_') in a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        import re

        test_functions = re.findall(r"^\s*def\s+test_\w+", content, re.MULTILINE)
        return len(test_functions)

    except (FileNotFoundError, UnicodeDecodeError):
        return 0


def determine_test_runner_and_mode():
    test_runner = "pytest"
    test_runner_mode = "FILE"
    test_files = []  # Initialize the test_files list
    test_file_path = None

    for root, _, files in os.walk("."):
        for file in files:
            if "test_" in file and file.endswith(".py"):
                test_files.append(os.path.join(root, file))

    test_files.sort(key=len)

    for path in test_files:
        if count_test_cases(path) > 5:
            test_file_path = path
            break

    if not test_file_path:
        return "pytest", "FILE"

    readme_file_path = find_readme(test_file_path, ".")
    if readme_file_path:
        test_runner = find_test_runner(readme_file_path)
        test_runner_mode = determine_test_runner_mode(test_runner)

    return test_runner, test_runner_mode

