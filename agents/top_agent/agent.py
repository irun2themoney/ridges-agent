from __future__ import annotations

import ast
import csv
import hashlib
import inspect
import json
import math
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

import requests

run_id = None

STOP_INSTRUCTION = textwrap.dedent(
    """
    # 🎨 
    DO NOT generate `observation:` in your response. It will be provided by user for you.
    Generate only SINGLE triplet of `next_thought`, `next_tool_name`, `next_tool_args` in your response.
    """
)

DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2100"))

PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"

GLM_MODEL_NAME = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS = [GLM_MODEL_NAME, KIMI_MODEL_NAME, DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME]
MAX_FIX_TASK_STEPS = 400

# Multi-phase workflow configuration
PHASE_INVESTIGATION = "investigation"
PHASE_PLANNING = "planning"
PHASE_IMPLEMENTATION = "implementation"
PHASE_VALIDATION = "validation"

DO_NOT_REPEAT_TOOL_CALLS = textwrap.dedent(
    """
    You're not allowed to repeat the same tool call with the same arguments.
    Your previous response: 
    {previous_response}
    
    Try to use something different!
    """
)

GENERATE_INITIAL_SOLUTION_PROMPT = textwrap.dedent(
    """
    You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.
    
    Strict Requirements:
    1. Output the full content of Python files along with their file names.
    2. Do not include explanations, comments, or markdown formatting in the main code.
    3. Use only standard Python (no external libraries).
    4. Implement all required classes and functions exactly with the same names as in the initial code stub.
    5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
    6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
    7. The solution must be executable as-is with no placeholders or TODOs.
    8. **IMPORTANT**: Add clear comments above each edge case handling section to identify which specific edge case is being addressed. Use the format: `# Edge Case: [description of the edge case]`
    9. **IMPORTANT**: Add a comment at the end of each function/class that lists all edge cases handled, using the format: `# Handled Edge Cases: [list of edge cases]`
    
    Return only the final Python code.
    """
)

GENERATE_TESTCASES_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
    """
    You are an expert Python unittest testcase developer. 
        Important points:-
        - you have generation limit of 2048 tokens. Hence you must stop generating more test cases when you are near the limit.
        - If you get syntax error, check if last assistant response was truncated. If yes, then skip last couple of test cases to fit in.
        
        You must respond directly with the test cases in the following format. 
        =========TEST_CASES
        <<test cases>>
        Do not include anything else. For Example:
        =========TEST_CASES
        # These tests are auto-generated with test data from:
        # https://github.com/xxxx.json
        # File last updated on 2023-07-19
        import unittest
        from main_module import (
            main_func
        )
    
        class TestFuncA(unittest.TestCase):
            def test_main_func(self):
                self.assertEqual(main_func(), "expected_output")
        if __name__ == "__main__":
            unittest.main()
    """
)

GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
    """
    You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.
    
    Strict Requirements:
    1. Output the full content of Python files along with their file names. You **MUST** output the **file name** along with file content.
    2. Do not include explanations, comments, or markdown formatting in the main code.
    3. Use only standard Python (no external libraries).
    4. Implement all required classes and functions exactly with the same names as in the initial code stub.
    5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
    6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
    7. The solution must be executable as-is with no placeholders or TODOs.
    8. If problem statement doesn't explicitely requires a list of strings as a response, do not use list of strings for multiline text problems, just use raw string format.
    9. **IMPORTANT**: Add clear comments above each edge case handling section to identify which specific edge case is being addressed. Use the format: `# Edge Case: [description of the edge case]`
    10. **IMPORTANT**: Add a comment at the end of each function/class that lists all edge cases handled, using the format: `# Handled Edge Cases: [list of edge cases]`
    
    Return only the final Python code.
    
    Response Examples:
    ```python
    a.py
    {content}
    
    b.py
    {content}
    ```
    """
)

FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""# Hey there! You're a Coding Assistant 🚀. You'll be given a problem statement and need to fix the issue. Steps: 1. Find relevant files. 2. Locate issue. 3. Edit code. 4. Handle edgecases. 5. Ensure backward compatibility. 6. Check entire codebase for exhaustive changes. 7. Limit changes to requested ones. 8. Use test generation tool, never edit existing test files. 9. Don't create new files unless necessary. 10. Check impacted tests. 11. Propose 2 different solutions. 12. Check expected output AND test output. 13. Don't solve missing dependencies. Multi-file awareness: Search all files, apply consistent changes, use context tools. Test generation: Find FAIL_TO_PASS test file, use generate_test_function. Generated tests excluded from patch. {extra_fix_request} Tools: {tools_docs} {format_prompt}""")

SOLVE_TASK_NON_FUNCTIONAL_TEST_PROMPT = textwrap.dedent(
    """
    
    # How to handle incorrect test data:
    If a few testcases keeps fail in several steps with different approaches, double check test data is correct and matched with the given problem statement.
    If you are sure those testcases are incorrect or mismatching, find the best fix based on following best practice guide.
    
    # Best Practice Code Fix Guide:
    1. Use inheritance to factor out shared behavior into a base class or base function instead of duplicating code across subclasses or functions.
    2. Add clear and complete docstrings for every function or method, especially for externally called functions defined in the code skeleton.
       - Describe expected parameter types(including function types for function passing parameters), return values, and calling behavior.
       - Generate docstrings as small as possible with all explicit info to avoid generation limit
    3. Consistency Rules
       - Verify consistency through docstrings and if you find any in-consistency generate 1~3 testcases to check the solutions works consistently through out the entire solution
       - Fix docstrings and code implementation based on fixed docstrings
    4. Security Rules
       - Verify Security Best Practice in solutions such as weakness of solutions imported modules, weakness of implemented algorithms, etc and improve
       - Generate 1~3 testcases that can verify if the imported module's ability is limited to the requirement technically
       - Generate 1~3 testcases that can verify if the implemented algorithms to solve the problem has weakness technically and logically
       - run testcases and improve the solution to be more safe by improving imported module's limitation by an well-known method or improving algorithm with better one.
    5. Multi-Type Support Rules
       - If problem statement and code skeleton doesn't mention explicit types of parameter, prototypes of function passing or etc, try to update docstrings to be more flexible for several possible types
       - Generate testcases to verify the code works for flexible types and fix the solution if it doesn't work for different typesmproving algorithm with better one.
    
    """
)

FIX_TASK_NEVER_EARLY_STOP_PROMPT = textwrap.dedent(
    """
    
    # Prevent Early Stop:
    As this is complex project and the issue is also quite complex so never early stop unless 
    - you found exact FAIL_TO_PASS test file and you fixed all failures all of them 
    - and there isn't any side-effects from your fix like other PASS_TO_PASS tests may fail
    If time it not out, double confirm above in several steps with different thoughts.
    
    """
)

FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent(
    """
    # Now let's start. Here is the problem statement:
    {problem_statement}
    """
)

TEST_RUNNER_MODE_PROMPT = textwrap.dedent("""You are a helpful assistant that determines test runner mode. Read test runner file and determine if it requires MODULE or FILE path. Output only MODULE or FILE. MODULE: requires module path. FILE: requires file path (e.g. pytest, unittest).""")

FORMAT_PROMPT_V0 = textwrap.dedent("""You must respond in this format: next_thought: <your thought> next_tool_name: <tool name> next_tool_args: <tool arguments as JSON>""")

class EnhancedCOT:
    class Action:
        def __init__(
            self,
            next_thought: str,
            next_tool_name: str,
            next_tool_args: dict,
            observation: list | tuple | str,
            is_error: bool = False,
            raw_response: str = None,
            total_attempts: int = 0,
            inference_error_counter: dict = None,
            request_data: list = None,
        ):
            self.next_thought = next_thought
            self.next_tool_name = next_tool_name
            self.next_tool_args = next_tool_args
            self.observation = ";".join(observation) if isinstance(observation, list) else observation
            self.is_error = is_error
            self.raw_response = raw_response
            self.total_attempts = total_attempts
            self.inference_error_counter = inference_error_counter
            self.request_data = request_data
            self.is_deleted = False
    def __init__(self, latest_observations_to_keep=5):
        self.thoughts: list[EnhancedCOT.Action] = []
        self.latest_observations_to_keep = latest_observations_to_keep
        self.repeated_thoughts = 0
    def add_action(self, action: EnhancedCOT.Action) -> bool:  # don't add if thought is repeated
        self.thoughts.append(action)
        return True
    def clear_and_restart(self):
        self.thoughts = self.thoughts[int(len(self.thoughts) / 2) :]
        self.repeated_thoughts = 0
    def is_thought_repeated(self) -> bool:
        if len(self.thoughts) < 2:
            self.repeated_thoughts = 0
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            self.repeated_thoughts += 1
            return True
        self.repeated_thoughts = 0
        return False
    def to_str(self):
        msgs, n = [], len(self.thoughts)
        for i, t in enumerate(self.thoughts):
            if t.is_deleted:
                continue
            asstr = f"next_thought:{t.next_thought}\nnext_tool_name:{t.next_tool_name}\nnext_tool_args:{t.next_tool_args}"
            if i < n - self.latest_observations_to_keep:
                obs_len = 0 if t.observation is None else len(t.observation) if isinstance(t.observation, (list, tuple)) else len(str(t.observation).splitlines())
                usstr = f"observation: {'error ocurred. ' if t.is_error else ''}output omitted ({obs_len}) lines\n"
            else:
                if t.is_error is None or i == n - 1:
                    obs_r = json.dumps(list(t.observation), ensure_ascii=False) if isinstance(t.observation, (list, tuple)) else str(t.observation)
                    try:
                        obs_r = json.dumps(list(t.observation), ensure_ascii=False) if isinstance(t.observation, (list, tuple)) else str(t.observation)
                    except:
                        obs_r = str(t.observation)
                    usstr = f"observation: {obs_r}"
                else:
                    if self.thoughts[-1].is_error == None and t.is_error != None:
                        obs_len = 0 if t.observation is None else len(t.observation) if isinstance(t.observation, (list, tuple)) else len(str(t.observation).splitlines())
                        usstr = f"observation: error ocurred. detailed output omitted ({obs_len}) lines\n"
                    else:
                        try:
                            obs_r = json.dumps(list(t.observation), ensure_ascii=False) if isinstance(t.observation, (list, tuple)) else str(t.observation)
                        except:
                            obs_r = str(t.observation)
                        usstr = f"observation: {obs_r}"
            msgs.append({"role": "assistant", "content": asstr})
            msgs.append({"role": "user", "content": usstr})
        return msgs
class EnhancedToolManager:
    logs = []
    TOOL_LIST = {}
    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR = 1
            RUNTIME_ERROR = 2
            TIMEOUT = 3
            FILE_NOT_FOUND = 4
            SEARCH_TERM_NOT_FOUND = 5
            UNKNOWN = 6
            THIRD_PARTY_DEPENDENCIES = 7
            MULTIPLE_SEARCH_RESULTS_FOUND = 8
            BUG_REPORT_REQUIRED = 9
            INVALID_RESPONSE_FORMAT = 10
            INVALID_TOOL_NAME = 11
            INVALID_FILE_PATH = 12
            INVALID_TOOL_CALL = 13
            IMPORT_ERROR = 14
        def __init__(self, error_type: ErrorType, message: str):
            self.error_type = error_type
            self.message = message
    def tool(fn):
        def wrapper(self, *args, **kwargs):
            self.tool_invocations[fn.__name__] += 1
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                self.tool_failure[fn.__name__][e.error_type] += 1
                return e.message
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool = True

        return wrapper

    def __init__(self, **kwargs):
        pass
    @classmethod
    def tool_parsing(cls, fn):
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        doc = doc_fn.split("Arguments:")[0]
        output_description = doc_fn.split("Output:")
        if len(output_description) > 1:
            output_description = "Output: " + output_description[1].strip()
            doc = doc + "\n\n" + output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == "self":
                continue
            if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description = re.search(f"{param.name}:([^\n]+)", doc_fn)
            if param_description:
                param_description = param_description.group(1)
            else:
                raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}")
            if ("list" in type_hint.lower()) and ("str" in type_hint):
                properties[param.name] = {"type": "array", "items": {"type": "string"}, "description": param_description}
                continue
            elif "str" in type_hint:
                json_type = "string"
            elif "int" in type_hint:
                json_type = "integer"
            elif "float" in type_hint:
                json_type = "number"
            elif "bool" in type_hint:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[param.name] = {"type": json_type, "description": param_description}
        parameters = {"type": "object", "properties": properties, "required": required}
        tool_schemas = {"name": name, "description": doc.strip(), "input_schema": parameters}
        return tool_schemas

    @classmethod
    def get_tool_args_for_tool(self, tool_name: str, required_only: bool = False) -> list[str]:
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        if not required_only:
            return list(self.TOOL_LIST[tool_name]["input_schema"]["properties"].keys())
        else:
            return self.TOOL_LIST[tool_name]["input_schema"]["required"]
    def get_tool_docs(self) -> str:
        return "\n\n".join([json.dumps(tool_metadata, ensure_ascii=False) for _, tool_metadata in self.TOOL_LIST.items()])
    def get_tool(self, tool_name: str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist. Please use one of the following tools: {', '.join(self.TOOL_LIST.keys())}"
        return tool_method

    def _check_syntax_error(self, content: str, file_path: str = "<unknown>") -> bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            return True, EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, f"Syntax error. {str(e)}")

    def _save(self, file_path: str, content: str) -> str:
        is_syntax_error, error = self._check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            return f"File {file_path} saved successfully"
        else:
            error.message = "Error saving file. " + error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, error.message)

    def get_final_git_patch(self) -> str:
        """
        Generates git diff patch containing all modifications in working directory
        Useful for capturing comprehensive change summary before finalization
        """
        try:
            command = f"""
            shopt -s globstar
            cp .gitignore .gitignore.backup 2>/dev/null || true
            echo 'src/agent.py' >> .gitignore
            echo 'src/agent_runner.py' >> .gitignore

            git add **/*.py 2>/dev/null || true
            git add **/*.toml 2>/dev/null || true
            git add **/*.cfg 2>/dev/null || true
            git add **/*.txt 2>/dev/null || true

            git diff --cached > .patch.txt
            cat .patch.txt

            mv .gitignore.backup .gitignore 2>/dev/null || true
            """

            output = subprocess.run(["bash", "-c", command], timeout=30, capture_output=True, text=True)

            return output.stdout
        except Exception as e:

            return f"Error generating git patch: {e}"

class Network:
    class ErrorType(Enum):
        EMPTY_RESPONSE = 1
        RESERVED_TOKEN_PRESENT = 2
        RATE_LIMIT_EXCEEDED = 3
        INVALID_RESPONSE_FORMAT = 4
        TIMEOUT = 5
        UNKNOWN = 6
        NETWORK_ERROR = 7
        AUTHENTICATION_ERROR = 8
        RESOURCE_EXHAUSTED = 9
    @classmethod
    def is_valid_response(cls, raw_text: str) -> bool:
        if type(raw_text) is dict and raw_text.get("error", None) is not None and raw_text.get("error") != "":
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if not raw_text.strip().endswith("}") and not raw_text.strip().endswith("}]"):
            return False, "Incomplete response, your response must be shorter to fit within context limit"
        if len(raw_text) == 0:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        if "API request failed with status 429" in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if "Read timed out" in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if "Network unreachable" in raw_text or "Connection refused" in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        return True, None
    @classmethod
    def get_error_counter(cls) -> dict[str, int]:
        return {k: 0 for k in cls.ErrorType.__members__}
    @classmethod
    def fix_json_string_with_llm(cls, json_string: str, attempt: int = 0) -> dict:
        messages = [
            {"role": "system", "content": "Fix the json string sent by the user.  Reply only with the json string and nothing else."},
            {"role": "user", "content": json_string},
        ]
        response = cls.make_request(messages, model=DEEPSEEK_MODEL_NAME)
        try:
            response = response.replace("```json", "").strip("```")
            response = json.loads(response)
            return response
        except JSONDecodeError as e:
            return None

    @classmethod
    def make_request(
        cls,
        messages: list,
        model: str,
        attempt: int = 0,
        temperature: float = 0.0,
        top_p: float = 1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_retries: int = 5,
    ) -> str:
        global run_id
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"
        request_data = {
            "run_id": run_id if run_id else str(uuid4()),
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

        headers = {"Content-Type": "application/json"}
        request_data["model"] = model

        for retry_attempt in range(max_retries + 1):
            try:
                response = requests.post(url, data=json.dumps(request_data), timeout=120, headers=headers)
                response.raise_for_status()
            except requests.exceptions.Timeout:
                return f"ERROR: Request timeout for model {model}"
            except requests.exceptions.ConnectionError as e:
                return f"ERROR: Connection failed for model {model}"
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                if status_code in [500, 504] and retry_attempt < max_retries:
                    sleep_time = 2 * retry_attempt
                    time.sleep(sleep_time)
                    continue
                return f"ERROR: HTTP error {status_code} for model {model}"
            except requests.exceptions.RequestException as e:
                return f"ERROR: Request failed for model {model}"

            try:
                response_json = response.json()
            except JSONDecodeError as e:
                return f"ERROR: Invalid JSON response for model {model}"

            try:
                is_oai_interface = (
                    type(response_json) is dict
                    and response_json.get("choices") is not None
                    and len(response_json.get("choices")) > 0
                    and response_json.get("choices")[0].get("message") is not None
                )
                if is_oai_interface:
                    raw_text = response_json["choices"][0]["message"]["content"]
                else:
                    if type(response_json) is str:
                        raw_text = response_json.strip("\n").strip()
                    else:
                        raw_text = response_json
                if type(raw_text) is not dict:
                    raw_text = raw_text.lstrip()
                return raw_text
            except (KeyError, IndexError, TypeError) as e:
                return f"ERROR: Invalid response structure for model {model}"
            except Exception as e:
                return f"ERROR: Unexpected error for model {model}"

        return f"ERROR: Max retries exceeded for model {model}"

    @classmethod
    def _request_next_action_with_retry(
        cls, messages: dict, model: str, max_retries: int = 5, base_delay: float = 1.0, temperature: float = 0.0
    ) -> str:
        raw_text = "not defined"
        error_counter = cls.get_error_counter()
        next_thought, next_tool_name, next_tool_args = None, None, None
        total_attempts = 0
        for attempt in range(max_retries):
            try:
                total_attempts += 1
                index = AGENT_MODELS.index(model) if model in AGENT_MODELS else -1
                raw_text = cls.make_request(messages, model=AGENT_MODELS[(index + attempt) % len(AGENT_MODELS)], temperature=temperature)
                is_valid, error_msg = cls.is_valid_response(raw_text)
                if not (is_valid):
                    raise Exception(error_msg)

                next_thought, next_tool_name, next_tool_args, error_msg = cls.parse_response(raw_text)
                if error_msg:
                    raise Exception(error_msg)
                break
            except Exception as e:
                error_body = str(e)

                if attempt < max_retries:
                    delay = base_delay

                    if "RATE_LIMIT_EXCEEDED" in error_body:
                        error_counter[cls.ErrorType.RATE_LIMIT_EXCEEDED.name] += 1
                    elif "RESERVED_TOKEN_PRESENT" in error_body:
                        error_counter[cls.ErrorType.RESERVED_TOKEN_PRESENT.name] += 1
                    elif "EMPTY_RESPONSE" in error_body:
                        error_counter[cls.ErrorType.EMPTY_RESPONSE.name] += 1
                    elif "TIMEOUT" in error_body:
                        error_counter[cls.ErrorType.TIMEOUT.name] += 1
                    elif "Invalid JSON" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name] += 1
                    elif "Invalid response" in error_body:
                        error_counter[cls.ErrorType.INVALID_RESPONSE_FORMAT.name] += 1
                    else:
                        error_counter[cls.ErrorType.UNKNOWN.name] += 1
                    if (
                        "RATE_LIMIT_EXCEEDED" not in error_body
                        and "RESERVED_TOKEN_PRESENT" not in error_body
                        and "EMPTY_RESPONSE" not in error_body
                        and "TIMEOUT" not in error_body
                    ):
                        messages.append({"role": "assistant", "content": raw_text})
                        messages.append({"role": "user", "content": "observation: " + error_body})
                    time.sleep(random.uniform(1.2 * delay, 1.5 * delay))
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name] += 1
                    raise RuntimeError(error_body)

        return next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages

    @classmethod
    def parse_malformed_json(cls, arguments: list[str], json_string: str) -> dict | str:
        pattern = ""
        for i, k in enumerate(arguments):
            pattern += f'"{k}": (.*)'
            if i != len(arguments) - 1:
                pattern += r",\s*"
        match = re.search(pattern, json_string)

        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"

        result_json = {}
        for i in range(len(arguments)):
            value = match.group(i + 1)
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            value = value.replace("\\n", "\n")
            result_json[arguments[i]] = value
        return result_json

    @classmethod
    def parse_next_tool_args(cls, tool_name: str, next_tool_args: str) -> dict | str:
        """
        parse string to json, fix unecaped " in values like this: '{"a": "text "text2" text3 "text4"", "b": "text3"}'
        returns json or error message
        """
        next_tool_args = next_tool_args.replace("```json", "").strip("```")
        error_msg = ""

        try:
            next_tool_args = Utils.load_json(next_tool_args.strip())
        except JSONDecodeError as e:
            error_msg = f"Invalid JSON: {next_tool_args}"
            try:
                next_tool_args = cls.parse_malformed_json(EnhancedToolManager.get_tool_args_for_tool(tool_name, required=True), next_tool_args)
            except EnhancedToolManager.Error as e:
                raise Exception(e.message)
            except Exception as e:
                raise Exception(error_msg)
        return next_tool_args

    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str = str(uuid4()), temperature: float = 0.0) -> dict:
        """Prod inference with caching"""
        cleaned_msgs: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            content = m.get("content", "")
            if role == "assistant" and not content.strip():
                continue

            cleaned_msgs.append({"role": role, "content": content})

        if not cleaned_msgs:
            raise RuntimeError("No valid messages to send to proxy.")

        next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages = cls._request_next_action_with_retry(
            cleaned_msgs, model=model, temperature=temperature
        )

        return next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter, messages

    @classmethod
    def sanitise_text_resp(cls, text_resp: str) -> str:
        text_resp = re.sub("['\"]*next_thought['\"]*:", "next_thought:", text_resp)
        text_resp = re.sub("['\"]*next_tool_name['\"]*:", "next_tool_name:", text_resp)
        text_resp = re.sub("['\"]*next_tool_args['\"]*:", "next_tool_args:", text_resp)
        text_resp = re.sub("['\"]*observation['\"]*:", "observation:", text_resp)
        if (
            "next_thought" not in text_resp
            and "next_tool_name:" in text_resp
            and "next_tool_args:" in text_resp
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")
            and text_resp.find("next_tool_name:") > 10
        ):
            text_resp = "next_thought: " + text_resp
        if (
            "next_tool_name:" in text_resp
            and "next_tool_args:" in text_resp
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")
        ):
            next_tool_name = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n").strip("'").strip('"').strip()
            text_resp = re.sub(f"next_tool_name:['\" ]*{next_tool_name}['\" ]*", "next_tool_name: " + next_tool_name, text_resp)

        return text_resp

    @classmethod
    def parse_response(cls, text_resp: str) -> tuple[str, Any, Any]:
        error_msg = None
        text_resp = text_resp.strip()
        text_resp = text_resp.split("observation:")[0]
        text_resp = text_resp.strip().strip("\n")
        text_resp = cls.sanitise_text_resp(text_resp)
        if (
            "next_thought:" in text_resp
            and "next_tool_name:" in text_resp
            and "next_tool_args:" in text_resp
            and text_resp.find("next_thought:") < text_resp.find("next_tool_name:")
            and text_resp.find("next_tool_name:") < text_resp.find("next_tool_args:")
        ):
            next_thought = text_resp.split("next_thought:")[1].split("next_tool_name:")[0].strip().strip("\n")
            next_tool_name_raw = text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n")
            next_tool_args_raw = text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip().strip("\n")
            try:
                if next_tool_name_raw.startswith("["):
                    next_tool_name = Utils.load_json(next_tool_name_raw)
                else:
                    next_tool_name = [next_tool_name_raw]
                parsed_args = cls.parse_next_tool_args(next_tool_name, next_tool_args_raw)
                if isinstance(parsed_args, list):
                    next_tool_args = parsed_args
                else:
                    next_tool_args = [parsed_args for _ in next_tool_name]
            except JSONDecodeError as e:
                error_msg = f"Invalid JSON: {str(e)}"
                Utils.log_to_failed_messages(text_resp)
        else:
            if "is longer than the model's context length" in text_resp:
                error_msg = "Invalid response. too long context"
            elif "next_thought:" not in text_resp:
                error_msg = "Invalid response. next_thought not found"
            elif "next_tool_name:" not in text_resp:
                error_msg = "Invalid response. next_tool_name not found"
            elif "next_tool_args:" not in text_resp:
                error_msg = "Invalid response. next_tool_args not found"
            elif text_resp.find("next_thought:") > text_resp.find("next_tool_name:"):
                error_msg = "Invalid response. next_thought is after next_tool_name"
            elif text_resp.find("next_tool_name:") > text_resp.find("next_tool_args:"):
                error_msg = "Invalid response. next_tool_name is after next_tool_args"

            Utils.log_to_failed_messages(text_resp)
            return None, None, None, error_msg

        if len(next_tool_name) == 1:
            return next_thought, next_tool_name[0], next_tool_args[0], error_msg

        return next_thought, next_tool_name, next_tool_args, error_msg

class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.functions = {}
        self.current_class = None
        self.class_hierarchy = []
        self.file_content = file_content
    def visit_ClassDef(self, node):
        self.class_hierarchy.append(node.name)
        self.current_class = "::".join(self.class_hierarchy)
        self.generic_visit(node)
        self.class_hierarchy.pop()
        self.current_class = "::".join(self.class_hierarchy) if self.class_hierarchy else None
    def _process_function(self, node):
        fn = f"{self.current_class}::{node.name}" if self.current_class else node.name
        ln = node.decorator_list[0].lineno if isinstance(node.decorator_list, list) and node.decorator_list else node.lineno
        eln = node.body[-1].lineno if isinstance(node.body, list) and node.body else ln
        body = "\n".join(self.file_content.split("\n")[ln-1:eln])
        self.functions[fn] = {"class": self.current_class, "body": body, "line_number": ln}
        self.generic_visit(node)
    def visit_FunctionDef(self, node):
        self._process_function(node)
    def visit_AsyncFunctionDef(self, node):
        self._process_function(node)
    def visit_Module(self, node):
        self.current_class = None
        self.generic_visit(node)
        self.current_class = None
class Utils:
    @classmethod
    def limit_strings(cls, strings: str, n=1000) -> str:
        sl = strings.split("\n")
        return "\n".join(sl[:n]) + f"\n...({len(sl) - n} more lines)" if len(sl) > n else strings
    @classmethod
    def load_json(cls, json_string: str) -> dict:
        try:
            return json.loads(json_string)
        except Exception as e:
            try:
                return eval(json_string)
            except Exception as e:
                fixed_json = Network.fix_json_string_with_llm(json_string)
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError("Invalid JSON", json_string, 0)
    @classmethod
    def log_to_failed_messages(cls, text_resp: str):
        with open("../failed_messages.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([text_resp])
class FixTaskEnhancedToolManager(EnhancedToolManager):
    def __init__(self, available_tools: Optional[list[str]] = [], test_runner: str = "pytest", test_runner_mode: str = "FILE"):
        self.new_files_created = []
        self.is_solution_approved = False
        self.test_runner = test_runner
        self.test_runner_mode = test_runner_mode
        self.generated_test_files = []
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if available_tools is not None and name not in available_tools:
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)

        self.tool_failure = {k: {j: 0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()}

        self.tool_invocations = {k: 0 for k in self.TOOL_LIST.keys()}

    def check_syntax_error(self, content: str, file_path: str = "<unknown>") -> bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            return True, EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, f"Syntax error. {str(e)}")

    def _get_file_content(
        self, file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None, limit: int = 5000
    ) -> str:
        if search_term is not None and search_term != "":
            return self.search_in_specified_file_v2(file_path, search_term)

        func_ranges = self.get_function_ranges(file_path)
        if search_start_line != None:
            for start, end, name in func_ranges:
                if start <= search_start_line <= end:
                    if start < search_start_line:

                        search_start_line = start
        if search_end_line != None:
            for start, end, name in func_ranges:
                if start <= search_end_line <= end:
                    if end > search_end_line:

                        search_end_line = end

        with open(file_path, "r") as f:
            if search_start_line is not None or search_end_line is not None:
                lines = f.readlines()
                start = max(0, (search_start_line or 1) - 1)
                end = min(len(lines), search_end_line or len(lines))
                content = "".join(lines[start:end])
                return f"Lines {start + 1}-{end} of {file_path}:\n{content}"
            else:
                content = f.read()

        return Utils.limit_strings(content, n=limit) if limit != -1 else content

    @EnhancedToolManager.tool
    def get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None) -> str:
        """
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file. This file must be python file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        """
        return self._get_file_content(file_path, search_start_line, search_end_line, search_term, limit=5000)
    @EnhancedToolManager.tool
    def save_file(self, file_path: str, content: str) -> str:
        """
        Writes text content to specified filesystem location. If there are any syntax errors in the code, it rejects the edit with an error message. Do not use this tool to create test or files to reproduce the error.
        Arguments:
            file_path: target filesystem path
            content: text data to write
        """
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: You cannot use this tool to create test or files to reproduce the error.",
            )
        return self._save(file_path, content)
    @EnhancedToolManager.tool
    def get_approval_for_solution(self, solutions: list[str], selected_solution: int, reason_for_selection: str) -> str:
        """
        This tool is used to get approval for your proposed solution. You need to propose at least 2 meaningfully different and elegant solutions to the problem.
        While all the solutions proposed needs to be accurate, but following are guidelines for selecting the best solution:
        1. Expected output should be closest to the most relevant test case.
        Arguments:
            solutions: list of solutions proposed by you. Here each solution individually should be very detailed and then must explain why they are better than the other solutions.
            selected_solution: Index of the solution you think is the best.
            reason_for_selection: Reason for selecting the solution over other solutions.
        Output:
            approval: approved/not approved. If approved, you can go ahead and implement the solution.
        """

        parsed_solutions = []
        for solution in solutions:
            sols = re.split(r"(Solution \d+:)", solution)
            sols = [f"{sols[i]}{sols[i + 1]}" for i in range(1, len(sols), 2)]
            parsed_solutions.extend(sols)

        solutions = parsed_solutions

        if type(solutions) is not list or len(solutions) < 2:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name, f"Error: solutions must be a list with length at least 2."
            )

        self.is_solution_approved = True
        return "Approved"

    def _save(self, file_path: str, content: str) -> str:
        is_syntax_error, error = self.check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            error.message = "Error saving file. " + error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, error.message)

    @EnhancedToolManager.tool
    def search_in_all_files_content(self, search_term: str, case_sensitive: bool = False) -> str:
        """
        Search for a text pattern across all .py files in the project, excluding any file with "test" in its path.
        Use at the beginning of the workflow to locate all possible references to a function, class, or variable.
        Arguments:
            search_term: text pattern to locate (e.g., "def test_function", "*SomeClass*")
            case_sensitive: flag to determine if the search should be case-sensitive
        Output:
            locations where pattern was found with file paths and line numbers
        """
        output = []
        search_flags = 0 if case_sensitive else re.IGNORECASE

        for root, _, files in os.walk("."):
            # Skip .git and docs directories
            if ".git" in root or "docs" in root:
                continue

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)

                    # Always check if search term is in the file name
                    if re.search(search_term, file_path, search_flags):
                        output.append(f"{file_path} | Filename match")

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        if not re.search(search_term, content, search_flags):
                            continue

                        # Parse the file content using AST
                        tree = ast.parse(content, filename=file_path)
                        visitor = FunctionVisitor(content)
                        visitor.visit(tree)

                        for function_name, function_info in visitor.functions.items():
                            body = function_info["body"]
                            if re.search(search_term, body, search_flags):
                                lines = body.split("\n")
                                for idx, line in enumerate(lines):
                                    if re.search(search_term, line, search_flags):
                                        line_number = function_info["line_number"] + idx
                                        output.append(f"{file_path}:{line_number} | {function_name} | {line.rstrip()}")
                    except Exception as e:
                        pass

        output = Utils.limit_strings("\n".join(output), n=100)
        if not output:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name, f"'{search_term}' not found in the codebase."
            )
        return output

    def get_function_ranges(self, file_path: str) -> list[tuple[int, int, str]]:
        # Try to parse the file to map lines to their enclosing functions.
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name, f"Error reading '{file_path}': {e}")
        try:
            tree = ast.parse("\n".join(source_lines), filename=file_path)
        except SyntaxError as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, f"Error parsing '{file_path}': {e}, {traceback.format_exc()}"
            )
            tree = None  # Fallback if file cannot be parsed.
        func_ranges: list[tuple[int, int, str]] = []  # (start, end, name)
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = getattr(node, "lineno", None)
                    end = getattr(node, "end_lineno", None)
                    if start is not None and end is not None:
                        func_ranges.append((start, end, node.name))
        return func_ranges

    def _extract_function_matches(self, file_path: str, search_term: str, *, max_output_lines: int = 1000) -> str:
        """
        Return the source code of any function definitions that contain `search_term`.
        If a match occurs outside of a function, only that line is returned. The final
        output is truncated with `limit_strings` to avoid excessive verbosity.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name, f"Error reading '{file_path}': {e}")

        # Identify all lines that contain the search term.
        match_lines = [idx + 1 for idx, line in enumerate(source_lines) if search_term in line]
        if not match_lines:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name, f"'{search_term}' not found in file '{file_path}'"
            )

        func_ranges = self.get_function_ranges(file_path)

        def _containing_function(line_no: int):
            for start, end, name in func_ranges:
                if start <= line_no <= end:
                    return (start, end, name)
            return None
        functions_to_return: list[tuple[int, int, str]] = []
        standalone_lines: list[int] = []
        for ln in match_lines:
            info = _containing_function(ln)
            if info and info not in functions_to_return:
                functions_to_return.append(info)
            elif not info:
                standalone_lines.append(ln)

        chunks: list[str] = []
        for start, end, name in functions_to_return:
            func_src = "\n".join(source_lines[start - 1 : end])
            chunks.append(f"(lines {start}-{end}):\n{func_src}")

        for ln in standalone_lines:
            chunks.append(f"{ln}:{source_lines[ln - 1]}")

        return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)

    @EnhancedToolManager.tool
    def search_in_specified_file_v2(self, file_path: str, search_term: str) -> str:
        """
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching. This file must be python file.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        Output:
            matching locations with line numbers, or error description
        """
        if not file_path.endswith(".py"):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH.name, f"Error: file '{file_path}' is not a python file."
            )
        return self._extract_function_matches(file_path, search_term)
    @EnhancedToolManager.tool
    def get_context_around_line(self, file_path: str, line_number: int, context_size: int = 5) -> str:
        """
        Get code context around a specific line number. Useful for investigating errors, test failures, or following up on search results.
        Arguments:
            file_path: target file to read from
            line_number: center line number to get context around (1-indexed)
            context_size: number of lines before and after to include (default: 5)
        Output:
            code snippet with line numbers, highlighting the target line
        """
        if not os.path.exists(file_path):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name, f"Error: file '{file_path}' does not exist.")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name, f"Error reading '{file_path}': {e}")

        total_lines = len(lines)

        if line_number < 1 or line_number > total_lines:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: line_number {line_number} is out of range. File has {total_lines} lines.",
            )

        # Calculate context boundaries
        start_line = max(1, line_number - context_size)
        end_line = min(total_lines, line_number + context_size)

        # Build output with line numbers
        result_lines = []
        result_lines.append(f"File: {file_path}")
        result_lines.append(f"Showing lines {start_line}-{end_line} (centered on line {line_number}):\n")

        for i in range(start_line - 1, end_line):
            current_line_num = i + 1
            line_content = lines[i].rstrip("\n")

            # Highlight the target line
            if current_line_num == line_number:
                prefix = ">>>"
            else:
                prefix = "   "

            result_lines.append(f"{prefix} {current_line_num:4}: {line_content}")

        return "\n".join(result_lines)

    @EnhancedToolManager.tool
    def list_directory(self, path: str = ".", pattern: str = None, show_hidden: bool = False) -> str:
        """
        List files and directories in a path with metadata. Essential for exploring project structure and finding files to work with.
        Arguments:
            path: directory to list (default: current directory)
            pattern: optional glob pattern to filter results (e.g., "*.py", "test_*")
            show_hidden: whether to show files/directories starting with . (default: False)
        Output:
            formatted list of files and directories with type and size
        """
        import glob
        # Validate path exists and is a directory
        if not os.path.exists(path):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name, f"Error: directory '{path}' does not exist.")

        if not os.path.isdir(path):
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name, f"Error: '{path}' is not a directory.")

        try:
            # Get all items in directory
            if pattern:
                # Use glob pattern if provided
                search_pattern = os.path.join(path, pattern)
                all_items = glob.glob(search_pattern)
                # Get just the basename for display
                items = [os.path.basename(item) for item in all_items]
            else:
                items = os.listdir(path)

            # Filter out hidden files if not requested
            if not show_hidden:
                items = [item for item in items if not item.startswith(".")]

            # Exclude common build/cache directories
            exclude_dirs = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", "node_modules"}
            items = [item for item in items if item not in exclude_dirs]

            if not items:
                return f"Directory '{path}' is empty (or only contains hidden/excluded items)."

            # Separate files and directories, collect metadata
            dirs = []
            files = []

            for item in items:
                item_path = os.path.join(path, item)

                try:
                    stat_info = os.stat(item_path)
                    size = stat_info.st_size

                    # Format size
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f} MB"

                    if os.path.isdir(item_path):
                        dirs.append((item, size_str))
                    else:
                        files.append((item, size_str))

                except (OSError, PermissionError):
                    # Skip items we can't stat
                    continue

            # Sort alphabetically
            dirs.sort(key=lambda x: x[0].lower())
            files.sort(key=lambda x: x[0].lower())

            # Build output
            result_lines = []
            result_lines.append(f"Directory: {path} ({len(dirs) + len(files)} items)")
            result_lines.append("")

            # Show directories first
            for name, size in dirs:
                result_lines.append(f"[DIR]  {name}/")

            # Then files
            for name, size in files:
                result_lines.append(f"[FILE] {name:<30} {size:>8}")

            return "\n".join(result_lines)

        except PermissionError:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name, f"Error: permission denied to read directory '{path}'."
            )
        except Exception as e:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.UNKNOWN.name, f"Error listing directory '{path}': {e}")

    def get_final_git_patch(self) -> str:
        """
        Generate a clean unified diff (staged changes only) that tools like `patch`
        or `git apply` can consume.
        """
        try:
            # Stage modified/untracked files with desired extensions, excluding agent files.
            exts = (".py", ".ini", ".cfg", ".toml")
            exclude = {"src/agent.py", "src/agent_runner.py"}
            # Exclude any generated test files or files modified via test generation tool
            try:
                for _p in getattr(self, "generated_test_files", []):
                    # store as relative paths similar to git ls-files output
                    exclude.add(os.path.relpath(_p))
            except Exception:
                pass
            # Discover modified + untracked files
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"], capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()

            to_add = [f for f in ls if f.endswith(exts) and f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)

            # Produce a clean, parseable patch (no colors; standard unified diff).
            diff = subprocess.run(["git", "diff", "--cached", "--no-color", "--unified=3"], capture_output=True, text=True, timeout=30, check=True)

            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:

            return f"Error generating git patch: {e}"

    @EnhancedToolManager.tool
    def generate_test_function(self, file_path: str, test_function_code: str, position: str = "append") -> str:
        """
        Create or append a test function to the specified test file. Generated tests are excluded from final patch.
        Arguments:
            file_path: path to the test file to create or modify
            test_function_code: the full test function code to insert
            position: where to place the function: "append", "top", "after_imports", "before_main", or "auto"
        Output:
            Success message or error message
        """
        if not file_path.endswith(".py"):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_FILE_PATH.name, f"Error: file '{file_path}' is not a python file."
            )
        # Ensure directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # Normalize newline handling
        test_fn = (test_function_code or "").strip()
        if not test_fn:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name, "Error: test_function_code cannot be empty.")

        is_new_file = not os.path.exists(file_path)

        def _insert_after_imports(content: str, block: str) -> str:
            lines = content.splitlines()
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    insert_idx = i + 1
                elif stripped == "" or stripped.startswith("#"):
                    # allow header comments/blank lines before imports
                    insert_idx = max(insert_idx, i + 1)
                else:
                    break
            lines = lines[:insert_idx] + (["", block, ""] if insert_idx < len(lines) else ["", block]) + lines[insert_idx:]
            return "\n".join(lines).rstrip() + "\n"
        def _insert_before_main(content: str, block: str) -> str:
            marker = 'if __name__ == "__main__":'
            idx = content.find(marker)
            if idx == -1:
                return None
            return content[:idx].rstrip() + "\n\n" + block + "\n\n" + content[idx:]
        if is_new_file:
            new_content = test_fn + "\n"
            # Validate standalone content before writing
            is_err, err = self.check_syntax_error(new_content)
            if is_err:
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, f"Error: generated test function has syntax error: {err}"
                )
        else:
            original = self._get_file_content(file_path, limit=-1)
            # Avoid duplicating exact same function text
            if test_fn in original:
                rel = os.path.relpath(file_path)
                if rel not in self.generated_test_files:
                    self.generated_test_files.append(rel)
                return f"Test already present in '{rel}', no changes made."

            # Build candidate insertion strategies in order
            candidates = []
            if position == "append":
                candidates = [lambda src: src.rstrip() + "\n\n" + test_fn + "\n"]
            elif position == "top":
                candidates = [lambda src: test_fn + "\n\n" + src]
            elif position == "after_imports":
                candidates = [lambda src: _insert_after_imports(src, test_fn)]
            elif position == "before_main":
                candidates = [lambda src: (_insert_before_main(src, test_fn) or src.rstrip() + "\n\n" + test_fn + "\n")]
            elif position == "auto":
                candidates = [
                    lambda src: (_insert_before_main(src, test_fn) or _insert_after_imports(src, test_fn)),
                    lambda src: src.rstrip() + "\n\n" + test_fn + "\n",
                    lambda src: test_fn + "\n\n" + src,
                ]
            else:
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                    f"Error: invalid position '{position}'. Use 'append', 'top', 'after_imports', 'before_main', or 'auto'.",
                )

            # Try each candidate until one passes syntax check
            new_content = None
            first_error = None
            for builder in candidates:
                try:
                    candidate = builder(original)
                    is_err, err = self.check_syntax_error(candidate)
                    if not is_err:
                        new_content = candidate
                        break
                    if first_error is None:
                        first_error = err
                except Exception as e:
                    if first_error is None:
                        first_error = e
                    continue

            if new_content is None:
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, f"Error: inserting test caused syntax error. First error: {first_error}"
                )

        self._save(file_path, new_content)

        # Track for exclusion from final patch
        rel = os.path.relpath(file_path)
        if rel not in self.generated_test_files:
            self.generated_test_files.append(rel)

        return f"Test {'created' if is_new_file else 'updated'} in '{rel}' (position={position})."

    @EnhancedToolManager.tool
    def run_repo_tests(self, file_paths: List[str]) -> str:
        """
        Runs the tests for the repository. This tool will only run the tests for the files provided.
        Arguments:
            file_paths: path of the files to run the tests for.
        Output:
            Returns the stdout/stderr from the executed files.
        """
        if self.test_runner == "pytest":
            result = subprocess.run(["pytest"] + file_paths, shell=True, capture_output=True, text=True, timeout=90)
            output = (result.stdout or "") + (result.stderr or "")
        elif self.test_runner == "unittest":

            output = ""
            for file_path in file_paths:
                result = subprocess.run(["python", file_path], capture_output=True, text=True, timeout=60)
                current_output = (result.stdout or "") + (result.stderr or "")
                output += current_output
        else:
            if self.test_runner_mode == "MODULE":
                modules = [convert_filepath_to_module(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(modules)}"

                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
            else:
                files_to_test = [purge_filepath(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(files_to_test)}"

                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
        return output

    @EnhancedToolManager.tool
    def run_code(self, content: str, file_path: str) -> str:
        """
        Runs any python code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it. Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.
        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.

        Output:
            Returns the stdout/stderr from the executed file.
            Returns error message if there are any third party dependencies.
        """
        self._save(file_path, content)
        self.generated_test_files.append(file_path)
        # Parse the file's AST to collect import statements

        with open(file_path, "r") as f:
            tree = ast.parse(f.read(), filename=file_path)

        disallowed_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Use the module specified in 'from x import y' if available
                # otherwise fall back to the imported name from plain 'import x'
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                else:
                    mod = node.names[0].name.split(".")[0]

                # Skip if built-in module
                if mod in sys.builtin_module_names:
                    continue

                # Skip relative imports ("from . import foo") which have level > 0
                if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                    continue

                # --- Additional check: allow local modules/packages in CWD ---
                cwd = os.getcwd()
                local_file = os.path.join(cwd, f"{mod}.py")
                local_pkg_init = os.path.join(cwd, mod, "__init__.py")
                local_pkg_dir = os.path.join(cwd, mod)
                # Also check inside a conventional 'lib' folder within cwd
                lib_dir = os.path.join(cwd, "lib")
                lib_file = os.path.join(lib_dir, f"{mod}.py")
                lib_pkg_init = os.path.join(lib_dir, mod, "__init__.py")
                lib_pkg_dir = os.path.join(lib_dir, mod)

                if (
                    os.path.isfile(local_file)
                    or os.path.isfile(local_pkg_init)
                    or os.path.isdir(local_pkg_dir)
                    or os.path.isfile(lib_file)
                    or os.path.isfile(lib_pkg_init)
                    or os.path.isdir(lib_pkg_dir)
                ):
                    # Treat as local dependency, allow it
                    continue

                # Any other module is considered disallowed
                disallowed_modules.add(mod)

        result = subprocess.run(["python", file_path], capture_output=True, text=True, check=False, timeout=60)
        if result.returncode != 0:

            error_type = EnhancedToolManager.Error.ErrorType.RUNTIME_ERROR
            if "ImportError" in result.stderr:
                error_type = EnhancedToolManager.Error.ErrorType.IMPORT_ERROR
            if "ModuleNotFoundError" in result.stderr:
                error_type = EnhancedToolManager.Error.ErrorType.THIRD_PARTY_DEPENDENCIES
            raise EnhancedToolManager.Error(error_type, f"Error running code: {result.stderr}\n")
        observation = f"{result.stdout}\n"

        return observation

    @EnhancedToolManager.tool
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        """
        Performs targeted text replacement within source files. If there are any syntax errors in the code, it rejects the edit with an error message. Please note use you can only use this tool after you have approval from user on your proposed solution.
        Arguments:
        file_path: target file for modification
        search: exact text pattern to locate and replace
        replace: new text content to substitute
        Output:
            operation status - success confirmation or detailed error with guidance
        """
        if not self.is_solution_approved:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: You cannot use this tool before you have approval from user on your proposed solution. Please call get_approval_for_solution tool first with list of proposed solutions.",
            )
        if not os.path.exists(file_path):

            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.FILE_NOT_FOUND.name, f"Error: file '{file_path}' does not exist.")

        original = self._get_file_content(file_path, limit=-1)

        match original.count(search):
            case 0:

                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.SEARCH_TERM_NOT_FOUND.name,
                    f"Error: search string not found in file {file_path}. You need to share the exact code you want to replace.",
                )
            case 1:

                new_content = original.replace(search, replace)
                try:
                    is_error, error = self.check_syntax_error(new_content)
                    if not is_error:
                        self.save_file(file_path, new_content)

                        return "ok, code edit applied successfully"
                    else:
                        error.message = "code edit failed. " + error.message
                        raise error
                except EnhancedToolManager.Error as e:
                    raise EnhancedToolManager.Error(
                        EnhancedToolManager.Error.ErrorType.SYNTAX_ERROR.name, f"Error: syntax error in file {file_path}. {e.message}"
                    )
            case num_hits:

                raise EnhancedToolManager.Error(
                    EnhancedToolManager.Error.ErrorType.MULTIPLE_SEARCH_RESULTS_FOUND.name,
                    f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change.",
                )

    @EnhancedToolManager.tool
    def finish(self, investigation_summary: str):
        """
        Signals completion of the current workflow execution
        Arguments:
            investigation_summary: Please provide a detailed summary of the findings from your investigation and detailed solution to the problem.Use the following format:
                Problem: <problem_statement>
                Investigation: <investigation_summary>
                Solution: <your solution>
        """
        qa_response = {"is_patch_correct": "yes"}
        if qa_response.get("is_patch_correct", "no").lower() == "yes":
            return "finish"
        else:
            raise EnhancedToolManager.Error(EnhancedToolManager.Error.ErrorType.BUG_REPORT_REQUIRED.name, qa_response.get("analysis", ""))
    @EnhancedToolManager.tool
    def get_project_metadata(self):
        """
        Extract project metadata from configuration files (setup.py, pyproject.toml, package.json, requirements.txt).
        Shows project name, version, dependencies, entry points, and other key information.
        Useful for understanding project context and setup.
        """
        try:
            metadata = {}
            config_files = {
                "setup.py": self._parse_setup_py,
                "pyproject.toml": self._parse_pyproject_toml,
                "package.json": self._parse_package_json,
                "requirements.txt": self._parse_requirements_txt,
                "setup.cfg": self._parse_setup_cfg,
            }

            for filename, parser in config_files.items():
                if os.path.exists(filename):
                    try:
                        data = parser(filename)
                        if data:
                            metadata[filename] = data
                    except Exception:
                        pass

            if not metadata:
                return "No project metadata files found (setup.py, pyproject.toml, package.json, requirements.txt, setup.cfg)"

            result = "## Project Metadata\n\n"
            for filename, data in metadata.items():
                result += f"### {filename}\n"
                for key, value in data.items():
                    if isinstance(value, list):
                        result += f"- **{key}**: {', '.join(str(v) for v in value[:10])}"
                        if len(value) > 10:
                            result += f" ... and {len(value) - 10} more"
                        result += "\n"
                    elif isinstance(value, dict):
                        result += f"- **{key}**:\n"
                        for k, v in list(value.items())[:10]:
                            result += f"  - {k}: {v}\n"
                        if len(value) > 10:
                            result += f"  ... and {len(value) - 10} more\n"
                    else:
                        result += f"- **{key}**: {value}\n"
                result += "\n"

            return result
        except Exception as e:
            return f"Error retrieving project metadata. Do not try again"

    def _parse_setup_py(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        data = {}

        patterns = {
            "name": r"name\s*=\s*['\"]([^'\"]+)['\"]",
            "version": r"version\s*=\s*['\"]([^'\"]+)['\"]",
            "description": r"description\s*=\s*['\"]([^'\"]+)['\"]",
            "author": r"author\s*=\s*['\"]([^'\"]+)['\"]",
            "python_requires": r"python_requires\s*=\s*['\"]([^'\"]+)['\"]",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                data[key] = match.group(1)

        install_requires = re.search(r"install_requires\s*=\s*\[(.*?)\]", content, re.DOTALL)
        if install_requires:
            deps = re.findall(r"['\"]([^'\"]+)['\"]", install_requires.group(1))
            data["dependencies"] = deps

        return data

    def _parse_pyproject_toml(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        data = {}

        name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
        if name_match:
            data["name"] = name_match.group(1)

        version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        if version_match:
            data["version"] = version_match.group(1)

        description_match = re.search(r'description\s*=\s*["\']([^"\']+)["\']', content)
        if description_match:
            data["description"] = description_match.group(1)

        python_match = re.search(r'python\s*=\s*["\']([^"\']+)["\']', content)
        if python_match:
            data["python_requires"] = python_match.group(1)

        deps_section = re.search(r"\[tool\.poetry\.dependencies\](.*?)(?=\[|\Z)", content, re.DOTALL)
        if deps_section:
            deps = re.findall(r"^([a-zA-Z0-9_-]+)\s*=", deps_section.group(1), re.MULTILINE)
            if deps:
                data["dependencies"] = [d for d in deps if d != "python"]

        return data

    def _parse_package_json(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            import json
            pkg = json.load(f)

        data = {}
        if "name" in pkg:
            data["name"] = pkg["name"]
        if "version" in pkg:
            data["version"] = pkg["version"]
        if "description" in pkg:
            data["description"] = pkg["description"]
        if "dependencies" in pkg:
            data["dependencies"] = list(pkg["dependencies"].keys())
        if "scripts" in pkg:
            data["scripts"] = pkg["scripts"]

        return data

    def _parse_requirements_txt(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        deps = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                deps.append(line.split("#")[0].strip())

        return {"dependencies": deps} if deps else {}

    def _parse_setup_cfg(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        data = {}

        name_match = re.search(r"name\s*=\s*(.+)", content)
        if name_match:
            data["name"] = name_match.group(1).strip()

        version_match = re.search(r"version\s*=\s*(.+)", content)
        if version_match:
            data["version"] = version_match.group(1).strip()

        return data

def get_directory_tree(start_path: str = ".") -> str:
    tree_lines = []
    def add_tree(p, pre="", last=True, root=False):
        try:
            d = os.path.basename(p) if p != "." else os.path.basename(os.getcwd())
            if not root:
                tree_lines.append(f"{pre}{'└── ' if last else '├── '}{d}/")
            try:
                items = sorted([i for i in os.listdir(p) if not i.startswith(".")])
                dirs = [i for i in items if os.path.isdir(os.path.join(p, i))]
                files = [i for i in items if not os.path.isdir(os.path.join(p, i))]
                for i, dn in enumerate(dirs):
                    add_tree(os.path.join(p, dn), pre + ("" if root else ("    " if last else "│   ")), i == len(dirs) - 1 and len(files) == 0, False)
                for i, fn in enumerate(files):
                    tree_lines.append(f"{pre}{'' if root else ('    ' if last else '│   ')}{'└── ' if i == len(files) - 1 else '├── '}{fn}")
            except PermissionError:
                tree_lines.append(f"{pre}{'' if root else ('    ' if last else '│   ')}└── [Permission Denied]")
        except Exception as e:
            tree_lines.append(f"{pre}└── [Error: {str(e)}]")
    add_tree(start_path, is_root=True)
    return "\n".join(tree_lines)
def find_readme(file_path: str, repo_path: str) -> Optional[str]:
    d = os.path.dirname(file_path)
    while True:
        for n in ["README.md", "README.rst"]:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
        if d == repo_path:
            break
        d = os.path.dirname(d)
    return None
def find_test_runner(readme_file_path: Optional[str] = None):
    if not readme_file_path:
        return "pytest"
    try:
        with open(readme_file_path, "r", encoding="utf-8") as f:
            r = f.read()
        p = textwrap.dedent("""You are a helpful assistant that can find the test runner for a given repository. The test runner is the file that can run individual test files (e.g. pytest, unittest). Read the README and find the test runner. If no test runner, return pytest. Output format: abc/test.py. No other texts.""")
        return Network.make_request([{"role": "system", "content": p}, {"role": "user", "content": r}], model=DEEPSEEK_MODEL_NAME).strip() or "pytest"
    except Exception:
        return "pytest"
def determine_test_runner_mode(test_runner: str):
    if test_runner == "pytest":
        return "FILE"
    try:
        with open(test_runner, "r", encoding="utf-8") as f:
            c = f.read()
        return Network.make_request([{"role": "system", "content": TEST_RUNNER_MODE_PROMPT}, {"role": "user", "content": c}], model=DEEPSEEK_MODEL_NAME).strip() or "FILE"
    except Exception:
        return "FILE"
def count_test_cases(file_path: str) -> int:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return len(re.findall(r"^\s*def\s+test_\w+", f.read(), re.MULTILINE))
    except (FileNotFoundError, UnicodeDecodeError):
        return 0
def determine_test_runner_and_mode():
    test_files = sorted([os.path.join(r, f) for r, _, fs in os.walk(".") for f in fs if "test_" in f and f.endswith(".py")], key=len)
    for p in test_files:
        if count_test_cases(p) > 5:
            r = find_readme(p, ".")
            if r:
                tr = find_test_runner(r)
                return tr, determine_test_runner_mode(tr)
            break
    return "pytest", "FILE"
def convert_filepath_to_module(file_path: str, repo_path: str, test_runner: str) -> str:
    mp = os.path.splitext(os.path.abspath(file_path))[0]
    rp = os.path.abspath(repo_path)
    if mp.startswith(rp):
        mp = mp[len(rp):].lstrip(os.path.sep)
    trd = os.path.dirname(test_runner)
    if trd and mp.startswith(trd):
        mp = mp[len(trd):].lstrip(os.path.sep)
    return mp.replace(os.path.sep, ".")
def purge_filepath(file_path: str, repo_path: str, test_runner: str) -> str:
    mp = os.path.splitext(os.path.abspath(file_path))[0]
    rp = os.path.abspath(repo_path)
    if mp.startswith(rp):
        mp = mp[len(rp):].lstrip(os.path.sep)
    trd = os.path.dirname(test_runner)
    if trd and mp.startswith(trd):
        mp = mp[len(trd):].lstrip(os.path.sep)
    return mp
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
) -> str:
    global run_id
    run_id = run_id_1
    try:
        from framework_ext import AgentPlanExecuteVerifyWorkflow, PhaseManager
        pev = AgentPlanExecuteVerifyWorkflow(enable_pev=enable_pev, enable_mcts=enable_mcts) if enable_pev or enable_mcts else None
        s = pev.run_planning_phase(problem_statement) if pev else {"name": "Default", "description": "Standard approach"}
        mp = pev.run_mcts_exploration(problem_statement) if pev and enable_mcts else []
        strategy_guidance = f"\n\nStrategic Plan: {s.get('name', 'Default')} - {s.get('description', 'Standard approach')}"
        mcts_guidance = f"\n\nMCTS Recommended Path: {' -> '.join(mp[:5])}" if mp else ""
        pm = PhaseManager(problem_statement, n_max_steps)
        use_multi_phase = pm.use_multi_phase_workflow()
    except ImportError:
        pev, pm = None, None
        strategy_guidance = mcts_guidance = ""
        class DummyPM:
            def use_multi_phase_workflow(self): return False
            def should_transition(self, *args): return False, None
            def transition_to_phase(self, *args): pass
            def get_phase_guidance(self): return ""
            def create_checkpoint(self, *args): pass
        pm = DummyPM()
        use_multi_phase = False
    try:
        from create_tasks_ext import enhance_problem_statement
        e = enhance_problem_statement(problem_statement)
        enhanced_problem = problem_statement + "\n\n---\n\n# Enhanced Problem Analysis\n\n" + e if e else problem_statement
    except ImportError:
        enhanced_problem = problem_statement
    phase_manager = pm
    cot = EnhancedCOT(latest_observations_to_keep=30)
    tool_manager = FixTaskEnhancedToolManager(
        available_tools=["get_file_content", "save_file", "get_approval_for_solution", "search_in_all_files_content",
            "search_in_specified_file_v2", "list_directory", "get_context_around_line", "run_repo_tests",
            "run_code", "apply_code_edit", "generate_test_function", "get_project_metadata", "finish"],
        test_runner=test_runner, test_runner_mode=test_runner_mode,
    )
    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(
        tools_docs=tool_manager.get_tool_docs(), format_prompt=FORMAT_PROMPT_V0, extra_fix_request=extra_fix_request
    )
    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE.format(problem_statement=enhanced_problem) + strategy_guidance + mcts_guidance
    start_time = time.time()
    for step in range(n_max_steps):
        if use_multi_phase and step > 0:
            st, np = phase_manager.should_transition(step, cot)
            if st:
                phase_manager.transition_to_phase(np, step)
        if time.time() - start_time > timeout:
            cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached", next_tool_name="", next_tool_args={}, observation="", is_error=True))
            break
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": instance_prompt}]
        messages.extend(cot.to_str())
        if use_multi_phase:
            messages.append({"role": "system", "content": phase_manager.get_phase_guidance()})
        messages.append({"role": "system", "content": STOP_INSTRUCTION})
        temperature = 0
        selected_model = GLM_MODEL_NAME
        if cot.is_thought_repeated():
            last_thought = cot.thoughts[-1]
            messages.append({
                "role": "user",
                "content": DO_NOT_REPEAT_TOOL_CALLS.format(
                    previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}"
                ),
            })
            if cot.repeated_thoughts > 1:
                temperature = min(cot.repeated_thoughts / 10, 0.7)
                selected_model = AGENT_MODELS[random.randint(0, len(AGENT_MODELS) - 1)] if cot.repeated_thoughts > 2 else GLM_MODEL_NAME
        try:
            nt, ntn, nta, rt, ta, ec, msgs = Network.inference(messages, model=selected_model, run_id=run_id, temperature=temperature)
            next_thought, next_tool_name, next_tool_args, raw_text, total_attempts, error_counter = nt, ntn, nta, rt, ta, ec
            messages = msgs
        except Exception as e:
            em = f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
            if "too long context" in em:
                cot.clear_and_restart()
            cot.add_action(EnhancedCOT.Action(next_thought=em, next_tool_name="", next_tool_args={}, observation="", is_error=True, raw_response="", total_attempts=0, inference_error_counter={}, request_data=messages))
            break
        try:
            if '"' in next_tool_name or "'" in next_tool_name:
                next_tool_name = next_tool_name.replace('"', "").replace("'", "")
            no = tool_manager.get_tool(next_tool_name)(**next_tool_args) if next_tool_args else tool_manager.get_tool(next_tool_name)()
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought, next_tool_name=next_tool_name, next_tool_args=next_tool_args, observation=no, is_error=False, raw_response=raw_text, total_attempts=total_attempts, inference_error_counter=error_counter, request_data=messages))
            if use_multi_phase and next_tool_name in ["run_repo_tests", "apply_code_edit", "get_approval_for_solution"]:
                if "passed" in str(no).lower() or "failed" in str(no).lower():
                    phase_manager.create_checkpoint(step, {"observation": str(no)[:200]})
            if enable_pev and enable_mcts and pev and pev.mcts:
                pev.mcts.update_root(next_tool_name, str(no), "error" not in str(no).lower())
        except EnhancedToolManager.Error as e:
            em = f"observation: {e.message}"
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought, next_tool_name=next_tool_name, next_tool_args=next_tool_args, observation=em, is_error=True, raw_response=raw_text, total_attempts=total_attempts, inference_error_counter=error_counter, request_data=messages))
            if enable_pev and enable_mcts and pev and pev.mcts:
                pev.mcts.update_root(next_tool_name, em, False)
            continue
        except Exception as e:
            et = traceback.format_exc()
            em = f"observation: {str(e)}" if isinstance(e, TypeError) else f"observation: {repr(e)} {et}"
            cot.add_action(EnhancedCOT.Action(next_thought=next_thought, next_tool_name=next_tool_name, next_tool_args=next_tool_args, observation=em, is_error=True, raw_response=raw_text, total_attempts=total_attempts, inference_error_counter=error_counter, request_data=messages))
            if enable_pev and enable_mcts and pev and pev.mcts:
                pev.mcts.update_root(next_tool_name, em, False)
            continue
        if next_tool_name == "finish":
            break
    else:
        cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached", next_tool_name="", next_tool_args={}, observation="", is_error=True))
        if n_max_steps < MAX_FIX_TASK_STEPS:
            return ""
    patch = tool_manager.get_final_git_patch()
    return patch
def ensure_git_initialized():
    wd = os.getcwd()
    try:
        os.chdir(wd)
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", wd])
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True, text=True)
        else:
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", wd])
    except Exception:
        pass
    finally:
        os.chdir(wd)
def set_env_for_agent():
    cwd = os.getcwd()
    pypath = os.environ.get("PYTHONPATH", "")
    if cwd not in pypath:
        os.environ["PYTHONPATH"] = pypath + ":" + cwd
    lib = cwd + "/lib"
    if Path(lib).exists() and lib not in pypath:
        os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + lib
def process_fix_task(input_dict: Dict[str, Any], enable_pev: bool = True, enable_mcts: bool = True):
    global run_id
    problem_text = input_dict.get("problem_statement")
    if not problem_text:
        raise ValueError("input_dict must contain 'problem_statement'.")
    timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT)))
    patch_text = ""
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_path.split("/")[-1]
    if os.path.exists(repod_dir):
        os.chdir(repod_dir)
    set_env_for_agent()
    cwd = os.getcwd()
    tr, trm = determine_test_runner_and_mode()
    try:
        patch_text = fix_task_solve_workflow(problem_text, timeout=timeout, run_id_1=run_id, test_runner=tr, test_runner_mode=trm, enable_pev=enable_pev, enable_mcts=enable_mcts, extra_fix_request=FIX_TASK_NEVER_EARLY_STOP_PROMPT)
        os.system("git reset --hard")
    except Exception:
        pass
    finally:
        os.chdir(cwd)
    return patch_text
def check_problem_type(problem_statement: str) -> str:
    p = textwrap.dedent("""You are the problem type checker. Categories: 1. CREATE: new functionality from scratch. 2. FIX: fixing bug, improving existing codebase. Only respond with "FIX" or "CREATE".""")
    retry = 0
    while retry < 10:
        try:
            msgs = [{"role": "system", "content": p}, {"role": "user", "content": f"{problem_statement}\n# Project Tree Structure: \n{get_directory_tree()}"}]
            r = Network.make_request(msgs, model=QWEN_MODEL_NAME)
            if r not in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                retry += 1
            else:
                break
        except Exception:
            retry += 1
        time.sleep(2)
    return r
def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", enable_pev: bool = True, enable_mcts: bool = True) -> Dict[str, str]:
    global DEFAULT_PROXY_URL, DEFAULT_TIMEOUT, run_id
    run_id = os.getenv("RUN_ID", "")
    repo_dir = os.path.abspath(repo_dir)
    sys.path.insert(0, repo_dir)
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)
    ensure_git_initialized()
    set_env_for_agent()
    try:
        pt = check_problem_type(input_dict.get("problem_statement"))
        if pt == PROBLEM_TYPE_FIX:
            result = process_fix_task(input_dict, enable_pev=enable_pev, enable_mcts=enable_mcts)
        else:
            try:
                from create_tasks_ext import process_create_task
                result = process_create_task(input_dict, enable_pev=enable_pev, enable_mcts=enable_mcts)
            except ImportError:
                result = process_fix_task(input_dict, enable_pev=enable_pev, enable_mcts=enable_mcts)
    except Exception:
        result = process_fix_task(input_dict, enable_pev=enable_pev, enable_mcts=enable_mcts)
    os.system("git reset --hard")
    return {"patch": result if result else ""}
