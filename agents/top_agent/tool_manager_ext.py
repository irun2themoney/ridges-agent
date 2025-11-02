import os
import re
import ast
import sys
import glob
import subprocess
import traceback
from typing import Optional, List


# Import EnhancedToolManager from agent_core (to avoid circular import, do it lazily)
_EnhancedToolManager = None

def get_enhanced_toolmanager():
    global _EnhancedToolManager
    if _EnhancedToolManager is None:
        # Import only when first needed
        try:
            from agent_core import EnhancedToolManager
            _EnhancedToolManager = EnhancedToolManager
        except ImportError:
            # Fallback: if agent_core is not available, try agent
            from agent import EnhancedToolManager
            _EnhancedToolManager = EnhancedToolManager
    return _EnhancedToolManager


# Don't set EnhancedToolManager at module load time - set it when FixTaskEnhancedToolManager is defined
# This prevents circular imports


# Create the class with proper inheritance - inherit from dynamically loaded parent
class FixTaskEnhancedToolManager(get_enhanced_toolmanager()):

    def __init__(self, available_tools: Optional[list[str]] = [], test_runner: str = "pytest",
                 test_runner_mode: str = "FILE"):
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

        self.tool_failure = {
            k: {j: 0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys()
        }

        self.tool_invocations = {
            k: 0 for k in self.TOOL_LIST.keys()
        }

    def check_syntax_error(self, content: str, file_path: str = "<unknown>") -> bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            return True, EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.SYNTAX_ERROR.name,
                f"Syntax error. {str(e)}"
            )

    def _get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None,
                          search_term: str = None, limit: int = 5000) -> str:
        from utils_helpers import Utils
        
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
                content = ''.join(lines[start:end])
                return f"Lines {start + 1}-{end} of {file_path}:\n{content}"
            else:
                content = f.read()

        return Utils.limit_strings(content, n=limit) if limit != -1 else content

    @staticmethod
    def get_tool_decorator():
        return EnhancedToolManager.tool

    # For each @EnhancedToolManager.tool decorated method, we apply it dynamically
    # This is a workaround for the decorator to work with inherited class

    def get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None,
                         search_term: str = None) -> str:
        return self._get_file_content(file_path, search_start_line, search_end_line, search_term, limit=5000)

    get_file_content = EnhancedToolManager.tool(get_file_content)

    def save_file(self, file_path: str, content: str) -> str:
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: You cannot use this tool to create test or files to reproduce the error."
            )
        return self._save(file_path, content)

    save_file = EnhancedToolManager.tool(save_file)

    def get_approval_for_solution(self, solutions: list[str], selected_solution: int, reason_for_selection: str) -> str:
        parsed_solutions = []
        for solution in solutions:
            sols = re.split(r"(Solution \d+[\s\-:]+)", solution)
            sols = [f"{sols[i]}{sols[i + 1]}" for i in range(1, len(sols), 2) if i + 1 < len(sols)]
            parsed_solutions.extend(sols)

        if parsed_solutions:
            solutions = parsed_solutions

        if type(solutions) is not list or len(solutions) < 2:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: solutions must be a list with length at least 2."
            )

        self.is_solution_approved = True
        return "Approved"

    get_approval_for_solution = EnhancedToolManager.tool(get_approval_for_solution)

    def _save(self, file_path: str, content: str) -> str:
        is_syntax_error, error = self.check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            error.message = "Error saving file. " + error.message
            raise EnhancedToolManager.Error(EnhancedToolManager.ErrorType.SYNTAX_ERROR.name, error.message)

    def search_in_all_files_content(self, search_term: str, case_sensitive: bool = False) -> str:
        from utils_helpers import FunctionVisitor, Utils
        
        output = []
        search_flags = 0 if case_sensitive else re.IGNORECASE
        for root, _, files in os.walk("."):
            if ".git" in root or "docs" in root:
                continue

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if re.search(search_term, file_path, search_flags):
                        output.append(f"{file_path} | Filename match")

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        if not re.search(search_term, content, search_flags):
                            continue
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
                EnhancedToolManager.ErrorType.SEARCH_TERM_NOT_FOUND.name,
                f"'{search_term}' not found in the codebase."
            )
        return output

    search_in_all_files_content = EnhancedToolManager.tool(search_in_all_files_content)

    def get_function_ranges(self, file_path: str) -> list[tuple[int, int, str]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.FILE_NOT_FOUND.name,
                f"Error reading '{file_path}': {e}"
            )
        try:
            tree = ast.parse("\n".join(source_lines), filename=file_path)
        except SyntaxError as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.SYNTAX_ERROR.name,
                f"Error parsing '{file_path}': {e}, {traceback.format_exc()}"
            )
            tree = None

        func_ranges: list[tuple[int, int, str]] = []
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = getattr(node, 'lineno', None)
                    end = getattr(node, 'end_lineno', None)
                    if start is not None and end is not None:
                        func_ranges.append((start, end, node.name))
        return func_ranges

    def _extract_function_matches(self, file_path: str, search_term: str, *, max_output_lines: int = 1000) -> str:
        from utils_helpers import Utils
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.FILE_NOT_FOUND.name,
                f"Error reading '{file_path}': {e}"
            )
        match_lines = [idx + 1 for idx, line in enumerate(source_lines) if search_term in line]
        if not match_lines:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.SEARCH_TERM_NOT_FOUND.name,
                f"'{search_term}' not found in file '{file_path}'"
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
            func_src = "\n".join(source_lines[start - 1:end])
            chunks.append(f"(lines {start}-{end}):\n{func_src}")

        for ln in standalone_lines:
            chunks.append(f"{ln}:{source_lines[ln - 1]}")

        return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)

    def search_in_specified_file_v2(self, file_path: str, search_term: str) -> str:
        if not file_path.endswith(".py"):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.INVALID_FILE_PATH.name,
                f"Error: file '{file_path}' is not a python file."
            )
        return self._extract_function_matches(file_path, search_term)

    search_in_specified_file_v2 = EnhancedToolManager.tool(search_in_specified_file_v2)

    def start_over(self, problem_with_old_approach: str, new_apprach_to_try: str):
        os.system("git reset --hard")
        return "Done, codebase reverted to initial state. You can start over with new approach."

    start_over = EnhancedToolManager.tool(start_over)

    def get_context_around_line(self, file_path: str, line_number: int, context_size: int = 5) -> str:
        if not os.path.exists(file_path):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.FILE_NOT_FOUND.name,
                f"Error: file '{file_path}' does not exist."
            )

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.FILE_NOT_FOUND.name,
                f"Error reading '{file_path}': {e}"
            )

        total_lines = len(lines)

        if line_number < 1 or line_number > total_lines:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: line_number {line_number} is out of range. File has {total_lines} lines."
            )

        start_line = max(1, line_number - context_size)
        end_line = min(total_lines, line_number + context_size)

        result_lines = []
        result_lines.append(f"File: {file_path}")
        result_lines.append(f"Showing lines {start_line}-{end_line} (centered on line {line_number}):\n")

        for i in range(start_line - 1, end_line):
            current_line_num = i + 1
            line_content = lines[i].rstrip('\n')

            if current_line_num == line_number:
                prefix = ">>>"
            else:
                prefix = "   "

            result_lines.append(f"{prefix} {current_line_num:4}: {line_content}")

        return '\n'.join(result_lines)

    get_context_around_line = EnhancedToolManager.tool(get_context_around_line)

    def list_directory(self, path: str = ".", pattern: str = None, show_hidden: bool = False) -> str:
        if not os.path.exists(path):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.FILE_NOT_FOUND.name,
                f"Error: directory '{path}' does not exist."
            )

        if not os.path.isdir(path):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: '{path}' is not a directory."
            )

        try:
            if pattern:
                search_pattern = os.path.join(path, pattern)
                all_items = glob.glob(search_pattern)
                items = [os.path.basename(item) for item in all_items]
            else:
                items = os.listdir(path)

            if not show_hidden:
                items = [item for item in items if not item.startswith('.')]

            exclude_dirs = {'__pycache__', '.git', '.pytest_cache', '.mypy_cache', 'node_modules'}
            items = [item for item in items if item not in exclude_dirs]

            if not items:
                return f"Directory '{path}' is empty (or only contains hidden/excluded items)."

            dirs = []
            files = []

            for item in items:
                item_path = os.path.join(path, item)

                try:
                    stat_info = os.stat(item_path)
                    size = stat_info.st_size

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
                    continue

            dirs.sort(key=lambda x: x[0].lower())
            files.sort(key=lambda x: x[0].lower())

            result_lines = []
            result_lines.append(f"Directory: {path} ({len(dirs) + len(files)} items)")
            result_lines.append("")

            for name, size in dirs:
                result_lines.append(f"[DIR]  {name}/")

            for name, size in files:
                result_lines.append(f"[FILE] {name:<30} {size:>8}")

            return '\n'.join(result_lines)

        except PermissionError:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.FILE_NOT_FOUND.name,
                f"Error: permission denied to read directory '{path}'."
            )
        except Exception as e:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.UNKNOWN.name,
                f"Error listing directory '{path}': {e}"
            )

    list_directory = EnhancedToolManager.tool(list_directory)

    def get_final_git_patch(self) -> str:
        try:
            exts = (".py", ".ini", ".cfg", ".toml")
            exclude = {"src/agent.py", "src/agent_runner.py"}
            try:
                for _p in getattr(self, "generated_test_files", []):
                    exclude.add(os.path.relpath(_p))
            except Exception:
                pass
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()

            to_add = [f for f in ls if f.endswith(exts) and f not in exclude]
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"],
                capture_output=True, text=True, timeout=30, check=True
            )

            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            return f"Error generating git patch: {e}"

    def generate_test_function(self, file_path: str, test_function_code: str, position: str = "append") -> str:
        if not file_path.endswith('.py'):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.INVALID_FILE_PATH.name,
                f"Error: file '{file_path}' is not a python file."
            )
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        test_fn = (test_function_code or "").strip()
        if not test_fn:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.INVALID_TOOL_CALL.name,
                "Error: test_function_code cannot be empty."
            )

        is_new_file = not os.path.exists(file_path)

        def _insert_after_imports(content: str, block: str) -> str:
            lines = content.splitlines()
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    insert_idx = i + 1
                elif stripped == "" or stripped.startswith("#"):
                    insert_idx = max(insert_idx, i + 1)
                else:
                    break
            lines = lines[:insert_idx] + (["", block, ""] if insert_idx < len(lines) else ["", block]) + lines[insert_idx:]
            return "\n".join(lines).rstrip() + "\n"

        def _insert_before_main(content: str, block: str) -> str:
            marker = "if __name__ == \"__main__\":"
            idx = content.find(marker)
            if idx == -1:
                return None
            return content[:idx].rstrip() + "\n\n" + block + "\n\n" + content[idx:]

        if is_new_file:
            new_content = test_fn + "\n"
            is_err, err = self.check_syntax_error(new_content)
            if is_err:
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.ErrorType.SYNTAX_ERROR.name,
                    f"Error: generated test function has syntax error: {err}"
                )
        else:
            original = self._get_file_content(file_path, limit=-1)
            if test_fn in original:
                rel = os.path.relpath(file_path)
                if rel not in self.generated_test_files:
                    self.generated_test_files.append(rel)
                return f"Test already present in '{rel}', no changes made."
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
                    EnhancedToolManager.ErrorType.INVALID_TOOL_CALL.name,
                    f"Error: invalid position '{position}'. Use 'append', 'top', 'after_imports', 'before_main', or 'auto'."
                )
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
                    EnhancedToolManager.ErrorType.SYNTAX_ERROR.name,
                    f"Error: inserting test caused syntax error. First error: {first_error}"
                )

        self._save(file_path, new_content)
        rel = os.path.relpath(file_path)
        if rel not in self.generated_test_files:
            self.generated_test_files.append(rel)

        return f"Test {'created' if is_new_file else 'updated'} in '{rel}' (position={position})."

    generate_test_function = EnhancedToolManager.tool(generate_test_function)

    def run_repo_tests(self, file_paths: List[str]) -> str:
        from ridges.evaluator.worker.utils import filepath_to_module, clean_filepath
        
        if self.test_runner == "pytest":
            result = subprocess.run(["pytest"] + file_paths, shell=True, capture_output=True, text=True, timeout=90)
            output = (result.stdout or "") + (result.stderr or "")
        elif self.test_runner == "unittest":
            output = ""
            for file_path in file_paths:
                result = subprocess.run(
                    ["python", file_path],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                current_output = (result.stdout or "") + (result.stderr or "")
                output += current_output
        else:
            if self.test_runner_mode == "MODULE":
                modules = [filepath_to_module(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(modules)}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
            else:
                files_to_test = [clean_filepath(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(files_to_test)}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
                output = (result.stdout or "") + (result.stderr or "")
        return output

    run_repo_tests = EnhancedToolManager.tool(run_repo_tests)

    def run_code(self, content: str, file_path: str) -> str:
        self._save(file_path, content)
        self.generated_test_files.append(file_path)

        with open(file_path, "r") as f:
            tree = ast.parse(f.read(), filename=file_path)

        disallowed_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0]
                else:
                    mod = node.names[0].name.split(".")[0]
                if mod in sys.builtin_module_names:
                    continue
                if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                    continue
                cwd = os.getcwd()
                local_file = os.path.join(cwd, f"{mod}.py")
                local_pkg_init = os.path.join(cwd, mod, "__init__.py")
                local_pkg_dir = os.path.join(cwd, mod)
                lib_dir = os.path.join(cwd, 'lib')
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
                    continue
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

    run_code = EnhancedToolManager.tool(run_code)

    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        if not self.is_solution_approved:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.INVALID_TOOL_CALL.name,
                f"Error: You cannot use this tool before you have approval from user on your proposed solution. Please call get_approval_for_solution tool first with list of proposed solutions."
            )
        if not os.path.exists(file_path):
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.FILE_NOT_FOUND.name,
                f"Error: file '{file_path}' does not exist."
            )

        original = self._get_file_content(file_path, limit=-1)

        match original.count(search):
            case 0:
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.ErrorType.SEARCH_TERM_NOT_FOUND.name,
                    f"Error: search string not found in file {file_path}. You need to share the exact code you want to replace."
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
                        EnhancedToolManager.ErrorType.SYNTAX_ERROR.name,
                        f"Error: syntax error in file {file_path}. {e.message}"
                    )
            case num_hits:
                raise EnhancedToolManager.Error(
                    EnhancedToolManager.ErrorType.MULTIPLE_SEARCH_RESULTS_FOUND.name,
                    f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change."
                )

    apply_code_edit = EnhancedToolManager.tool(apply_code_edit)

    def finish(self, investigation_summary: str):
        qa_response = {"is_patch_correct": "yes"}
        if qa_response.get("is_patch_correct", "no").lower() == "yes":
            return "finish"
        else:
            raise EnhancedToolManager.Error(
                EnhancedToolManager.ErrorType.BUG_REPORT_REQUIRED.name,
                qa_response.get("analysis", "")
            )

    finish = EnhancedToolManager.tool(finish)
