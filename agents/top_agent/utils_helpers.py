"""
Utility classes and helper functions for the Ridges agent.
Extracted from agent.py to reduce main file size.
"""

import ast
import csv
import json
from json import JSONDecodeError
from typing import Dict, Any, List


class FunctionVisitor(ast.NodeVisitor):
    """AST visitor for extracting function definitions and metadata."""
    
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
        full_function_name = f"{self.current_class}::{node.name}" if self.current_class else node.name
        line_number = node.lineno
        if isinstance(node.decorator_list, list) and len(node.decorator_list) > 0:
            line_number = node.decorator_list[0].lineno

        end_line_number = line_number
        if isinstance(node.body, list) and len(node.body) > 0:
            end_line_number = node.body[-1].lineno

        lines = self.file_content.split("\n")
        body = "\n".join(lines[line_number - 1:end_line_number])

        self.functions[full_function_name] = {
            "class": self.current_class,
            "body": body,
            "line_number": line_number
        }
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
    """Utility methods for string processing and JSON handling."""
    
    @classmethod
    def limit_strings(cls, strings: str, n=1000) -> str:
        """Limit the number of strings to n lines."""
        strings_list = strings.split("\n")
        if len(strings_list) > n:
            return "\n".join(strings_list[:n]) + "\n..." + f"({len(strings_list) - n} more lines)"
        else:
            return strings

    @classmethod
    def load_json(cls, json_string: str) -> dict:
        """Parse JSON string with fallback to LLM-based fixing."""
        try:
            return json.loads(json_string)
        except Exception as e:
            try:
                return eval(json_string)
            except Exception as e:
                # Lazy import to avoid circular dependencies
                from agent import EnhancedNetwork
                fixed_json = EnhancedNetwork.fix_json_string_with_llm(json_string)
                if fixed_json:
                    return fixed_json
                else:
                    raise JSONDecodeError("Invalid JSON", json_string, 0)

    @classmethod
    def log_to_failed_messages(cls, text_resp: str):
        """Log failed LLM responses to CSV."""
        with open("../failed_messages.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([text_resp])


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


__all__ = ['FunctionVisitor', 'Utils', 'VariableNormalizer']
