"""
PEV (Plan-Execute-Verify) Workflow and Verification Framework
Extended framework with strategic planning and solution verification.
Extracted from agent.py to reduce main file size.
"""

import json
import textwrap
from typing import Dict, Any, List


class StrategicPlanner:
    """Generates high-level solution strategies"""

    STRATEGY_PROMPT = textwrap.dedent(
        """
            Analyze this problem and generate 3 distinct solution strategies. Each strategy should include:
            - Name and approach description
            - Key steps (high-level)
            - Complexity (low/medium/high)
            - Risk level (low/medium/high)
            - Confidence score (0-1)
            
            Problem: {problem_statement}
            
            Respond in JSON format:
            {{
                "strategies": [
                    {{
                        "name": "strategy_name",
                        "description": "approach description",
                        "steps": ["step1", "step2", "step3"],
                        "complexity": "low/medium/high",
                        "risk": "low/medium/high",
                        "confidence": 0.8
                    }}
                ]
            }}
            """
    )

    def __init__(self, model_name: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"):
        self.model_name = model_name

    def generate_strategies(self, problem_statement: str) -> Dict[str, Any]:
        try:
            # Lazy import to avoid circular dependencies
            from agent import EnhancedNetwork
            
            messages = [
                {"role": "system", "content": "You are a strategic planning expert."},
                {"role": "user", "content": self.STRATEGY_PROMPT.format(problem_statement=problem_statement)}
            ]

            response = EnhancedNetwork.make_request(messages, model=self.model_name)

            if response.strip().startswith('```json'):
                response = response.strip()[7:]
            if response.strip().startswith('```'):
                response = response.strip()[3:]
            if response.strip().endswith('```'):
                response = response.strip()[:-3]
            response = response.strip()

            parsed_response = json.loads(response)

            if parsed_response and "strategies" in parsed_response:
                return parsed_response

        except Exception as e:
            pass

        return {
            "strategies": [
                {
                    "name": "Conservative Fix",
                    "description": "Minimal targeted changes",
                    "steps": ["Locate issue", "Apply minimal fix", "Test"],
                    "complexity": "low",
                    "risk": "low",
                    "confidence": 0.7
                },
                {
                    "name": "Comprehensive Solution",
                    "description": "Root cause analysis and fix",
                    "steps": ["Analyze root cause", "Design solution", "Implement", "Verify"],
                    "complexity": "high",
                    "risk": "medium",
                    "confidence": 0.6
                }
            ]
        }

    def select_best_strategy(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select best strategy based on scoring"""

        def score_strategy(s):
            confidence = s.get("confidence", 0.5)
            risk_score = {"low": 1.0, "medium": 0.7, "high": 0.4}.get(s.get("risk", "medium"), 0.7)
            complexity_score = {"low": 1.0, "medium": 0.8, "high": 0.6}.get(s.get("complexity", "medium"), 0.8)
            return confidence * 0.5 + risk_score * 0.3 + complexity_score * 0.2

        return max(strategies, key=score_strategy)


class Verifier:
    """Automated solution verification"""

    QUALITY_PROMPT = textwrap.dedent(
        """
            Analyze this code for quality issues. Check for:
            - Logic errors
            - Edge case handling
            - Performance issues
            - Code smells/anti-patterns
            
            Code: {code}
            Problem: {problem_statement}
            
            Respond in JSON format:
            {{
                "quality_score": 0.8,
                "issues": ["issue1", "issue2"],
                "edge_cases_handled": true,
                "recommendations": ["rec1", "rec2"]
            }}
            """
    )

    def __init__(self, model_name: str = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"):
        self.model_name = model_name

    def verify_solution(self, problem_statement: str, tool_manager) -> Dict[str, Any]:
        """Run comprehensive verification checks"""
        import os
        from agent import EnhancedNetwork
        
        verification_report = {
            "tests_passed": False,
            "syntax_ok": True,
            "code_quality_ok": False,
            "edge_cases_handled": False,
            "overall_pass": False,
            "issues": [],
            "quality_score": 0.0
        }

        try:
            test_files = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.py')][:3]
            if test_files:
                test_output = tool_manager.run_repo_tests(test_files)
                verification_report[
                    "tests_passed"] = "passed" in test_output.lower() and "failed" not in test_output.lower()
        except Exception as e:
            pass

        try:
            py_files = [f for f in os.listdir('.') if f.endswith('.py') and 'test' not in f][:2]
            if py_files:
                code_content = ""
                for f in py_files:
                    with open(f, 'r') as file:
                        code_content += f"\n=== {f} ===\n{file.read()}"

                messages = [
                    {"role": "system", "content": "You are a code quality expert."},
                    {"role": "user",
                     "content": self.QUALITY_PROMPT.format(code=code_content, problem_statement=problem_statement)}
                ]

                response = EnhancedNetwork.make_request(messages, model=self.model_name)

                if response.strip().startswith('```json'):
                    response = response.strip()[7:]
                if response.strip().startswith('```'):
                    response = response.strip()[3:]
                if response.strip().endswith('```'):
                    response = response.strip()[:-3]
                response = response.strip()

                quality_response = json.loads(response)

                if quality_response:
                    verification_report["quality_score"] = quality_response.get("quality_score", 0.0)
                    verification_report["code_quality_ok"] = verification_report["quality_score"] >= 0.7
                    verification_report["edge_cases_handled"] = quality_response.get("edge_cases_handled", False)
                    verification_report["issues"] = quality_response.get("issues", [])

        except Exception as e:
            pass

        verification_report["overall_pass"] = (
            verification_report["tests_passed"] and
            verification_report["syntax_ok"] and
            (verification_report["code_quality_ok"] or verification_report["edge_cases_handled"])
        )

        return verification_report


class PEVWorkflow:
    """Plan-Execute-Verify workflow orchestrator"""

    def __init__(self, enable_pev: bool = True, enable_mcts: bool = True, max_refinement_iterations: int = 3):
        self.enable_pev = enable_pev
        self.enable_mcts = enable_mcts
        self.max_refinement_iterations = max_refinement_iterations
        self.refinement_count = 0

        if enable_pev:
            self.planner = StrategicPlanner()
            self.verifier = Verifier()
            if enable_mcts:
                # Lazy import to avoid circular dependencies
                from pev_mcts_framework import MCTS
                self.mcts = MCTS(max_depth=10, max_iterations=30)
            else:
                self.mcts = None

    def run_planning_phase(self, problem_statement: str) -> Dict[str, Any]:
        """Phase 1: Strategic Planning"""
        if not self.enable_pev:
            return {"name": "Default", "description": "Standard approach"}
        strategies = self.planner.generate_strategies(problem_statement)
        selected = self.planner.select_best_strategy(strategies["strategies"])
        return selected

    def run_mcts_exploration(self, problem_statement: str) -> List[str]:
        """Phase 2: MCTS Exploration"""
        if not self.enable_pev or not self.enable_mcts:
            return []
        self.mcts.initialize(problem_statement)
        best_path = self.mcts.search()
        return best_path

    def run_verification_phase(self, problem_statement: str, tool_manager) -> Dict[str, Any]:
        """Phase 4: Verification"""
        if not self.enable_pev:
            return {"overall_pass": True}
        verification_result = self.verifier.verify_solution(problem_statement, tool_manager)

        return verification_result

    def should_refine(self, verification_result: Dict[str, Any]) -> bool:
        """Check if refinement is needed"""
        if not verification_result.get("overall_pass", True) and self.refinement_count < self.max_refinement_iterations:
            self.refinement_count += 1
            return True
        return False

    def get_refinement_guidance(self, verification_result: Dict[str, Any]) -> str:
        """Generate refinement guidance"""
        issues = verification_result.get("issues", [])
        if not issues:
            return "No specific issues found for refinement."

        guidance = f"Refinement needed (iteration {self.refinement_count}/{self.max_refinement_iterations}):\n"
        for i, issue in enumerate(issues[:3], 1):
            guidance += f"{i}. {issue}\n"
        return guidance


__all__ = ['StrategicPlanner', 'Verifier', 'PEVWorkflow']
