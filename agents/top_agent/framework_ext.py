# Extracted framework classes for optional enhancements
# These are imported dynamically and are NOT required for core functionality

import math
import json
import re
import textwrap
from typing import Dict, Any, List
from agent import Network, DEEPSEEK_MODEL_NAME

class MCTSNode:
    """Monte Carlo Tree Search node"""

    def __init__(self, state: str, action: str = None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.unexplored_actions = []

    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        return len(self.unexplored_actions) == 0

    def best_child(self) -> "MCTSNode":
        return max(self.children, key=lambda c: c.ucb1_score())

    def add_child(self, action: str, state: str) -> "MCTSNode":
        child = MCTSNode(state, action, self)
        self.children.append(child)
        return child



class MonteCarloTreeSearch:
    """Monte Carlo Tree Search implementation"""

    def __init__(self, max_depth: int = 10, max_iterations: int = 30, exploration_constant: float = 1.414):
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.root = None

    def initialize(self, initial_state: str):
        self.root = MCTSNode(initial_state)
        self.root.unexplored_actions = self._get_possible_actions(initial_state)

    def search(self, iterations: int = None) -> List[str]:
        if iterations is None:
            iterations = self.max_iterations

        for _ in range(iterations):
            node = self._select()
            if not node.is_fully_expanded() and len(node.children) < self.max_depth:
                node = self._expand(node)
            value = self._simulate(node)
            self._backpropagate(node, value)
        best_path = []
        current = self.root
        while current.children:
            current = current.best_child()
            if current.action:
                best_path.append(current.action)
        return best_path

    def _select(self) -> MCTSNode:
        current = self.root
        while current.children and current.is_fully_expanded():
            current = current.best_child()
        return current

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if not node.unexplored_actions:
            return node
        action = node.unexplored_actions.pop(0)
        new_state = f"{node.state} -> {action}"
        return node.add_child(action, new_state)

    def _simulate(self, node: MCTSNode) -> float:
        actions = self._get_action_sequence(node)
        score = 0.5
        if "search" in str(actions):
            score += 0.2
        if "apply_code_edit" in str(actions):
            score += 0.3
        if "finish" in str(actions):
            score += 0.4
        return max(0.0, min(1.0, score - len(actions) * 0.05))

    def _backpropagate(self, node: MCTSNode, value: float):
        current = node
        while current:
            current.visits += 1
            current.value += value
            current = current.parent

    def _get_action_sequence(self, node: MCTSNode) -> List[str]:
        actions = []
        current = node
        while current.parent:
            if current.action:
                actions.append(current.action)
            current = current.parent
        return actions[::-1]

    def _get_possible_actions(self, state: str) -> List[str]:
        return ["search_in_all_files_content", "get_file_content", "apply_code_edit", "run_code", "finish"]

    def update_root(self, action_taken: str, observation: str, success: bool):
        """Update MCTS tree after action execution"""
        if not self.root:
            return

        matching_child = None
        for child in self.root.children:
            if child.action == action_taken:
                matching_child = child
                break

        if matching_child:
            self.root = matching_child
        else:
            new_state = f"{self.root.state} -> {action_taken}"
            self.root = MCTSNode(new_state, action_taken)
            self.root.unexplored_actions = self._get_possible_actions(new_state)

        if success:
            self.root.value += 0.1
        else:
            self.root.value -= 0.1



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

    def __init__(self, model_name: str = DEEPSEEK_MODEL_NAME):
        self.model_name = model_name

    def generate_strategies(self, problem_statement: str) -> Dict[str, Any]:
        try:
            messages = [
                {"role": "system", "content": "You are a strategic planning expert."},
                {"role": "user", "content": self.STRATEGY_PROMPT.format(problem_statement=problem_statement)},
            ]

            response = Network.make_request(messages, model=self.model_name)

            if response.strip().startswith("```json"):
                response = response.strip()[7:]
            if response.strip().startswith("```"):
                response = response.strip()[3:]
            if response.strip().endswith("```"):
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
                    "confidence": 0.7,
                },
                {
                    "name": "Comprehensive Solution",
                    "description": "Root cause analysis and fix",
                    "steps": ["Analyze root cause", "Design solution", "Implement", "Verify"],
                    "complexity": "high",
                    "risk": "medium",
                    "confidence": 0.6,
                },
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


class AgentVerifier:
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

    def __init__(self, model_name: str = DEEPSEEK_MODEL_NAME):
        self.model_name = model_name

    def verify_solution(self, problem_statement: str, tool_manager) -> Dict[str, Any]:
        """Run comprehensive verification checks"""
        verification_report = {
            "tests_passed": False,
            "syntax_ok": True,
            "code_quality_ok": False,
            "edge_cases_handled": False,
            "overall_pass": False,
            "issues": [],
            "quality_score": 0.0,
        }

        try:
            test_files = [f for f in os.listdir(".") if f.startswith("test_") and f.endswith(".py")][:3]
            if test_files:
                test_output = tool_manager.run_repo_tests(test_files)
                verification_report["tests_passed"] = "passed" in test_output.lower() and "failed" not in test_output.lower()
        except Exception as e:
            pass

        try:
            py_files = [f for f in os.listdir(".") if f.endswith(".py") and "test" not in f][:2]
            if py_files:
                code_content = ""
                for f in py_files:
                    with open(f, "r") as file:
                        code_content += f"\n=== {f} ===\n{file.read()}"

                messages = [
                    {"role": "system", "content": "You are a code quality expert."},
                    {"role": "user", "content": self.QUALITY_PROMPT.format(code=code_content, problem_statement=problem_statement)},
                ]

                response = Network.make_request(messages, model=self.model_name)

                if response.strip().startswith("```json"):
                    response = response.strip()[7:]
                if response.strip().startswith("```"):
                    response = response.strip()[3:]
                if response.strip().endswith("```"):
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
            verification_report["tests_passed"]
            and verification_report["syntax_ok"]
            and (verification_report["code_quality_ok"] or verification_report["edge_cases_handled"])
        )

        return verification_report


class AgentPlanExecuteVerifyWorkflow:
    """Plan-Execute-Verify workflow orchestrator"""

    def __init__(self, enable_pev: bool = True, enable_mcts: bool = True, max_refinement_iterations: int = 3):
        self.enable_pev = enable_pev
        self.enable_mcts = enable_mcts
        self.max_refinement_iterations = max_refinement_iterations
        self.refinement_count = 0

        if enable_pev:
            self.planner = StrategicPlanner()
            self.verifier = AgentVerifier()
            if enable_mcts:
                self.mcts = MonteCarloTreeSearch(max_depth=10, max_iterations=30)
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



class PhaseManager:
    """Manages multi-phase workflow for complex problem solving"""

    def __init__(self, problem_statement: str, total_steps: int):
        self.problem_statement = problem_statement
        self.total_steps = total_steps
        self.current_phase = PHASE_INVESTIGATION
        self.phase_history = []
        self.complexity = self._assess_complexity()
        self.step_allocation = self._allocate_steps()
        self.phase_start_step = 0
        self.phase_checkpoints = {}

    def _assess_complexity(self) -> dict:
        """Assess problem complexity using multiple indicators"""

        problem_lower = self.problem_statement.lower()

        indicators = {
            "multi_file": len(re.findall(r"\bfile[s]?\b", problem_lower)) > 2,
            "algorithm": any(kw in problem_lower for kw in ["algorithm", "optimization", "performance", "complexity", "efficient"]),
            "edge_cases": any(kw in problem_lower for kw in ["edge case", "boundary", "corner case", "special case"]),
            "refactor": any(kw in problem_lower for kw in ["refactor", "redesign", "restructure", "rewrite"]),
            "debugging": any(kw in problem_lower for kw in ["bug", "error", "crash", "fail", "incorrect", "fix"]),
            "multiple_components": len(re.findall(r"\bclass\b|\bfunction\b|\bmethod\b", problem_lower)) > 3,
            "integration": any(kw in problem_lower for kw in ["integrate", "interaction", "between", "across"]),
            "backward_compat": any(kw in problem_lower for kw in ["backward", "compatibility", "breaking", "legacy"]),
        }

        score = sum(indicators.values())

        # Determine complexity level
        if score >= 5:
            level = "HIGH"
        elif score >= 3:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {"level": level, "score": score, "indicators": indicators}

    def _allocate_steps(self) -> dict:
        """Allocate steps to each phase based on complexity"""

        if self.complexity["level"] == "HIGH":
            # High complexity: thorough investigation and validation
            allocation = {PHASE_INVESTIGATION: 0.30, PHASE_PLANNING: 0.15, PHASE_IMPLEMENTATION: 0.40, PHASE_VALIDATION: 0.15}
        elif self.complexity["level"] == "MEDIUM":
            # Medium complexity: balanced approach
            allocation = {PHASE_INVESTIGATION: 0.25, PHASE_PLANNING: 0.15, PHASE_IMPLEMENTATION: 0.45, PHASE_VALIDATION: 0.15}
        else:
            # Low complexity: streamlined workflow
            allocation = {PHASE_INVESTIGATION: 0.20, PHASE_PLANNING: 0.10, PHASE_IMPLEMENTATION: 0.55, PHASE_VALIDATION: 0.15}

        # Adjust based on specific indicators
        if self.complexity["indicators"].get("algorithm"):
            allocation[PHASE_PLANNING] += 0.05
            allocation[PHASE_IMPLEMENTATION] -= 0.05

        if self.complexity["indicators"].get("edge_cases"):
            allocation[PHASE_VALIDATION] += 0.05
            allocation[PHASE_IMPLEMENTATION] -= 0.05

        # Convert to actual step counts
        return {phase: max(int(ratio * self.total_steps), 10) for phase, ratio in allocation.items()}  # Minimum 10 steps per phase

    def should_transition(self, current_step: int, cot: "EnhancedCOT") -> tuple[bool, str]:
        """Determine if phase should transition"""

        steps_in_phase = current_step - self.phase_start_step
        allocated_steps = self.step_allocation[self.current_phase]

        # Check if allocated steps for this phase are exhausted
        if steps_in_phase >= allocated_steps:
            next_phase = self._get_next_phase()
            if next_phase:
                return True, next_phase

        # Early transition conditions based on phase goals
        if self.current_phase == PHASE_INVESTIGATION:
            # Transition if we've done sufficient investigation
            if steps_in_phase >= 10 and len(cot.thoughts) >= 10:
                recent_tools = [t.next_tool_name for t in cot.thoughts[-10:]]
                search_count = sum(1 for t in recent_tools if "search" in t or "get_file" in t)

                # If investigation tools used heavily and we have findings
                if search_count >= 6:
                    next_phase = self._get_next_phase()
                    if next_phase:
                        return True, next_phase

        elif self.current_phase == PHASE_PLANNING:
            # Transition when solution is approved
            if len(cot.thoughts) >= 2:
                recent_tools = [t.next_tool_name for t in cot.thoughts[-5:]]
                if "get_approval_for_solution" in recent_tools:
                    next_phase = self._get_next_phase()
                    if next_phase:
                        return True, next_phase

        elif self.current_phase == PHASE_IMPLEMENTATION:
            # Check if significant changes made and tests passing
            if steps_in_phase >= 15 and len(cot.thoughts) >= 15:
                recent_tools = [t.next_tool_name for t in cot.thoughts[-15:]]
                edit_count = sum(1 for t in recent_tools if "edit" in t or "save" in t)
                test_count = sum(1 for t in recent_tools if "test" in t or "run" in t)

                # If we've made changes and run tests
                if edit_count >= 3 and test_count >= 2:
                    next_phase = self._get_next_phase()
                    if next_phase:
                        return True, next_phase

        return False, self.current_phase

    def _get_next_phase(self) -> str:
        """Get the next phase in sequence"""
        phase_sequence = [PHASE_INVESTIGATION, PHASE_PLANNING, PHASE_IMPLEMENTATION, PHASE_VALIDATION]

        try:
            current_index = phase_sequence.index(self.current_phase)
            if current_index < len(phase_sequence) - 1:
                return phase_sequence[current_index + 1]
        except ValueError:
            pass

        return None

    def transition_to_phase(self, new_phase: str, current_step: int):
        """Transition to a new phase"""
        old_phase = self.current_phase
        self.phase_history.append(
            {"phase": old_phase, "start_step": self.phase_start_step, "end_step": current_step, "steps_used": current_step - self.phase_start_step}
        )

        self.current_phase = new_phase
        self.phase_start_step = current_step

    def get_phase_guidance(self) -> str:
        """Get guidance for current phase"""
        return PHASE_SPECIFIC_GUIDANCE.get(self.current_phase, "")

    def create_checkpoint(self, step: int, test_results: dict = None):
        """Save checkpoint for current phase"""
        self.phase_checkpoints[self.current_phase] = {"step": step, "test_results": test_results, "timestamp": time.time()}

    def get_progress_summary(self, current_step: int) -> str:
        """Get summary of progress across phases"""
        steps_in_phase = current_step - self.phase_start_step
        allocated = self.step_allocation[self.current_phase]
        progress_pct = (steps_in_phase / allocated * 100) if allocated > 0 else 0

        summary = f"""
        [PHASE: {self.current_phase}] 
        Progress: {steps_in_phase}/{allocated} steps ({progress_pct:.1f}%)
        Overall: Step {current_step}/{self.total_steps}
        """
        return summary.strip()

    def use_multi_phase_workflow(self) -> bool:
        """Determine if multi-phase workflow should be used"""
        return self.complexity["level"] in ["HIGH", "MEDIUM"]

