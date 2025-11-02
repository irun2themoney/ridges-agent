"""
Phase Manager for Multi-Phase Workflow Orchestration
Extracted from agent.py to reduce main file size.

Manages complex problem-solving workflows by breaking them into distinct phases:
- Investigation: Explore and understand the problem
- Planning: Design the solution approach
- Implementation: Execute the solution  
- Validation: Verify correctness
"""

import re
import time
import math
import os
from typing import Dict, Any

# Phase constants
PHASE_INVESTIGATION = "investigation"
PHASE_PLANNING = "planning"
PHASE_IMPLEMENTATION = "implementation"
PHASE_VALIDATION = "validation"

# Phase-specific guidance
PHASE_SPECIFIC_GUIDANCE = {
    PHASE_INVESTIGATION: "Focus on understanding the problem thoroughly.",
    PHASE_PLANNING: "Plan your approach carefully before implementing.",
    PHASE_IMPLEMENTATION: "Execute your plan step by step.",
    PHASE_VALIDATION: "Verify your solution works correctly.",
}


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
            "multi_file": len(re.findall(r'\bfile[s]?\b', problem_lower)) > 2,
            "algorithm": any(
                kw in problem_lower for kw in
                ['algorithm', 'optimization', 'performance', 'complexity', 'efficient']
            ),
            "edge_cases": any(
                kw in problem_lower for kw in
                ['edge case', 'boundary', 'corner case', 'special case']
            ),
            "refactor": any(
                kw in problem_lower for kw in
                ['refactor', 'redesign', 'restructure', 'rewrite']
            ),
            "debugging": any(
                kw in problem_lower for kw in
                ['bug', 'error', 'crash', 'fail', 'incorrect', 'fix']
            ),
            "multiple_components": len(re.findall(r'\bclass\b|\bfunction\b|\bmethod\b', problem_lower)) > 3,
            "integration": any(
                kw in problem_lower for kw in
                ['integrate', 'interaction', 'between', 'across']
            ),
            "backward_compat": any(
                kw in problem_lower for kw in
                ['backward', 'compatibility', 'breaking', 'legacy']
            )
        }

        score = sum(indicators.values())
        if score >= 5:
            level = "HIGH"
        elif score >= 3:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "level": level,
            "score": score,
            "indicators": indicators
        }

    def _allocate_steps(self) -> dict:
        """Allocate steps to each phase based on complexity"""

        if self.complexity["level"] == "HIGH":
            allocation = {
                PHASE_INVESTIGATION: 0.30,
                PHASE_PLANNING: 0.15,
                PHASE_IMPLEMENTATION: 0.40,
                PHASE_VALIDATION: 0.15
            }
        elif self.complexity["level"] == "MEDIUM":
            allocation = {
                PHASE_INVESTIGATION: 0.25,
                PHASE_PLANNING: 0.15,
                PHASE_IMPLEMENTATION: 0.45,
                PHASE_VALIDATION: 0.15
            }
        else:
            allocation = {
                PHASE_INVESTIGATION: 0.20,
                PHASE_PLANNING: 0.10,
                PHASE_IMPLEMENTATION: 0.55,
                PHASE_VALIDATION: 0.15
            }
        if self.complexity["indicators"].get("algorithm"):
            allocation[PHASE_PLANNING] += 0.05
            allocation[PHASE_IMPLEMENTATION] -= 0.05

        if self.complexity["indicators"].get("edge_cases"):
            allocation[PHASE_VALIDATION] += 0.05
            allocation[PHASE_IMPLEMENTATION] -= 0.05
        return {
            phase: max(int(ratio * self.total_steps), 10)  # Minimum 10 steps per phase
            for phase, ratio in allocation.items()
        }

    def should_transition(self, current_step: int, cot: 'SummarizedCOT') -> tuple[bool, str]:
        """Determine if phase should transition"""

        steps_in_phase = current_step - self.phase_start_step
        allocated_steps = self.step_allocation[self.current_phase]
        if steps_in_phase >= allocated_steps:
            next_phase = self._get_next_phase()
            if next_phase:
                return True, next_phase
        if self.current_phase == PHASE_INVESTIGATION:
            if steps_in_phase >= 10 and len(cot.thoughts) >= 10:
                recent_tools = [t.next_tool_name for t in cot.thoughts[-10:]]
                search_count = sum(1 for t in recent_tools if 'search' in t or 'get_file' in t)
                if search_count >= 6:
                    next_phase = self._get_next_phase()
                    if next_phase:
                        return True, next_phase

        elif self.current_phase == PHASE_PLANNING:
            if len(cot.thoughts) >= 2:
                recent_tools = [t.next_tool_name for t in cot.thoughts[-5:]]
                if 'get_approval_for_solution' in recent_tools:
                    next_phase = self._get_next_phase()
                    if next_phase:
                        return True, next_phase

        elif self.current_phase == PHASE_IMPLEMENTATION:
            if steps_in_phase >= 15 and len(cot.thoughts) >= 15:
                recent_tools = [t.next_tool_name for t in cot.thoughts[-15:]]
                edit_count = sum(1 for t in recent_tools if 'edit' in t or 'save' in t)
                test_count = sum(1 for t in recent_tools if 'test' in t or 'run' in t)
                if edit_count >= 3 and test_count >= 2:
                    next_phase = self._get_next_phase()
                    if next_phase:
                        return True, next_phase

        return False, self.current_phase

    def _get_next_phase(self) -> str:
        """Get the next phase in sequence"""
        phase_sequence = [
            PHASE_INVESTIGATION,
            PHASE_PLANNING,
            PHASE_IMPLEMENTATION,
            PHASE_VALIDATION
        ]

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
            {
                "phase": old_phase,
                "start_step": self.phase_start_step,
                "end_step": current_step,
                "steps_used": current_step - self.phase_start_step
            }
        )

        self.current_phase = new_phase
        self.phase_start_step = current_step

    def get_phase_guidance(self) -> str:
        """Get guidance for current phase"""
        return PHASE_SPECIFIC_GUIDANCE.get(self.current_phase, "")

    def create_checkpoint(self, step: int, test_results: dict = None):
        """Save checkpoint for current phase"""
        self.phase_checkpoints[self.current_phase] = {
            "step": step,
            "test_results": test_results,
            "timestamp": time.time()
        }

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


__all__ = ['PhaseManager', 'PHASE_INVESTIGATION', 'PHASE_PLANNING', 'PHASE_IMPLEMENTATION', 'PHASE_VALIDATION']
