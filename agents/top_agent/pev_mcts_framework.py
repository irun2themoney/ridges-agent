"""
PEV (Plan-Execute-Verify) Workflow and MCTS (Monte Carlo Tree Search) Framework
Extracted from agent.py to reduce main file size while maintaining functionality.

This module contains the strategic planning, verification, and tree search components
used by the agent for complex problem-solving.
"""

import math
from typing import Dict, Any, List


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
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        return len(self.unexplored_actions) == 0

    def best_child(self) -> 'MCTSNode':
        return max(self.children, key=lambda c: c.ucb1_score())

    def add_child(self, action: str, state: str) -> 'MCTSNode':
        child = MCTSNode(state, action, self)
        self.children.append(child)
        return child


class MCTS:
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


__all__ = ['MCTSNode', 'MCTS']
