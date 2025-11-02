from __future__ import annotations

import os
import sys
from typing import Any, Dict

# Import all extracted components
try:
    from utils_helpers import Utils, FunctionVisitor, VariableNormalizer
    from pev_mcts_framework import MCTS, MCTSNode
    from pev_verifier_framework import StrategicPlanner, Verifier, PEVWorkflow
    from phase_manager_ext import PhaseManager, PhaseConstants
    from tool_manager_ext import FixTaskEnhancedToolManager
    from create_tasks_ext import (
        process_create_task, process_create_task_streamlined,
        validate_edge_case_comments, analyze_missing_edge_cases,
        generate_initial_solution, determine_model_order,
        generate_solution_with_multi_step_reasoning,
        generate_testcases_with_multi_step_reasoning
    )
except ImportError as e:
    print(f"[WARNING] Some modules failed to import: {e}")
    print("[INFO] Attempting to import from fallback location...")

import requests
import subprocess
import json
import re
import textwrap
import time
from enum import Enum
from pathlib import Path

# ============================================================================
# CONSTANTS & GLOBALS
# ============================================================================

run_id = None
DEFAULT_PROXY_URL = "http://localhost:5000"
DEFAULT_TIMEOUT = 300

PROBLEM_TYPE_CHECK_PROMPT = textwrap.dedent(
    '''
    You are the problem type checker that will categories problem type into:
    
    1. CREATE: If the problem statement is about creating a new functionality from scratch.
    2. FIX: If the problem statement is about fixing a bug, creating a new functionality or improving the existing codebase.
    
    Only respond with the "FIX" or "CREATE".
    '''
)

PROBLEM_TYPE_FIX = "FIX"
PROBLEM_TYPE_CREATE = "CREATE"

# ============================================================================
# PROMPTS (CORE ONLY - Others moved to prompts_ext.py)
# ============================================================================

FIX_TASK_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert software engineer. Analyze the problem and repository structure, then solve the bug systematically.
    
    IMPORTANT: 
    - Generate only UNIFIED DIFF format patches
    - Use `git diff` compatible format
    - Include proper context lines
    - Do NOT generate "observation:" - it will be provided
    - Be precise and minimal in changes
    """
)

AGENT_ROUTER_PROMPT = textwrap.dedent(
    """
    Analyze this problem and repository structure. Determine if we should use:
    - NCTS_AGENT: For complex, research-style problems requiring deep exploration
    - STEAMEDLINE_AGENT: For straightforward bug fixes and simple tasks
    
    Only respond with ONE of: NCTS_AGENT or STEAMEDLINE_AGENT
    """
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_git_initialized():
    """Ensure git is initialized in repo."""
    if not os.path.exists(".git"):
        subprocess.run(["git", "init"], capture_output=True)
        subprocess.run(["git", "config", "user.email", "agent@ridges.ai"], capture_output=True)
        subprocess.run(["git", "config", "user.name", "Ridges Agent"], capture_output=True)

def set_env_for_agent():
    """Set environment variables for the agent."""
    pass

def get_directory_tree(root: str = ".", max_depth: int = 3) -> str:
    """Get directory tree structure."""
    result = []
    for root_dir, dirs, files in os.walk(root):
        level = root_dir.replace(root, "").count(os.sep)
        if level < max_depth:
            indent = " " * 2 * level
            result.append(f"{indent}{os.path.basename(root_dir)}/")
            subindent = " " * 2 * (level + 1)
            for file in files[:10]:
                result.append(f"{subindent}{file}")
    return "\n".join(result[:100])

def check_problem_type(problem_statement: str) -> str:
    """Check if problem is FIX or CREATE type."""
    if "fix" in problem_statement.lower() or "bug" in problem_statement.lower():
        return PROBLEM_TYPE_FIX
    return PROBLEM_TYPE_CREATE

def select_agent_strategy(problem_statement: str, repo_structure: str) -> Dict[str, Any]:
    """Select between NCTS and STREAMLINED agents."""
    return {
        "agent": "NCTS_AGENT",
        "confidence": 0.8,
        "reasoning": "Using default NCTS strategy"
    }

def process_fix_task(input_dict: Dict[str, Any], enable_pev: bool = True, enable_mcts: bool = True) -> Dict[str, str]:
    """Process a FIX task."""
    return {
        "patch": "--- a/test.py\n+++ b/test.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
    }

# ============================================================================
# MAIN AGENT ENTRY POINT
# ============================================================================

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo", enable_pev: bool = True, enable_mcts: bool = True) -> Dict[str, str]:
    """
    Main entry point for the Ridges agent.
    
    Args:
        input_dict: Dictionary with 'problem_statement' and optional 'run_id'
        repo_dir: Path to repository root
        enable_pev: Enable PEV workflow
        enable_mcts: Enable MCTS exploration
        
    Returns:
        Dictionary with 'patch' key containing unified diff
    """
    global run_id, DEFAULT_PROXY_URL
    
    run_id = input_dict.get("run_id", os.getenv("RUN_ID", ""))
    repo_dir = os.path.abspath(repo_dir)
    
    sys.path.insert(0, repo_dir)
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)
    
    ensure_git_initialized()
    set_env_for_agent()
    
    try:
        problem_type = check_problem_type(input_dict.get("problem_statement", ""))
        
        if problem_type == PROBLEM_TYPE_FIX:
            result = process_fix_task(input_dict, enable_pev=enable_pev, enable_mcts=enable_mcts)
        else:
            result = process_create_task(input_dict, enable_pev=enable_pev, enable_mcts=enable_mcts)
    
    except Exception as e:
        print(f"[ERROR] Agent exception: {e}")
        result = {"patch": ""}
    
    finally:
        try:
            os.system("git reset --hard")
        except:
            pass
    
    return result


# ============================================================================
# PLACEHOLDER IMPLEMENTATIONS (These should import from extracted modules)
# ============================================================================

class EnhancedNetwork:
    """Placeholder - actual implementation in full agent"""
    pass

class SummarizedCOT:
    """Placeholder - actual implementation in full agent"""
    pass

class EnhancedToolManager:
    """Placeholder - actual implementation in full agent"""
    pass
