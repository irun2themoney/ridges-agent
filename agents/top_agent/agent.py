from __future__ import annotations

import os
import sys
import subprocess
import json
from typing import Any, Dict

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

run_id = None
DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2000"))

PROBLEM_TYPE_FIX = "FIX"
PROBLEM_TYPE_CREATE = "CREATE"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_git_initialized():
    """Ensure git is initialized in repo."""
    if not os.path.exists(".git"):
        subprocess.run(["git", "init"], capture_output=True)
        subprocess.run(["git", "config", "user.email", "agent@ridges.ai"], capture_output=True)
        subprocess.run(["git", "config", "user.name", "Ridges Agent"], capture_output=True)

def set_env_for_agent():
    """Set environment variables for the agent."""
    if os.getcwd() not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.getcwd()

def check_problem_type(problem_statement: str) -> str:
    """Determine if problem is FIX or CREATE type."""
    if "fix" in problem_statement.lower() or "bug" in problem_statement.lower():
        return PROBLEM_TYPE_FIX
    return PROBLEM_TYPE_CREATE

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

# ============================================================================
# PLACEHOLDER STUBS (for compatibility with extracted modules)
# ============================================================================

class EnhancedNetwork:
    """Placeholder for core network inference."""
    pass

class SummarizedCOT:
    """Placeholder for chain-of-thought tracking."""
    pass

class EnhancedToolManager:
    """Placeholder for tool management."""
    pass

# ============================================================================
# MAIN AGENT ENTRY POINT
# ============================================================================

def agent_main(
    input_dict: Dict[str, Any],
    repo_dir: str = "repo",
    enable_pev: bool = True,
    enable_mcts: bool = True
) -> Dict[str, str]:
    """
    Main entry point for the Ridges agent.
    
    Implements the required Ridges interface:
    - Input: Dictionary with 'problem_statement' and 'run_id'
    - Output: Dictionary with 'patch' key containing unified diff
    
    Args:
        input_dict: Dictionary with problem_statement and run_id
        repo_dir: Path to repository root
        enable_pev: Enable PEV workflow (Plan-Execute-Verify)
        enable_mcts: Enable MCTS exploration
        
    Returns:
        Dictionary with 'patch' key containing unified diff
    """
    global run_id
    
    # Initialize global state
    run_id = input_dict.get("run_id", os.getenv("RUN_ID", ""))
    repo_dir = os.path.abspath(repo_dir)
    
    # Setup workspace
    sys.path.insert(0, repo_dir)
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)
    
    ensure_git_initialized()
    set_env_for_agent()
    
    try:
        problem_statement = input_dict.get("problem_statement", "")
        
        # Determine problem type
        problem_type = check_problem_type(problem_statement)
        
        # Route to appropriate processing function
        if problem_type == PROBLEM_TYPE_FIX:
            # Try to import and use actual fix task processor
            try:
                from create_tasks_ext import process_create_task
                result_patch = process_create_task(input_dict, enable_pev=enable_pev, enable_mcts=enable_mcts)
            except Exception as e:
                # Fallback: return empty patch if actual processor fails
                result_patch = ""
        else:
            # CREATE task - try to import actual processor
            try:
                from create_tasks_ext import process_create_task_streamlined
                result_patch = process_create_task_streamlined(input_dict, enable_pev=enable_pev, enable_mcts=enable_mcts)
            except Exception as e:
                # Fallback: return empty patch
                result_patch = ""
        
        # Ensure we have a valid result
        if isinstance(result_patch, dict) and "patch" in result_patch:
            return result_patch
        elif isinstance(result_patch, str):
            return {"patch": result_patch}
        else:
            return {"patch": ""}
    
    except Exception as e:
        # Return safe fallback if anything fails
        return {"patch": ""}
    
    finally:
        # Cleanup: reset git state
        try:
            os.system("git reset --hard")
        except:
            pass


# ============================================================================
# SCRIPT ENTRY POINT (for testing)
# ============================================================================

if __name__ == "__main__":
    # Simple test
    test_input = {
        "problem_statement": "Test problem",
        "run_id": "test-123"
    }
    result = agent_main(test_input)
    print(f"Result: {json.dumps(result, indent=2)}")
