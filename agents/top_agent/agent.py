import os
import json
import difflib
import pathlib
from typing import List, Tuple
import requests

RUN_ID = os.getenv("RUN_ID")
SANDBOX_PROXY_URL = os.getenv("SANDBOX_PROXY_URL")

# Inference helpers use the gateway provided by the validator

def call_inference(model: str, temperature: float, messages: List[dict]) -> str:
    payload = {"run_id": RUN_ID, "model": model, "temperature": temperature, "messages": messages}
    r = requests.post(f"{SANDBOX_PROXY_URL}/api/inference", headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    r.raise_for_status()
    return r.text.strip('"')


def read_problem_statement() -> str:
    # Provided by validator via input_data
    # The AGENT_RUNNER passes input JSON to stdin; in our env, we rely on the environment contract used in ridges/agent.py
    # For parity, read from a file if present; otherwise, fall back to minimal.
    possible_input = pathlib.Path("/sandbox/input.json")
    if possible_input.exists():
        data = json.loads(possible_input.read_text())
        return data.get("problem_statement", "")
    return ""


def list_repo_files(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = os.path.join(dirpath, name)
            # limit size to keep prompts reasonable
            if os.path.getsize(p) <= 100_000:
                files.append(p)
    return sorted(files)


def snapshot_files(paths: List[str]) -> List[Tuple[str, str]]:
    results = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                results.append((p, f.read()))
        except Exception:
            # Binary or unreadable
            continue
    return results


def propose_changes(problem_statement: str, repo_snapshot: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    # Build a compact context
    tree_lines = []
    for p, _ in repo_snapshot:
        rel = os.path.relpath(p, "/sandbox/repo")
        tree_lines.append(rel)
    tree_str = "\n".join(tree_lines[:200])  # cap tree for prompt brevity

    # Focus on main.py if present (polyglot tasks) else let model choose
    main_file = None
    for p, _ in repo_snapshot:
        if p.endswith("/main.py"):
            main_file = p
            break

    files_excerpt = []
    consider = [main_file] if main_file else [p for p, _ in repo_snapshot[:5]]
    for p, content in repo_snapshot:
        if p in consider:
            rel = os.path.relpath(p, "/sandbox/repo")
            excerpt = content[:8000]
            files_excerpt.append(f"--- {rel} ---\n{excerpt}")
    files_blob = "\n\n".join(files_excerpt)

    system = {
        "role": "system",
        "content": (
            "You are a senior software engineer agent competing on a strict evaluator. "
            "Modify only what is required to fully solve the problem. Keep code clean, deterministic, and testable. "
            "Return ONLY the complete new content for the specific file(s) you change, in JSON with keys: file_path and new_content."
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"Problem statement:\n{problem_statement}\n\n"
            f"Repo tree (relative to /sandbox/repo):\n{tree_str}\n\n"
            f"Key files:\n{files_blob}\n\n"
            "Respond with a JSON array. Choose minimal set of files to modify to pass tests (prefer main.py).\n"
            "Example: [{\"file_path\": \"main.py\", \"new_content\": \"<entire file content>\"}]"
        ),
    }

    # Favor deterministic output
    raw = call_inference("moonshotai/Kimi-K2-Instruct", 0.1, [system, user])

    # Be tolerant to minor formatting issues
    start = raw.find("[")
    end = raw.rfind("]")
    json_str = raw[start : end + 1] if start != -1 and end != -1 else raw
    try:
        changes = json.loads(json_str)
        if isinstance(changes, dict):
            changes = [changes]
    except Exception:
        # Fallback: no changes
        changes = []

    normalized = []
    for item in changes:
        fp = item.get("file_path")
        nc = item.get("new_content")
        if not fp or nc is None:
            continue
        # Normalize path relative to repo root
        abs_fp = os.path.join("/sandbox/repo", fp) if not fp.startswith("/sandbox/") else fp
        normalized.append((abs_fp, nc))
    return normalized


def unified_diff(from_path: str, from_text: str, to_path: str, to_text: str) -> str:
    from_lines = from_text.splitlines(keepends=True)
    to_lines = to_text.splitlines(keepends=True)
    rel_from = os.path.relpath(from_path, "/sandbox/repo")
    rel_to = os.path.relpath(to_path, "/sandbox/repo")
    diff = difflib.unified_diff(
        from_lines,
        to_lines,
        fromfile=f"a/{rel_from}",
        tofile=f"b/{rel_to}",
        lineterm="",
    )
    return "\n".join(diff)


def write_and_build_diff(repo_snapshot: List[Tuple[str, str]], changes: List[Tuple[str, str]]) -> str:
    path_to_old = {p: content for p, content in repo_snapshot}
    all_diffs: List[str] = []

    for abs_fp, new_content in changes:
        os.makedirs(os.path.dirname(abs_fp), exist_ok=True)
        old_content = path_to_old.get(abs_fp, "")
        # Write new content
        with open(abs_fp, "w", encoding="utf-8") as f:
            f.write(new_content)
        # Build diff
        all_diffs.append(unified_diff(abs_fp, old_content, abs_fp, new_content))

    # Join diffs
    return "\n".join(d for d in all_diffs if d)


def agent_main(input):
    problem_statement = read_problem_statement() or input.get("problem_statement", "")

    # Snapshot repo
    repo_files = list_repo_files("/sandbox/repo")
    snapshot = snapshot_files(repo_files)

    # Propose minimal change set
    proposed = propose_changes(problem_statement, snapshot)

    # If model returned nothing, attempt a targeted main.py completion
    if not proposed:
        target = None
        for p, _ in snapshot:
            if p.endswith("/main.py"):
                target = p
                break
        if target:
            base = next((c for p, c in snapshot if p == target), "")
            user = [{"role": "user", "content": f"Complete this main.py to solve: \n{problem_statement}\n\nCurrent main.py:\n{base}"}]
            completion = call_inference("moonshotai/Kimi-K2-Instruct", 0.1, user)
            proposed = [(target, completion)]

    # Apply and return patch
    patch = write_and_build_diff(snapshot, proposed)
    return patch if patch.strip() else ""
