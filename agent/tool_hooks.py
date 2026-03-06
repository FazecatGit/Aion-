"""
Tool hooks: linting, test automation, log parsing, git analysis.

Each hook runs a CLI tool, parses output, and returns structured JSON
suitable for feeding back into the agent's edit loop.
"""

import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("tool_hooks")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: str | None = None, timeout: int = 60) -> dict:
    """Run a command and return {stdout, stderr, returncode}."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout
        )
        return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}
    except FileNotFoundError:
        return {"stdout": "", "stderr": f"Command not found: {cmd[0]}", "returncode": -1}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timeout after {timeout}s", "returncode": -2}


def _detect_language(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    lang_map = {
        ".py": "python", ".go": "go", ".ts": "typescript", ".tsx": "typescript",
        ".js": "javascript", ".jsx": "javascript", ".cpp": "cpp", ".c": "c",
        ".h": "cpp", ".hpp": "cpp", ".rs": "rust",
    }
    return lang_map.get(ext, "unknown")


# ═══════════════════════════════════════════════════════════════════════════════
# LINTING & STATIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_ruff(file_path: str, fix: bool = False) -> dict:
    """Run ruff linter on a Python file. Returns structured diagnostics."""
    cmd = ["ruff", "check", file_path, "--output-format", "json"]
    if fix:
        cmd.append("--fix")
    result = _run(cmd)
    diagnostics = []
    if result["stdout"].strip():
        try:
            diagnostics = json.loads(result["stdout"])
        except json.JSONDecodeError:
            diagnostics = [{"raw": result["stdout"]}]

    return {
        "tool": "ruff",
        "file": file_path,
        "diagnostics": diagnostics,
        "count": len(diagnostics),
        "stderr": result["stderr"],
    }


def run_pylint(file_path: str) -> dict:
    """Run pylint on a Python file. Returns structured diagnostics."""
    cmd = ["pylint", file_path, "--output-format", "json", "--disable=C,R"]
    result = _run(cmd)
    diagnostics = []
    if result["stdout"].strip():
        try:
            diagnostics = json.loads(result["stdout"])
        except json.JSONDecodeError:
            diagnostics = [{"raw": result["stdout"]}]
    return {
        "tool": "pylint",
        "file": file_path,
        "diagnostics": diagnostics,
        "count": len(diagnostics),
    }


def run_eslint(file_path: str, fix: bool = False) -> dict:
    """Run eslint on a TypeScript/JavaScript file."""
    cmd = ["npx", "eslint", file_path, "--format", "json"]
    if fix:
        cmd.append("--fix")
    cwd = str(Path(file_path).parent)
    result = _run(cmd, cwd=cwd)
    diagnostics = []
    if result["stdout"].strip():
        try:
            parsed = json.loads(result["stdout"])
            if parsed and isinstance(parsed, list):
                diagnostics = parsed[0].get("messages", [])
        except json.JSONDecodeError:
            diagnostics = [{"raw": result["stdout"]}]
    return {
        "tool": "eslint",
        "file": file_path,
        "diagnostics": diagnostics,
        "count": len(diagnostics),
    }


def run_go_vet(file_path: str) -> dict:
    """Run go vet on a Go file/package."""
    cwd = str(Path(file_path).parent)
    result = _run(["go", "vet", "./..."], cwd=cwd)
    issues = [line for line in result["stderr"].splitlines() if line.strip()]
    return {
        "tool": "go_vet",
        "file": file_path,
        "diagnostics": issues,
        "count": len(issues),
    }


def run_staticcheck(file_path: str) -> dict:
    """Run staticcheck on a Go file/package."""
    cwd = str(Path(file_path).parent)
    result = _run(["staticcheck", "./..."], cwd=cwd)
    issues = [line for line in result["stdout"].splitlines() if line.strip()]
    return {
        "tool": "staticcheck",
        "file": file_path,
        "diagnostics": issues,
        "count": len(issues),
    }


def run_mypy(file_path: str) -> dict:
    """Run mypy type checker on a Python file."""
    result = _run(["mypy", file_path, "--no-color-output"])
    issues = [line for line in result["stdout"].splitlines() if line.strip() and "error" in line.lower()]
    return {
        "tool": "mypy",
        "file": file_path,
        "diagnostics": issues,
        "count": len(issues),
    }


def run_tsc_check(file_path: str) -> dict:
    """Run tsc --noEmit for TypeScript type checking."""
    cwd = str(Path(file_path).parent)
    result = _run(["npx", "tsc", "--noEmit"], cwd=cwd)
    issues = [line for line in result["stdout"].splitlines() if "error TS" in line]
    return {
        "tool": "tsc",
        "file": file_path,
        "diagnostics": issues,
        "count": len(issues),
    }


def lint_file(file_path: str, fix: bool = False) -> dict:
    """Auto-detect language and run the appropriate linter."""
    lang = _detect_language(file_path)
    if lang == "python":
        return run_ruff(file_path, fix=fix)
    elif lang == "go":
        return run_go_vet(file_path)
    elif lang in ("typescript", "javascript"):
        return run_eslint(file_path, fix=fix)
    elif lang == "cpp" or lang == "c":
        # Basic: compile with warnings
        compiler = "g++" if lang == "cpp" else "gcc"
        result = _run([compiler, "-fsyntax-only", "-Wall", "-Wextra", file_path])
        issues = [line for line in result["stderr"].splitlines() if line.strip()]
        return {"tool": compiler, "file": file_path, "diagnostics": issues, "count": len(issues)}
    return {"tool": "none", "file": file_path, "diagnostics": [], "count": 0, "message": f"No linter for {lang}"}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST AUTOMATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_pytest(target: str = ".", args: list[str] | None = None, cwd: str | None = None) -> dict:
    """Run pytest and return structured results."""
    cmd = ["python", "-m", "pytest", target, "-v", "--tb=short", "--no-header"]
    if args:
        cmd.extend(args)
    result = _run(cmd, cwd=cwd, timeout=120)
    lines = result["stdout"].splitlines()
    passed = sum(1 for l in lines if " PASSED" in l)
    failed = sum(1 for l in lines if " FAILED" in l)
    errors = sum(1 for l in lines if " ERROR" in l)
    return {
        "tool": "pytest",
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "all_passed": failed == 0 and errors == 0,
        "output": result["stdout"],
        "stderr": result["stderr"],
    }


def run_pytest_with_coverage(target: str = ".", cwd: str | None = None) -> dict:
    """Run pytest with coverage and return results + coverage data."""
    cmd = ["python", "-m", "pytest", target, "-v", "--tb=short",
           "--cov", "--cov-report=json", "--no-header"]
    result = _run(cmd, cwd=cwd, timeout=180)
    lines = result["stdout"].splitlines()
    passed = sum(1 for l in lines if " PASSED" in l)
    failed = sum(1 for l in lines if " FAILED" in l)

    coverage_data = {}
    cov_file = Path(cwd or ".") / "coverage.json"
    if cov_file.exists():
        try:
            with open(cov_file) as f:
                cov_raw = json.load(f)
            total = cov_raw.get("totals", {}).get("percent_covered", 0)
            files = {}
            for fpath, fdata in cov_raw.get("files", {}).items():
                files[fpath] = {
                    "covered": fdata.get("summary", {}).get("percent_covered", 0),
                    "missing_lines": fdata.get("missing_lines", []),
                }
            coverage_data = {"total_percent": round(total, 1), "files": files}
        except Exception:
            pass

    return {
        "tool": "pytest+coverage",
        "passed": passed,
        "failed": failed,
        "all_passed": failed == 0,
        "coverage": coverage_data,
        "output": result["stdout"],
    }


def compare_test_results(current: dict, previous: dict) -> dict:
    """Compare two test run results and flag regressions."""
    regressions = []
    current_lines = current.get("output", "").splitlines()
    previous_lines = previous.get("output", "").splitlines()

    current_tests = {l.split("::")[1].split(" ")[0]: "PASSED" if "PASSED" in l else "FAILED"
                     for l in current_lines if "::" in l and ("PASSED" in l or "FAILED" in l)}
    previous_tests = {l.split("::")[1].split(" ")[0]: "PASSED" if "PASSED" in l else "FAILED"
                      for l in previous_lines if "::" in l and ("PASSED" in l or "FAILED" in l)}

    for test_name, status in current_tests.items():
        prev_status = previous_tests.get(test_name)
        if prev_status == "PASSED" and status == "FAILED":
            regressions.append(test_name)

    return {
        "regressions": regressions,
        "new_failures": len(regressions),
        "previously_passing": len([t for t in previous_tests.values() if t == "PASSED"]),
        "currently_passing": len([t for t in current_tests.values() if t == "PASSED"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOG & OBSERVABILITY HOOKS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_structured_logs(log_text: str, max_groups: int = 20) -> dict:
    """Parse structured logs, group repeated errors, and summarize."""
    import re
    lines = log_text.splitlines()
    error_patterns: dict[str, list[int]] = {}
    errors = []

    for i, line in enumerate(lines, 1):
        if re.search(r'\b(ERROR|CRITICAL|FATAL|Exception|Traceback)\b', line, re.IGNORECASE):
            errors.append({"line": i, "text": line.strip()})
            # Group by first 80 chars of normalized text
            key = re.sub(r'\d+', 'N', line.strip()[:80])
            error_patterns.setdefault(key, []).append(i)

    # Sort groups by frequency
    groups = sorted(error_patterns.items(), key=lambda x: -len(x[1]))[:max_groups]
    grouped = [{"pattern": k, "count": len(v), "first_line": v[0], "last_line": v[-1]} for k, v in groups]

    return {
        "total_errors": len(errors),
        "unique_patterns": len(error_patterns),
        "top_groups": grouped,
        "sample_errors": errors[:10],
    }


def read_log_file(log_path: str, tail: int = 500) -> dict:
    """Read the last N lines of a log file and parse for errors."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        tail_text = "".join(lines[-tail:])
        return {
            "status": "ok",
            "total_lines": len(lines),
            "tail_lines": min(tail, len(lines)),
            "analysis": parse_structured_logs(tail_text),
        }
    except FileNotFoundError:
        return {"status": "error", "error": f"Log file not found: {log_path}"}


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE CONTROL & CODE REVIEW
# ═══════════════════════════════════════════════════════════════════════════════

def git_diff(ref: str = "HEAD", file_path: str | None = None, cwd: str | None = None) -> dict:
    """Get git diff for review. Returns structured diff output."""
    cmd = ["git", "diff", ref]
    if file_path:
        cmd.extend(["--", file_path])
    result = _run(cmd, cwd=cwd)
    diff_text = result["stdout"]

    # Parse stats
    stat_result = _run(["git", "diff", ref, "--stat"], cwd=cwd)
    files_changed = 0
    insertions = 0
    deletions = 0
    for line in stat_result["stdout"].splitlines():
        if "file" in line and "changed" in line:
            parts = line.split(",")
            for part in parts:
                part = part.strip()
                if "file" in part:
                    files_changed = int(part.split()[0])
                elif "insertion" in part:
                    insertions = int(part.split()[0])
                elif "deletion" in part:
                    deletions = int(part.split()[0])

    return {
        "tool": "git_diff",
        "ref": ref,
        "diff": diff_text,
        "stats": {
            "files_changed": files_changed,
            "insertions": insertions,
            "deletions": deletions,
        },
    }


def git_diff_staged(cwd: str | None = None) -> dict:
    """Get staged changes diff (what's about to be committed)."""
    cmd = ["git", "diff", "--cached"]
    result = _run(cmd, cwd=cwd)
    return {
        "tool": "git_diff_staged",
        "diff": result["stdout"],
    }


def git_log_summary(count: int = 20, cwd: str | None = None) -> dict:
    """Generate a readable changelog from recent commits."""
    cmd = ["git", "log", f"-{count}", "--pretty=format:%h|%an|%ar|%s"]
    result = _run(cmd, cwd=cwd)
    commits = []
    for line in result["stdout"].splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            commits.append({
                "hash": parts[0],
                "author": parts[1],
                "time_ago": parts[2],
                "message": parts[3],
            })
    return {
        "tool": "git_log",
        "commits": commits,
        "count": len(commits),
    }


def git_blame_line(file_path: str, line_number: int, cwd: str | None = None) -> dict:
    """Get blame info for a specific line."""
    cmd = ["git", "blame", "-L", f"{line_number},{line_number}", "--porcelain", file_path]
    result = _run(cmd, cwd=cwd)
    info = {"line": line_number, "file": file_path}
    for bline in result["stdout"].splitlines():
        if bline.startswith("author "):
            info["author"] = bline[7:]
        elif bline.startswith("author-time "):
            info["timestamp"] = bline[12:]
        elif bline.startswith("summary "):
            info["commit_message"] = bline[8:]
    return {"tool": "git_blame", **info}


def pre_commit_check(cwd: str | None = None) -> dict:
    """Run pre-commit style checks on staged files.

    Checks:
    - TODO/FIXME/debug print detection
    - Naming convention violations (basic)
    - Large file detection
    """
    # Get staged files
    result = _run(["git", "diff", "--cached", "--name-only"], cwd=cwd)
    staged_files = [f.strip() for f in result["stdout"].splitlines() if f.strip()]

    issues = []
    for fpath in staged_files:
        full_path = Path(cwd or ".") / fpath
        if not full_path.exists() or not full_path.is_file():
            continue

        # Skip binary files
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            # Debug/TODO detection
            for pattern in ["TODO", "FIXME", "HACK", "XXX"]:
                if pattern in line and not line.strip().startswith("#!"):
                    issues.append({
                        "file": fpath, "line": i, "severity": "warning",
                        "message": f"Found {pattern} comment",
                    })
            # Debug print detection
            if _detect_language(fpath) == "python":
                stripped = line.strip()
                if stripped.startswith("print(") and "debug" in stripped.lower():
                    issues.append({
                        "file": fpath, "line": i, "severity": "warning",
                        "message": "Debug print statement detected",
                    })
                if stripped.startswith("breakpoint(") or stripped.startswith("import pdb"):
                    issues.append({
                        "file": fpath, "line": i, "severity": "error",
                        "message": "Debugger statement left in code",
                    })

        # Large file check (> 500KB)
        size = full_path.stat().st_size
        if size > 500_000:
            issues.append({
                "file": fpath, "line": 0, "severity": "warning",
                "message": f"Large file ({size // 1024}KB) — consider if this belongs in git",
            })

    return {
        "tool": "pre_commit_check",
        "staged_files": staged_files,
        "issues": issues,
        "issue_count": len(issues),
        "pass": len([i for i in issues if i["severity"] == "error"]) == 0,
    }
