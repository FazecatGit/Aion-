from importlib.resources import path
import subprocess
from pathlib import Path

def read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def write_file(path: str, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")

def run_git_command(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout

def list_files(path: str) -> list[str]:
    return [str(p) for p in Path(path).rglob("*") if p.is_file()]

def run_python_file(path: str) -> str:
    result = subprocess.run(
        ["python", path],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def run_shell_command(command: str) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        shell=True,
        check=False,
    )
    return result.stdout

def show_diff(original: str, modified: str) -> str:
    import difflib
    diff = difflib.unified_diff(
        original.splitlines(),
        modified.splitlines(),
        fromfile="original",
        tofile="modified",
        lineterm=""
    )
    return "\n".join(diff)
