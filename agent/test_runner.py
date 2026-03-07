import os
import sys
import subprocess
import tempfile
import logging
import shutil
from typing import List, Dict, Optional, Tuple

from langchain_ollama import OllamaLLM
from brain.config import LLM_MODEL, LANG_FENCE
from .code_editing_helpers import strip_markdown

logger = logging.getLogger("code_agent")


def _find_cpp_compiler() -> Tuple[Optional[str], str]:
    """Find an available C++ compiler on the system.
    
    Returns: (compiler_path, compiler_name) or (None, error_message)
    """
    # Try g++ first (MinGW on Windows or GCC on Unix)
    if shutil.which("g++"):
        return "g++", "g++"
    
    # Try clang++ (LLVM/Clang)
    if shutil.which("clang++"):
        return "clang++", "clang++"
    
    # Try cl.exe (MSVC on Windows)
    if shutil.which("cl"):
        return "cl", "MSVC"
    
    # No compiler found
    error = (
        "No C++ compiler found. Please install one of:\n"
        "  Windows: MinGW (g++), LLVM (clang++), or Visual Studio (cl.exe)\n"
        "  macOS: Xcode Command Line Tools (g++ or clang++)\n"
        "  Linux: GCC (g++) or Clang (clang++)\n"
        "\nTo install MinGW on Windows: https://www.mingw-w64.org/downloads/"
    )
    return None, error


def _find_c_compiler() -> Tuple[Optional[str], str]:
    """Find an available C compiler on the system.
    
    Returns: (compiler_path, compiler_name) or (None, error_message)
    """
    if shutil.which("gcc"):
        return "gcc", "gcc"
    if shutil.which("clang"):
        return "clang", "clang"
    if shutil.which("cl"):
        return "cl", "MSVC"
    
    error = (
        "No C compiler found. Please install one of:\n"
        "  Windows: MinGW (gcc), LLVM (clang), or Visual Studio (cl.exe)\n"
        "  macOS: Xcode Command Line Tools (gcc or clang)\n"
        "  Linux: GCC (gcc) or Clang (clang)"
    )
    return None, error




def _generate_harness(source: str, ext: str, test_inputs: List[str], expected_outputs: List[str], llm: OllamaLLM) -> str:
    """Ask LLM to write a complete runnable program that calls the function
    once per test input and prints only the return value per line."""
    lang = LANG_FENCE.get(ext, ext.lstrip("."))
    cases_block = "\n".join(
        f"  Test {i + 1}: input={inp}  →  expected output={exp}"
        for i, (inp, exp) in enumerate(zip(test_inputs, expected_outputs))
    )

    prompt = (
        f"Given this {lang} source code:\n"
        f"```{lang}\n{source}\n```\n\n"
        f"Write a complete, self-contained, runnable {lang} program that:\n"
        f"1. Contains the EXACT function(s) and type definitions from above — copy them verbatim, do NOT modify their logic\n"
        f"2. Has a main entry point that calls the function once for each test input below\n"
        f"3. Prints ONLY the return value of each call, one result per line, in order\n"
        f"4. No debug output, no labels, no extra text — just one result per line\n"
        f"5. The printed output for each test MUST match the expected output format exactly\n\n"
        f"Test cases:\n{cases_block}\n\n"
        f"Language-specific rules:\n"
        f"- Go: use 'package main' and 'import \"fmt\"'. Use fmt.Println() for simple values. "
        f"If the function returns a linked list / pointer-based structure, traverse it and print in the same format as the expected output.\n"
        f"- Python: define the function first, then use 'if __name__ == \"__main__\":' block.\n"
        f"- C++: use #include <iostream>, #include <vector>, #include <algorithm>, #include <climits> as needed. "
        f"Use std::cout << result << std::endl for each result. Parse the test inputs into proper C++ types "
        f"(e.g. vector<int> from array notation). Include a main() function.\n"
        f"- C: use #include <stdio.h>. Use printf() for output. Include a main() function.\n"
        f"- If the input contains arrays like [1,2,3], convert them to the appropriate data structure (e.g. linked list, vector).\n"
        f"- The function logic must be IDENTICAL to the original — do NOT 'fix' or change it.\n\n"
        f"Output ONLY the complete program — no explanation, no markdown fences."
    )

    harness = llm.invoke(prompt).strip()
    if "```" in harness:
        harness = strip_markdown(harness)
    return harness


def run_tests(
    source: str,
    path: str,
    test_cases: List[Dict[str, str]],
    llm: Optional[OllamaLLM] = None
) -> List[Dict]:
    """
    Run test cases against source code by generating a harness and executing it.

    Args:
        source:     Current source code string.
        path:       Original file path (used to detect language via extension).
        test_cases: List of {"input": str, "expected": str} dicts.
        llm:        Optional LLM instance; creates one if not provided.

    Returns:
        List of {"input", "expected", "actual", "passed", "error"} dicts.
    """
    ext = os.path.splitext(path)[1].lower()

    if llm is None:
        llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)

    inputs = [tc["input"] for tc in test_cases]
    expected_outputs = [tc["expected"] for tc in test_cases]
    logger.info("[TEST_RUNNER] Generating harness for %d test case(s) [%s]...", len(test_cases), ext)

    try:
        harness = _generate_harness(source, ext, inputs, expected_outputs, llm)
    except Exception as e:
        logger.error("[TEST_RUNNER] Harness generation failed: %s", e)
        err = f"Harness generation failed: {e}"
        return [
            {"input": tc["input"], "expected": tc["expected"], "actual": None, "passed": False, "error": err}
            for tc in test_cases
        ]

    logger.debug("[TEST_RUNNER] Generated harness:\n%s", harness[:800])

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pick harness file name and run command by language
            if ext == ".go":
                harness_path = os.path.join(tmpdir, "main.go")
                cmd = ["go", "run", harness_path]
            elif ext == ".py":
                harness_path = os.path.join(tmpdir, "harness.py")
                cmd = [sys.executable, harness_path]
            elif ext == ".js":
                harness_path = os.path.join(tmpdir, "harness.js")
                cmd = ["node", harness_path]
            elif ext == ".cpp":
                harness_path = os.path.join(tmpdir, "harness.cpp")
                exe_path = os.path.join(tmpdir, "harness.exe" if os.name == "nt" else "harness")
                compiler, compiler_info = _find_cpp_compiler()
                if not compiler:
                    logger.error("[TEST_RUNNER] %s", compiler_info)
                    return [
                        {
                            "input": tc["input"],
                            "expected": tc["expected"],
                            "actual": None,
                            "passed": False,
                            "error": compiler_info,
                        }
                        for tc in test_cases
                    ]
                compile_cmd = [compiler, "-std=c++17", "-O2", "-o", exe_path, harness_path]
                cmd = None  # will be set after compile
            elif ext == ".c":
                harness_path = os.path.join(tmpdir, "harness.c")
                exe_path = os.path.join(tmpdir, "harness.exe" if os.name == "nt" else "harness")
                compiler, compiler_info = _find_c_compiler()
                if not compiler:
                    logger.error("[TEST_RUNNER] %s", compiler_info)
                    return [
                        {
                            "input": tc["input"],
                            "expected": tc["expected"],
                            "actual": None,
                            "passed": False,
                            "error": compiler_info,
                        }
                        for tc in test_cases
                    ]
                compile_cmd = [compiler, "-std=c11", "-O2", "-o", exe_path, harness_path]
                cmd = None
            else:
                err = f"Unsupported language for test runner: '{ext}'"
                logger.warning("[TEST_RUNNER] %s", err)
                return [
                    {"input": tc["input"], "expected": tc["expected"], "actual": None, "passed": False, "error": err}
                    for tc in test_cases
                ]

            with open(harness_path, "w", encoding="utf-8") as f:
                f.write(harness)

            # Compiled languages (C/C++): compile first, then set cmd to run the executable
            if ext in (".cpp", ".c"):
                logger.info("[TEST_RUNNER] Compiling: %s", " ".join(compile_cmd))
                compile_result = subprocess.run(
                    compile_cmd, capture_output=True, text=True, timeout=30, cwd=tmpdir
                )
                if compile_result.returncode != 0:
                    err_msg = (compile_result.stderr or "Compilation failed").strip()[:800]
                    logger.error("[TEST_RUNNER] Compilation failed:\n%s", err_msg)
                    return [
                        {
                            "input": tc["input"],
                            "expected": tc["expected"],
                            "actual": None,
                            "passed": False,
                            "error": f"Compilation error:\n{err_msg}",
                        }
                        for tc in test_cases
                    ]
                cmd = [exe_path]
                logger.info("[TEST_RUNNER] Compilation successful")

            logger.info("[TEST_RUNNER] Running: %s", " ".join(cmd))
            run_result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=20, cwd=tmpdir
            )

            if run_result.returncode != 0:
                err_msg = (run_result.stderr or "Unknown error").strip()[:800]
                logger.error("[TEST_RUNNER] Harness execution failed:\n%s", err_msg)
                return [
                    {
                        "input": tc["input"],
                        "expected": tc["expected"],
                        "actual": None,
                        "passed": False,
                        "error": f"Compilation/runtime error:\n{err_msg}",
                    }
                    for tc in test_cases
                ]

            output_lines = [l for l in run_result.stdout.split("\n") if l.strip() != ""]
            results = []
            for i, tc in enumerate(test_cases):
                actual   = output_lines[i].strip() if i < len(output_lines) else None
                expected = str(tc["expected"]).strip()
                passed   = actual == expected
                results.append({
                    "input":    tc["input"],
                    "expected": expected,
                    "actual":   actual,
                    "passed":   passed,
                    "error":    None,
                })
                logger.info(
                    "[TEST_RUNNER] Test %d: %s  expected=%s  got=%s",
                    i + 1, "PASS ✓" if passed else "FAIL ✗", expected, actual
                )
            return results

    except subprocess.TimeoutExpired:
        err = "Execution timed out (>20 s) — possible infinite loop"
        logger.warning("[TEST_RUNNER] %s", err)
        return [
            {"input": tc["input"], "expected": tc["expected"], "actual": None, "passed": False, "error": err}
            for tc in test_cases
        ]
    except Exception as e:
        logger.error("[TEST_RUNNER] Unexpected error: %s", e)
        return [
            {"input": tc["input"], "expected": tc["expected"], "actual": None, "passed": False, "error": str(e)}
            for tc in test_cases
        ]


def build_test_failure_note(test_results: List[Dict], instruction: str, source: str = "", llm: Optional[OllamaLLM] = None, attempt: int = 0, failed_approaches: Optional[List[str]] = None) -> str:
    """
    Build a structured prompt note from failing test cases.
    
    Includes a self-reasoning step: the LLM reads the problem, traces through
    the test case, and identifies the root cause before being asked to fix.
    
    On later attempts (attempt >= 2), the prompt explicitly requests a different
    algorithmic approach to break out of circular reasoning.
    """
    failures = [r for r in test_results if not r["passed"]]
    if not failures:
        return ""

    # Build basic failure list
    failure_lines = ""
    for i, r in enumerate(failures):
        got = r["actual"] if r["actual"] is not None else f"ERROR: {r.get('error', 'unknown')}"
        failure_lines += f"  Test {i + 1}: input={r['input']}  →  expected={r['expected']},  got={got}\n"

    # Self-reasoning: LLM traces through the test to identify root cause
    reasoning = ""
    if source and llm:
        try:
            pivot_hint = ""
            if attempt >= 2:
                pivot_hint = (
                    "\nIMPORTANT: Previous fix attempts using the same logic have FAILED REPEATEDLY. "
                    "The current approach is fundamentally flawed.\n"
                    "You MUST identify a DIFFERENT algorithm or strategy — do NOT keep tweaking the same logic.\n"
                )
                if failed_approaches:
                    pivot_hint += "APPROACHES ALREADY TRIED AND FAILED:\n"
                    for fa in failed_approaches:
                        pivot_hint += f"  - {fa}\n"
                    pivot_hint += "Do NOT use any of the above approaches again.\n"

            trace_prompt = (
                f"You are debugging incorrect code. Read the problem and trace through the test.\n\n"
                f"PROBLEM: {instruction}\n\n"
                f"CODE:\n```\n{source}\n```\n\n"
                f"FAILED TEST:\n{failure_lines}\n\n"
                f"{pivot_hint}"
                "Step by step:\n"
                "1. Read the problem statement — what does it actually require?\n"
                "2. Trace through your code with the failing test input\n"
                "3. At which specific line does the logic diverge from what the problem requires?\n"
                "4. What is the exact root cause (wrong condition, wrong variable, wrong loop, etc.)?\n"
            )
            if attempt >= 2:
                trace_prompt += (
                    "5. What COMPLETELY DIFFERENT algorithm or approach would solve this correctly?\n\n"
                    "Output your analysis concisely — max 8 lines. Be precise about which line is wrong, "
                    "why the current APPROACH is fundamentally flawed, and what different strategy to use."
                )
            else:
                trace_prompt += (
                    "\nOutput your analysis concisely — max 5 lines. Be precise about which line is wrong and why."
                )
            reasoning = llm.invoke(trace_prompt).strip()
            logger.info("[TEST_RUNNER] Self-reasoning result: %s", reasoning[:300])
        except Exception as e:
            logger.warning("[TEST_RUNNER] Self-reasoning failed: %s", e)

    note = (
        "TEST FAILURES — the current code does not produce correct output.\n"
        f"Original requirement: {instruction}\n\n"
        "FAILED TESTS:\n"
        f"{failure_lines}\n"
    )

    if reasoning:
        note += (
            "ROOT CAUSE ANALYSIS (trace-through):\n"
            f"{reasoning}\n\n"
        )

    if attempt >= 2:
        note += (
            "CRITICAL: Previous attempts to fix this with small tweaks have FAILED. "
            "You MUST use a COMPLETELY DIFFERENT algorithm or approach. "
            "Rewrite the function logic from scratch using a new strategy. "
            "Output a SEARCH/REPLACE block."
        )
    else:
        note += (
            "Fix the specific root cause identified above. "
            "Do NOT restructure the entire function — target ONLY the incorrect logic. "
            "Output a SEARCH/REPLACE block."
        )
    return note
