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


def _generate_debug_harness(
    source: str,
    ext: str,
    failing_input: str,
    expected_output: str,
    llm: OllamaLLM,
) -> str:
    """Generate a test harness that prints intermediate variable values.

    Instead of just printing the final return value, this harness instruments
    the function so that key variables (loop counters, accumulators, conditions)
    are printed at each iteration.  The runtime output lets the LLM see the
    actual execution flow rather than having to guess.
    """
    lang = LANG_FENCE.get(ext, ext.lstrip("."))
    prompt = (
        f"Given this {lang} source code:\n"
        f"```{lang}\n{source}\n```\n\n"
        f"I need to debug why it returns the WRONG answer for input: {failing_input}\n"
        f"Expected output: {expected_output}\n\n"
        f"Write a complete, self-contained, runnable {lang} DEBUG program that:\n"
        f"1. Contains a MODIFIED version of the function(s) above with print/fmt.Println/cout "
        f"statements added INSIDE loops and at key decision points\n"
        f"2. Prints the values of ALL important variables at each loop iteration "
        f"(loop index, accumulators, counters, conditions being checked)\n"
        f"3. Format each debug line as: STEP <iteration>: <var1>=<val1>, <var2>=<val2>, ...\n"
        f"4. At the end, prints RESULT: <final_return_value>\n"
        f"5. Has a main entry point that calls the function with input: {failing_input}\n\n"
        f"Language-specific rules:\n"
        f"- Go: use 'package main' and 'import \"fmt\"'. Use fmt.Printf for debug output.\n"
        f"- Python: use print() for debug output.\n"
        f"- C++: use std::cerr for debug output (so it doesn't mix with stdout), "
        f"then std::cout for RESULT line.\n\n"
        f"The goal is to see the ACTUAL runtime values at each step so we can find "
        f"where the logic diverges from what's expected.\n\n"
        f"Output ONLY the complete program — no explanation, no markdown fences."
    )
    harness = llm.invoke(prompt).strip()
    if "```" in harness:
        harness = strip_markdown(harness)
    return harness


def run_debug_trace(
    source: str,
    path: str,
    failing_input: str,
    expected_output: str,
    llm: Optional[OllamaLLM] = None,
) -> Optional[str]:
    """Run a debug-instrumented version of the code and capture intermediate values.

    Returns the debug output as a string, or None on failure.
    """
    ext = os.path.splitext(path)[1].lower()
    if llm is None:
        llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)

    try:
        harness = _generate_debug_harness(source, ext, failing_input, expected_output, llm)
    except Exception as e:
        logger.warning("[DEBUG_TRACE] Harness generation failed: %s", e)
        return None

    logger.debug("[DEBUG_TRACE] Generated debug harness:\n%s", harness[:1000])

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if ext == ".go":
                harness_path = os.path.join(tmpdir, "main.go")
                cmd = ["go", "run", harness_path]
            elif ext == ".py":
                harness_path = os.path.join(tmpdir, "debug_harness.py")
                cmd = [sys.executable, harness_path]
            elif ext == ".cpp":
                harness_path = os.path.join(tmpdir, "debug.cpp")
                exe_path = os.path.join(tmpdir, "debug.exe" if os.name == "nt" else "debug")
                compiler, _ = _find_cpp_compiler()
                if not compiler:
                    return None
                compile_cmd = [compiler, "-std=c++17", "-O0", "-o", exe_path, harness_path]
                cmd = None
            elif ext == ".c":
                harness_path = os.path.join(tmpdir, "debug.c")
                exe_path = os.path.join(tmpdir, "debug.exe" if os.name == "nt" else "debug")
                compiler, _ = _find_c_compiler()
                if not compiler:
                    return None
                compile_cmd = [compiler, "-std=c11", "-O0", "-o", exe_path, harness_path]
                cmd = None
            else:
                return None

            with open(harness_path, "w", encoding="utf-8") as f:
                f.write(harness)

            if ext in (".cpp", ".c"):
                compile_result = subprocess.run(
                    compile_cmd, capture_output=True, text=True, timeout=30, cwd=tmpdir
                )
                if compile_result.returncode != 0:
                    logger.warning("[DEBUG_TRACE] Compilation failed: %s", compile_result.stderr[:400])
                    return None
                cmd = [exe_path]

            run_result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10, cwd=tmpdir
            )
            # Combine stdout and stderr (C++ debug may use cerr)
            output = (run_result.stdout or "") + (run_result.stderr or "")
            output = output.strip()
            if output:
                # Truncate very long traces to keep prompt reasonable
                lines = output.split("\n")
                if len(lines) > 60:
                    output = "\n".join(lines[:30] + ["... (truncated) ..."] + lines[-10:])
                logger.info("[DEBUG_TRACE] Captured %d lines of debug output", len(lines))
                return output
            return None

    except subprocess.TimeoutExpired:
        logger.warning("[DEBUG_TRACE] Execution timed out")
        return None
    except Exception as e:
        logger.warning("[DEBUG_TRACE] Failed: %s", e)
        return None


def build_test_failure_note(
    test_results: List[Dict],
    instruction: str,
    source: str = "",
    llm: Optional[OllamaLLM] = None,
    attempt: int = 0,
    failed_approaches: Optional[List[str]] = None,
    debug_trace: Optional[str] = None,
    previous_diff: Optional[str] = None,
) -> str:
    """
    Build a structured prompt note from failing test cases.

    Includes:
      - Debug trace: actual runtime variable values from executing the code
        (when available), so the LLM sees real execution data instead of guessing
      - Self-reasoning: LLM reads problem + debug trace to identify root cause
      - Previous attempt diff: what changed last time (so LLM doesn't repeat it)
      - Graduated escalation: early attempts target specific bugs, later attempts
        suggest broader rethinking

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
                    "The current approach may be fundamentally flawed.\n"
                    "Consider whether a DIFFERENT algorithm or strategy is needed — "
                    "do NOT keep making small tweaks if the core logic is wrong.\n"
                )
                if failed_approaches:
                    pivot_hint += "APPROACHES ALREADY TRIED AND FAILED:\n"
                    for fa in failed_approaches:
                        pivot_hint += f"  - {fa}\n"
                    pivot_hint += "Do NOT use any of the above approaches again.\n"

            # Include debug trace data in the reasoning prompt
            debug_section = ""
            if debug_trace:
                debug_section = (
                    "\nACTUAL RUNTIME EXECUTION TRACE (from running the code):\n"
                    "```\n"
                    f"{debug_trace}\n"
                    "```\n"
                    "The above shows the REAL variable values at each step when the code runs.\n"
                    "Use this trace to find EXACTLY where the logic produces the wrong value.\n\n"
                )

            prev_diff_section = ""
            if previous_diff:
                prev_diff_section = (
                    "\nPREVIOUS FIX ATTEMPT (this change did NOT solve the problem):\n"
                    f"```diff\n{previous_diff}\n```\n"
                    "Do NOT repeat this same change. Try a different fix.\n\n"
                )

            trace_prompt = (
                f"You are debugging incorrect code. Read the problem, study the execution trace, "
                f"and identify the root cause.\n\n"
                f"PROBLEM: {instruction}\n\n"
                f"CODE:\n```\n{source}\n```\n\n"
                f"FAILED TEST:\n{failure_lines}\n\n"
                f"{debug_section}"
                f"{prev_diff_section}"
                f"{pivot_hint}"
                "Step by step:\n"
                "1. What does the problem actually require? State it precisely.\n"
                "2. Look at the execution trace — at which specific step do the variable values "
                "diverge from what the correct algorithm would produce?\n"
                "3. What SPECIFIC line or expression in the code causes this wrong value?\n"
                "4. What is the exact fix needed (be precise: wrong operator, wrong index, "
                "wrong condition, missing case, etc.)?\n"
            )
            if attempt >= 2:
                trace_prompt += (
                    "5. If the same type of error keeps recurring, what DIFFERENT algorithmic "
                    "approach would avoid this class of bugs entirely?\n\n"
                    "Output your analysis concisely — max 10 lines. Be precise about which "
                    "line is wrong, what the variable values should be vs what they are, "
                    "and what specific change fixes it."
                )
            else:
                trace_prompt += (
                    "\nOutput your analysis concisely — max 8 lines. Reference the execution "
                    "trace values to pinpoint the exact line and expression that's wrong."
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

    if debug_trace:
        note += (
            "RUNTIME EXECUTION TRACE (actual variable values when code runs):\n"
            f"```\n{debug_trace}\n```\n\n"
        )

    if previous_diff:
        note += (
            "PREVIOUS FIX THAT DID NOT WORK (do NOT repeat this):\n"
            f"```diff\n{previous_diff}\n```\n\n"
        )

    if reasoning:
        note += (
            "ROOT CAUSE ANALYSIS (based on execution trace):\n"
            f"{reasoning}\n\n"
        )

    if attempt >= 2:
        note += (
            "Previous fix attempts have not resolved the issue. "
            "If the root cause analysis points to a specific bug, fix that precisely. "
            "If the same kind of error keeps repeating, consider restructuring the "
            "core logic of the function. "
            "Output a SEARCH/REPLACE block."
        )
    else:
        note += (
            "Fix the specific root cause identified above. "
            "Use the execution trace values to verify your fix is correct. "
            "Do NOT restructure the entire function — target ONLY the incorrect logic. "
            "Output a SEARCH/REPLACE block."
        )
    return note
