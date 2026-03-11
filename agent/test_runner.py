import os
import re
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


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & COMPILER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

_COMPILER_CANDIDATES: Dict[str, List[Tuple[str, str]]] = {
    "cpp": [("g++", "g++"), ("clang++", "clang++"), ("cl", "MSVC")],
    "c":   [("gcc", "gcc"), ("clang", "clang"), ("cl", "MSVC")],
}

_COMPILER_STD: Dict[str, str] = {
    ".cpp": "-std=c++17",
    ".c":   "-std=c11",
}

_EXT_HARNESS_NAME: Dict[str, str] = {
    ".go":  "main.go",
    ".py":  "harness.py",
    ".js":  "harness.js",
    ".cpp": "harness.cpp",
    ".c":   "harness.c",
}


def _find_compiler(lang: str) -> Tuple[Optional[str], str]:
    """Find an available compiler for *lang* ('cpp' or 'c').

    Returns: (compiler_path, compiler_name) or (None, error_message)
    """
    for binary, name in _COMPILER_CANDIDATES.get(lang, []):
        if shutil.which(binary):
            return binary, name
    friendly = "C++" if lang == "cpp" else "C"
    bins = ", ".join(b for b, _ in _COMPILER_CANDIDATES.get(lang, []))
    return None, f"No {friendly} compiler found. Looked for: {bins}"


def _compile_and_run(
    harness_code: str,
    ext: str,
    *,
    opt_level: str = "-O0",
    run_timeout: int = 15,
    asan: bool = False,
    label: str = "HARNESS",
) -> Tuple[Optional[str], Optional[str]]:
    """Write *harness_code* to a temp file, compile (if needed), and run.

    Returns ``(stdout_and_stderr_combined, None)`` on success,
    or ``(None, error_message)`` on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = _EXT_HARNESS_NAME.get(ext)
        if fname is None:
            return None, f"Unsupported extension: {ext}"
        harness_path = os.path.join(tmpdir, fname)

        with open(harness_path, "w", encoding="utf-8") as f:
            f.write(harness_code)

        # Interpreted languages -------------------------------------------------
        if ext == ".go":
            cmd = ["go", "run", harness_path]
        elif ext == ".py":
            cmd = [sys.executable, harness_path]
        elif ext == ".js":
            cmd = ["node", harness_path]
        elif ext in (".cpp", ".c"):
            lang_key = "cpp" if ext == ".cpp" else "c"
            compiler, info = _find_compiler(lang_key)
            if not compiler:
                return None, info
            exe = os.path.join(tmpdir, "a.exe" if os.name == "nt" else "a.out")
            flags = [compiler, _COMPILER_STD[ext], opt_level, "-g"]
            if asan:
                flags += ["-fsanitize=address", "-fno-omit-frame-pointer"]
            flags += ["-o", exe, harness_path]
            comp = subprocess.run(flags, capture_output=True, text=True,
                                  timeout=30, cwd=tmpdir)
            if comp.returncode != 0:
                return None, (comp.stderr or "Compilation failed").strip()[:800]
            cmd = [exe]
        else:
            return None, f"Unsupported extension: {ext}"

        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=run_timeout, cwd=tmpdir)
        combined = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.returncode != 0:
            return None, combined[:2000] or "Non-zero exit code"
        return combined, None


# ═══════════════════════════════════════════════════════════════════════════════
# HARNESS GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


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


def _make_error_list(test_cases: List[Dict[str, str]], err: str) -> List[Dict]:
    """Return a uniform failure result for every test case."""
    return [
        {"input": tc["input"], "expected": tc["expected"],
         "actual": None, "passed": False, "error": err}
        for tc in test_cases
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════


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
    from .code_editing_helpers import source_matches_ext

    ext = os.path.splitext(path)[1].lower()

    # Language safety check: if the source code doesn't match the file extension
    # (e.g. Python code in a .go file due to LLM language confusion), report
    # the mismatch instead of generating a broken harness.
    if not source_matches_ext(source, ext):
        err = (
            f"Language mismatch: file is {ext} but source appears to be a different language. "
            "The LLM likely generated code in the wrong language."
        )
        logger.error("[TEST_RUNNER] %s", err)
        return _make_error_list(test_cases, err)

    if llm is None:
        llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)

    inputs = [tc["input"] for tc in test_cases]
    expected_outputs = [tc["expected"] for tc in test_cases]
    logger.info("[TEST_RUNNER] Generating harness for %d test case(s) [%s]...", len(test_cases), ext)

    try:
        harness = _generate_harness(source, ext, inputs, expected_outputs, llm)
    except Exception as e:
        logger.error("[TEST_RUNNER] Harness generation failed: %s", e)
        return _make_error_list(test_cases, f"Harness generation failed: {e}")

    logger.debug("[TEST_RUNNER] Generated harness:\n%s", harness[:800])

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Resolve file name, compiler, and run command by extension
            fname = _EXT_HARNESS_NAME.get(ext)
            if not fname:
                return _make_error_list(test_cases, f"Unsupported language for test runner: '{ext}'")
            harness_path = os.path.join(tmpdir, fname)
            exe_path = os.path.join(tmpdir, "harness.exe" if os.name == "nt" else "harness")
            compiler = None

            if ext == ".go":
                cmd = ["go", "run", harness_path]
            elif ext == ".py":
                cmd = [sys.executable, harness_path]
            elif ext == ".js":
                cmd = ["node", harness_path]
            elif ext in (".cpp", ".c"):
                lang_key = "cpp" if ext == ".cpp" else "c"
                compiler, compiler_info = _find_compiler(lang_key)
                if not compiler:
                    logger.error("[TEST_RUNNER] %s", compiler_info)
                    return _make_error_list(test_cases, compiler_info)
                compile_cmd = [compiler, _COMPILER_STD[ext], "-O2", "-o", exe_path, harness_path]
                cmd = None
            else:
                return _make_error_list(test_cases, f"Unsupported language for test runner: '{ext}'")

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
                    return _make_error_list(test_cases, f"Compilation error:\n{err_msg}")
                cmd = [exe_path]
                logger.info("[TEST_RUNNER] Compilation successful")

            logger.info("[TEST_RUNNER] Running: %s", " ".join(cmd))
            run_result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=20, cwd=tmpdir
            )

            if run_result.returncode != 0:
                err_msg = (run_result.stderr or "Unknown error").strip()[:800]
                logger.error("[TEST_RUNNER] Harness execution failed:\n%s", err_msg)

                # ASan rerun for C/C++ crashes — recompile with AddressSanitizer
                asan_detail = ""
                if ext in (".cpp", ".c") and compiler and shutil.which(compiler):
                    try:
                        asan_output, _ = _compile_and_run(
                            harness, ext, asan=True, run_timeout=10, label="ASan"
                        )
                        if asan_output:
                            asan_detail = f"\n\nAddressSanitizer report:\n{asan_output[:2000]}"
                            logger.info("[TEST_RUNNER] ASan report captured (%d chars)", len(asan_output))
                    except Exception as e:
                        logger.debug("[TEST_RUNNER] ASan rerun failed: %s", e)

                return _make_error_list(
                    test_cases, f"Compilation/runtime error:\n{err_msg}{asan_detail}"
                )

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
        return _make_error_list(test_cases, "Execution timed out (>20 s) — possible infinite loop")
    except Exception as e:
        logger.error("[TEST_RUNNER] Unexpected error: %s", e)
        return _make_error_list(test_cases, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# DEBUG TRACING
# ═══════════════════════════════════════════════════════════════════════════════


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
        output, err = _compile_and_run(harness, ext, run_timeout=10, label="DEBUG_TRACE")
    except subprocess.TimeoutExpired:
        logger.warning("[DEBUG_TRACE] Execution timed out")
        return None
    except Exception as e:
        logger.warning("[DEBUG_TRACE] Failed: %s", e)
        return None

    if err:
        logger.warning("[DEBUG_TRACE] %s", err[:400])
        return None
    if not output:
        return None

    lines = output.split("\n")
    if len(lines) > 60:
        output = "\n".join(lines[:30] + ["... (truncated) ..."] + lines[-10:])
    logger.info("[DEBUG_TRACE] Captured %d lines of debug output", len(lines))
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE REPORTING
# ═══════════════════════════════════════════════════════════════════════════════


def build_test_failure_note(
    test_results: List[Dict],
    instruction: str,
    source: str = "",
    llm: Optional[OllamaLLM] = None,
    attempt: int = 0,
    failed_approaches: Optional[List[str]] = None,
    debug_trace: Optional[str] = None,
    previous_diff: Optional[str] = None,
    step_verification: Optional[str] = None,
) -> str:
    """
    Build a structured prompt note from failing test cases.

    Includes:
      - Step verification: concrete assertion results showing which logical step fails
      - Debug trace: actual runtime variable values from executing the code
        (when available), so the LLM sees real execution data instead of guessing
      - Self-reasoning: LLM reads problem + debug trace + step verification to find root cause
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

    # Detect if all failures are crashes (no actual output) — different debugging approach needed
    all_crashes = all(r.get("actual") is None for r in failures)
    has_asan = any("AddressSanitizer" in str(r.get("error", "")) or "sanitizer" in str(r.get("error", "")).lower() for r in failures)
    crash_hint = ""
    if all_crashes:
        crash_hint = (
            "\nCRITICAL: The code CRASHES at runtime — it does not produce any output.\n"
            "This is a MEMORY ACCESS bug, not a logic error. Focus on:\n"
            "- Array/vector/string index expressions that go OUT OF BOUNDS\n"
            "- If an array has length N, index N+i is INVALID — use (N+i) % N or (N+i-1) % N\n"
            "- 'Stack overflow' on LeetCode = out-of-bounds memory access, NOT infinite recursion\n"
            "- Check the MAXIMUM value each index expression can reach across ALL loop iterations\n"
        )

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

            # Include step verification — the most precise diagnostic
            verify_section = ""
            if step_verification:
                verify_section = (
                    "\nSTEP-BY-STEP ASSERTION RESULTS (from actually running the code):\n"
                    f"{step_verification}\n\n"
                    "The above was EXECUTED — these are real values, not guesses.\n"
                    "Focus your fix on the FIRST FAILING STEP. That is where the bug is.\n\n"
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
                f"{crash_hint}"
                f"{verify_section}"
                f"{debug_section}"
                f"{prev_diff_section}"
                f"{pivot_hint}"
                "Step by step:\n"
                "1. What does the problem actually require? State it precisely.\n"
            )
            if all_crashes:
                trace_prompt += (
                    "2. List EVERY array/vector/string index expression in the code.\n"
                    "3. For each index expression, compute its MAXIMUM possible value across all loop iterations.\n"
                    "4. Compare each maximum index to the array's length — which one exceeds the bounds?\n"
                    "5. What is the exact fix (e.g., wrap with % N, change loop bound, resize array)?\n"
                )
            else:
                trace_prompt += (
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
    )

    if all_crashes:
        note += (
            "RUNTIME CRASH — the code crashes before producing any output.\n"
            "This is a memory access bug (out-of-bounds array index), NOT a logic error.\n"
            "Do NOT rewrite the algorithm — find and fix the specific index expression "
            "that goes out of bounds.\n\n"
        )

    note += (
        "FAILED TESTS:\n"
        f"{failure_lines}\n"
    )

    if debug_trace:
        note += (
            "RUNTIME EXECUTION TRACE (actual variable values when code runs):\n"
            f"```\n{debug_trace}\n```\n\n"
        )

    if step_verification:
        note += f"\n{step_verification}\n\n"

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


# ═══════════════════════════════════════════════════════════════════════════════
# STEP VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# Unlike debug_trace (which prints values for a human to read), this generates
# executable assertions that HALT with a clear error message when a specific
# logical step produces the wrong intermediate value.
#
# Flow:
#   1. LLM decomposes the algorithm into numbered logical steps
#   2. For each step, LLM generates an assertion with the expected value
#   3. The assertion harness is compiled and run
#   4. On failure: returns "STEP 3 FAILED: expected flips1=2, got flips1=5"
#   5. On success: returns "ALL STEPS VERIFIED" (the function is correct)

def _generate_assertion_harness(
    source: str,
    ext: str,
    failing_input: str,
    expected_output: str,
    instruction: str,
    llm: OllamaLLM,
) -> str:
    """Generate a test harness with step-by-step assertions for intermediate values."""
    lang = LANG_FENCE.get(ext, ext.lstrip("."))
    prompt = (
        f"You are writing a STEP-BY-STEP VERIFICATION PROGRAM for this {lang} code.\n\n"
        f"PROBLEM: {instruction}\n\n"
        f"CODE UNDER TEST:\n```{lang}\n{source}\n```\n\n"
        f"FAILING TEST:\n  Input: {failing_input}\n  Expected output: {expected_output}\n\n"
        f"Write a complete, self-contained, runnable {lang} program that:\n\n"
        f"1. COPIES the function(s) from above EXACTLY — do NOT fix or change the logic\n"
        f"2. DECOMPOSES the algorithm into logical steps (initialization, each loop phase, etc.)\n"
        f"3. After each logical step, prints a verification line in this EXACT format:\n"
        f"   STEP <N> (<description>): <var1>=<val1>, <var2>=<val2> | EXPECTED: <what_correct_value_should_be>\n"
        f"4. After each step, prints PASS or FAIL:\n"
        f"   STEP <N>: PASS    (if values match what a correct algorithm would produce)\n"
        f"   STEP <N>: FAIL    (if values diverge from correct — this is where the bug is)\n"
        f"5. After ALL steps, prints: RESULT: <actual_return_value>\n"
        f"6. Has a main entry point that calls the function with input: {failing_input}\n\n"
        f"CRITICAL RULES:\n"
        f"- You must MANUALLY compute what the correct intermediate values should be for this specific input\n"
        f"- Do NOT just check if the function 'runs' — check if EACH STEP produces the RIGHT VALUE\n"
        f"- Include at least 3-5 verification steps covering: initialization, key loop iterations, final result\n"
        f"- For loops that run N iterations, check at least iteration 0, a middle iteration, and the last iteration\n"
        f"- The function logic must be IDENTICAL to the original — do NOT 'fix' it\n\n"
        f"Language-specific rules:\n"
        f"- Go: use 'package main' and fmt.Printf for output\n"
        f"- Python: use print() for output\n"
        f"- C++: use std::cout for ALL output. Include necessary headers.\n"
        f"- C: use printf() for output\n\n"
        f"Output ONLY the complete program — no explanation, no markdown fences."
    )
    harness = llm.invoke(prompt).strip()
    if "```" in harness:
        harness = strip_markdown(harness)
    return harness


def run_step_verification(
    source: str,
    path: str,
    failing_input: str,
    expected_output: str,
    instruction: str,
    llm: Optional[OllamaLLM] = None,
) -> Optional[Dict]:
    """Run step-by-step assertion verification and return structured results.

    Returns:
        {
            "steps": [
                {"step": 1, "description": "...", "values": "...", "expected": "...", "passed": True/False},
                ...
            ],
            "first_failure": {"step": N, "description": "...", "values": "...", "expected": "..."},
            "all_passed": True/False,
            "raw_output": "...",
            "actual_result": "...",
        }
        or None on infrastructure failure (compilation error, timeout, etc.)
    """
    ext = os.path.splitext(path)[1].lower()
    if llm is None:
        llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)

    try:
        harness = _generate_assertion_harness(source, ext, failing_input, expected_output, instruction, llm)
    except Exception as e:
        logger.warning("[STEP_VERIFY] Harness generation failed: %s", e)
        return None

    logger.debug("[STEP_VERIFY] Generated assertion harness:\n%s", harness[:1200])

    try:
        output, err = _compile_and_run(harness, ext, run_timeout=15, label="STEP_VERIFY")
    except subprocess.TimeoutExpired:
        logger.warning("[STEP_VERIFY] Execution timed out")
        return None
    except Exception as e:
        logger.warning("[STEP_VERIFY] Failed: %s", e)
        return None

    if err:
        logger.warning("[STEP_VERIFY] %s", err[:500])
        return None
    if not output:
        return None

    lines = output.split("\n")
    if len(lines) > 80:
        output = "\n".join(lines[:40] + ["... (truncated) ..."] + lines[-15:])

    return _parse_step_verification(output)


def _parse_step_verification(raw_output: str) -> Dict:
    """Parse the step verification harness output into structured data."""
    steps = []
    first_failure = None
    actual_result = None

    for line in raw_output.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Parse RESULT line
        if line.upper().startswith("RESULT:"):
            actual_result = line[len("RESULT:"):].strip()
            continue

        # Parse STEP lines: "STEP 3 (description): var=val | EXPECTED: ..."
        step_match = re.match(
            r'STEP\s+(\d+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)',
            line, re.IGNORECASE
        )
        if step_match:
            step_num = int(step_match.group(1))
            description = step_match.group(2) or ""
            rest = step_match.group(3).strip()

            # Check for PASS/FAIL verdict
            if rest.upper() in ("PASS", "FAIL"):
                # This is a verdict line, update the previous step
                for s in reversed(steps):
                    if s["step"] == step_num:
                        s["passed"] = rest.upper() == "PASS"
                        if not s["passed"] and first_failure is None:
                            first_failure = s.copy()
                        break
                continue

            # This is a data line with values
            values = rest
            expected = ""
            if "| EXPECTED:" in rest.upper():
                parts = re.split(r'\|\s*EXPECTED:\s*', rest, flags=re.IGNORECASE)
                values = parts[0].strip()
                expected = parts[1].strip() if len(parts) > 1 else ""

            steps.append({
                "step": step_num,
                "description": description,
                "values": values,
                "expected": expected,
                "passed": None,  # Will be updated by PASS/FAIL line
            })

    # If no explicit PASS/FAIL lines, infer from last step
    for s in steps:
        if s["passed"] is None:
            s["passed"] = True  # Assume pass if no verdict (harness didn't crash before it)

    # Try to detect first failure from crash (steps that never got a verdict)
    if first_failure is None:
        for s in steps:
            if not s["passed"]:
                first_failure = s.copy()
                break

    return {
        "steps": steps,
        "first_failure": first_failure,
        "all_passed": all(s.get("passed", True) for s in steps) if steps else False,
        "raw_output": raw_output,
        "actual_result": actual_result,
    }


def format_step_verification_for_prompt(verification: Dict) -> str:
    """Format step verification results into a prompt-friendly string."""
    if not verification or not verification.get("steps"):
        return ""

    parts = ["STEP-BY-STEP EXECUTION VERIFICATION (ran against your current code):"]

    for s in verification["steps"]:
        status = "PASS ✓" if s.get("passed", True) else "**FAIL ✗**"
        desc = f" ({s['description']})" if s.get('description') else ""
        line = f"  Step {s['step']}{desc}: {s['values']}"
        if s.get("expected"):
            line += f"  | Expected: {s['expected']}"
        line += f"  → {status}"
        parts.append(line)

    if verification.get("actual_result"):
        parts.append(f"  Final result: {verification['actual_result']}")

    if verification.get("first_failure"):
        fail = verification["first_failure"]
        desc = f" ({fail['description']})" if fail.get('description') else ""
        parts.append(f"\n>>> BUG LOCATION: Step {fail['step']}{desc}")
        parts.append(f"    Got: {fail['values']}")
        if fail.get("expected"):
            parts.append(f"    Expected: {fail['expected']}")
        parts.append(
            "    Fix the code so this specific step produces the expected value. "
            "Do NOT rewrite the entire algorithm — target ONLY the expression(s) "
            "that produce the wrong value at this step."
        )
    elif verification.get("all_passed"):
        parts.append(
            "\nAll steps verified correctly — the intermediate values are right. "
            "The bug may be in how the final result is assembled or returned."
        )

    return "\n".join(parts)
