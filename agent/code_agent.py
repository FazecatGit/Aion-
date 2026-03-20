import subprocess
import re
import logging
import sys
import traceback
from datetime import datetime
from types import SimpleNamespace
import os

from typing import List, Dict, Optional

from brain.config import LLM_MODEL, LANG_NAMES, LANG_FENCE, IMPLEMENT_KEYWORDS, make_llm
from brain.fast_search import fast_topic_search
from db.db_reader import get_code_documents
from .tools import read_file, write_file, run_git_command, show_diff, run_python_file, run_shell_command, list_files
from .code_editing_helpers import (
    collect_syntax_errors, parse_compiler_errors, get_focused_context,
    detect_indent_char, extract_function, apply_block_to_lines,
    is_oversized_block, find_first_mismatch, whole_function_replace,
    strip_markdown, parse_multiple_blocks, parse_blocks_with_retry,
    build_syntax_error_note, build_runtime_error_note, build_post_error_note,
    validate_code_structure, build_structural_issue_note,
    count_top_level_functions, extract_error_lines_from_text,
    source_matches_ext,
    FUNC_PATTERN, _C_FUNC_PATTERN
)
from .code_context_builders import build_all_contexts, run_lint_checks
from .session_memory import SessionMemory
from .orchestration import plan_task, format_plan_for_prompt, critique_code, build_critic_feedback_note
from print_logger import get_logger

logger = get_logger("code_agent")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# Shared session memory instance (persists across requests within a server lifetime)
_session_memory: Optional[SessionMemory] = None

def get_session_memory() -> SessionMemory:
    """Lazy-init singleton session memory."""
    global _session_memory
    if _session_memory is None:
        _session_memory = SessionMemory()
    return _session_memory


def _apply_repair_blocks(
    blocks: list[tuple[str, str]],
    source_lines: list[str],
    prefix: str = "Block",
) -> list[str]:
    """Apply a list of SEARCH/REPLACE blocks to *source_lines*, skipping oversized ones.

    Returns the (possibly mutated) source_lines list.
    Each block is validated after application: if it causes top-level
    function definitions to disappear the block is rolled back.
    """
    for idx, (search_text, replace_text) in enumerate(blocks):
        if is_oversized_block(search_text, source_lines, idx + 1, prefix=prefix):
            continue
        match_idx, replace_lines, matched_len = apply_block_to_lines(
            source_lines, search_text, replace_text
        )
        if match_idx == -1:
            logger.warning("%s %s: Unable to locate SEARCH block. Skipping.", prefix, idx + 1)
            continue
        logger.debug("Applying %s %s at line %s replacing %s lines", prefix, idx + 1, match_idx, matched_len)
        prev_lines = source_lines[:]
        source_lines = (
            source_lines[:match_idx]
            + replace_lines
            + source_lines[match_idx + matched_len:]
        )
        # Safety: reject block if it deleted function definitions
        prev_count = count_top_level_functions(prev_lines)
        new_count = count_top_level_functions(source_lines)
        if prev_count > 1 and new_count < prev_count:
            logger.warning(
                "%s %s: REJECTED — applying this block reduced functions from %d to %d. Rolling back.",
                prefix, idx + 1, prev_count, new_count,
            )
            source_lines = prev_lines
    return source_lines


# ═══════════════════════════════════════════════════════════════════════════════
# CODE AGENT
# ═══════════════════════════════════════════════════════════════════════════════


class CodeAgent:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.edit_log: dict[str, list] = {}
        self.memory = get_session_memory()

    @staticmethod
    def _is_implement_task(instruction: str) -> bool:
        """Return True when the instruction is asking to build/implement something new
        rather than fix/edit existing code."""
        instruction_lower = instruction.lower()
        return any(kw in instruction_lower for kw in IMPLEMENT_KEYWORDS)

    @staticmethod
    def _classify_problem_pattern(instruction: str, source: str, ext: str, llm) -> str:
        """Pre-pass: classify the algorithmic pattern before code generation.

        Returns a short classification block (pattern name, key invariants,
        edge cases) that gets injected into the analysis/edit prompts so the
        LLM reasons from a correct framework instead of guessing.
        """
        lang_name = LANG_NAMES.get(ext, ext.lstrip('.').upper() if ext else 'code')
        prompt = (
            "You are an algorithm classification expert. Given the problem below "
            "and the current code, identify the CORE algorithmic pattern needed.\n\n"
            f"PROBLEM: {instruction[:1500]}\n\n"
            f"CURRENT CODE ({lang_name}):\n```\n{source[:2000]}\n```\n\n"
            "Respond with EXACTLY this format (no other text):\n"
            "PATTERN: <pattern name, e.g. two-pointer, binary search, BFS, DP, "
            "sliding window, greedy, backtracking, divide-and-conquer, union-find, "
            "trie, monotonic stack, topological sort, etc.>\n"
            "INVARIANT: <the key invariant that must hold at every step>\n"
            "EDGE CASES: <comma-separated list of edge cases to handle>\n"
            "COMPLEXITY: <expected time and space complexity>\n"
        )
        try:
            classification = llm.invoke(prompt).strip()
            # Validate it has the expected structure
            if "PATTERN:" in classification:
                logger.info("[CLASSIFY] %s", classification.replace('\n', ' | '))
                return classification
            logger.debug("[CLASSIFY] LLM returned unstructured output, discarding")
            return ""
        except Exception as e:
            logger.debug("[CLASSIFY] Classification failed: %s", e)
            return ""

    @staticmethod
    def _resolve_rerank_method(instruction: str, rerank_method: str) -> str:
        """Auto-select rerank method based on instruction complexity.
        
        fast  → keyword reranking  (no ML model, instant)
        deep  → cross_encoder       (sentence-transformers, ~2s overhead, best quality)
        auto  → chooses based on instruction complexity
        """
        if rerank_method != "auto":
            return rerank_method
        # Heuristic: long/complex instructions warrant cross-encoder; quick/simple ones use keyword
        COMPLEX_KEYWORDS = {
            "refactor", "fix", "bug", "error", "crash", "broken", "wrong", "fail",
            "implement", "add feature", "redesign", "rewrite", "optimize", "performance",
            "async", "concurrent", "race condition", "exception", "traceback"
        }
        instruction_lower = instruction.lower()
        word_count = len(instruction_lower.split())
        is_complex = (
            word_count > 12 or
            any(kw in instruction_lower for kw in COMPLEX_KEYWORDS)
        )
        chosen = "cross_encoder" if is_complex else "keyword"
        logger.info("[AGENT] rerank auto-selected: %s (complex=%s)", chosen, is_complex)
        return chosen

    @staticmethod
    def _strip_file_level_declarations(code: str, ext: str) -> str:
        """Strip package/import/include declarations that the LLM should not generate.

        The whole-function rewrite prompt asks for function code only, but 7B models
        often prepend ``package main`` or ``import (...)`` blocks. This strips them
        so the output can be spliced into the existing file safely.
        """
        if ext not in ('.go', '.py', '.c', '.cpp', '.h', '.hpp', '.rs', '.java'):
            return code
        lines = code.split('\n')
        cleaned: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Go: strip 'package ...' declarations
            if ext == '.go' and re.match(r'^package\s+\w+', line):
                i += 1
                # skip blank lines after package
                while i < len(lines) and not lines[i].strip():
                    i += 1
                continue
            # Go: strip 'import (...)' or 'import "..."' blocks
            if ext == '.go' and line.startswith('import'):
                if '(' in line:
                    # multi-line import block
                    i += 1
                    while i < len(lines) and ')' not in lines[i]:
                        i += 1
                    i += 1  # skip closing ')'
                    # skip blank lines after import
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    continue
                else:
                    i += 1  # single-line import
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    continue
            # Python: strip 'import ...' or 'from ... import ...' at top level
            if ext == '.py' and re.match(r'^(import |from \S+ import )', line):
                i += 1
                continue
            cleaned.append(lines[i])
            i += 1
        # Don't return empty string
        result = '\n'.join(cleaned).strip()
        return result if result else code
        # Heuristic: long/complex instructions warrant cross-encoder; quick/simple ones use keyword
        COMPLEX_KEYWORDS = {
            "refactor", "fix", "bug", "error", "crash", "broken", "wrong", "fail",
            "implement", "add feature", "redesign", "rewrite", "optimize", "performance",
            "async", "concurrent", "race condition", "exception", "traceback"
        }
        instruction_lower = instruction.lower()
        word_count = len(instruction_lower.split())
        is_complex = (
            word_count > 12 or
            any(kw in instruction_lower for kw in COMPLEX_KEYWORDS)
        )
        chosen = "cross_encoder" if is_complex else "keyword"
        logger.info("[AGENT] rerank auto-selected: %s (complex=%s)", chosen, is_complex)
        return chosen

    @staticmethod
    def _clarify_instruction(instruction: str, source: str, ext: str) -> str:
        """Normalize casual/vague user instructions into precise technical requirements.

        Many users describe tasks in natural, imprecise language:
          - "make it faster" → "Optimize the time complexity of the existing algorithm"
          - "it's broken when empty" → "Fix the function to handle empty input without crashing"

        This step runs a lightweight LLM pass to extract a clean, unambiguous instruction
        while preserving the user's original intent.  Skipped for instructions that are
        already clear programming task descriptions.
        """
        # Heuristic: skip clarification for instructions that are already precise
        # (e.g. LeetCode problem statements, instructions with code keywords)
        PRECISE_SIGNALS = {
            'search/replace', 'implement', 'function that', 'return the', 'return minimum',
            'return maximum', 'given an array', 'given a string', 'given a linked list',
            'given n', 'two pointers', 'dynamic programming', 'binary search',
            'time complexity', 'space complexity',
        }
        lower = instruction.lower()
        if any(sig in lower for sig in PRECISE_SIGNALS):
            return instruction
        # Also skip if the instruction is very short (1-3 words) — not enough to clarify
        if len(instruction.split()) <= 3:
            return instruction
        # Also skip if the instruction is already long and detailed (>60 words)
        if len(instruction.split()) > 60:
            return instruction

        # Detect vague/casual language that needs clarification
        VAGUE_SIGNALS = {
            'make it', 'broken', "doesn't work", "don't work", 'not working',
            'something wrong', 'messed up', 'weird', 'acting up', 'help me',
            'can you', 'could you', 'fix this', 'fix it', 'make this', 'change this',
            'do something', 'it should', 'supposed to', 'need it to', 'want it to',
            'the thing', 'that part', 'this part', 'stuff',
        }
        needs_clarification = any(sig in lower for sig in VAGUE_SIGNALS)

        if not needs_clarification:
            return instruction

        try:
            llm = make_llm(temperature=0.0)
            lang_name = LANG_FENCE.get(ext, ext.lstrip('.'))
            prompt = (
                f"A user gave this instruction about their {lang_name} code:\n"
                f"USER: \"{instruction}\"\n\n"
                f"CODE:\n```{lang_name}\n{source[:2000]}\n```\n\n"
                "Rewrite the user's instruction as a clear, precise technical requirement.\n"
                "Rules:\n"
                "1. Keep the user's EXACT intent — do not add requirements they didn't ask for.\n"
                "2. Reference specific function/variable names from the code when possible.\n"
                "3. Be concise — one to three sentences maximum.\n"
                "4. Output ONLY the rewritten instruction — no explanation, no quotes.\n"
            )
            clarified = llm.invoke(prompt).strip()
            # Sanity check: reject if LLM produced something much longer or nonsensical
            if clarified and len(clarified) < len(instruction) * 5 and len(clarified) > 5:
                logger.info("[AGENT] Instruction clarified: '%s' → '%s'", instruction[:80], clarified[:120])
                return clarified
        except Exception as e:
            logger.debug("[AGENT] Instruction clarification failed: %s", e)

        return instruction

    def edit_code(
        self,
        path: str,
        instruction: str,
        dry_run: bool = True,
        use_rag: bool = False,
        session_chat_history: Optional[List[Dict[str, str]]] = None,
        rerank_method: str = "cross_encoder",
        max_chunks: Optional[int] = None,
        task_mode: str = "auto",
        search_method: str = "both",
        session_id: Optional[str] = None,
    ) -> str:
        """Edit code in a file based on instruction using LLM.
        
        Args:
            path: File path to edit
            instruction: Editing instruction
            dry_run: If True, only show diff without writing
            use_rag: If True, include RAG context from code search
            session_chat_history: Optional conversation history for context
            rerank_method: Reranking method for RAG - "cross_encoder", "keyword", or "none"
            task_mode: "fix" forces fix mode, "solve" forces implement mode, "auto" detects from instruction
            search_method: "bm25", "semantic", or "both" (default) for RAG search method
        """
        file_source = read_file(path)
        ext = os.path.splitext(path)[1].lower()
        is_python = ext == ".py"
        lang_fence = LANG_FENCE.get(ext, ext.lstrip('.') if ext else 'text')

        lang_name = LANG_NAMES.get(ext, ext.lstrip('.').upper() if ext else 'the same language as the file')
        lang_directive = f"LANGUAGE REQUIREMENT: You MUST write {lang_name} code. The file is a {lang_name} file ({ext}). Do NOT use any other programming language.\n\n"

        # Resolve "auto" rerank method based on instruction complexity
        rerank_method = self._resolve_rerank_method(instruction, rerank_method)

        file_lines = file_source.split('\n')
        # For small files (< 50 lines), allow larger SEARCH/REPLACE blocks since
        # a single-function file (e.g. LeetCode) requires replacing most of itself.
        MAX_BLOCK_RATIO = 0.95 if len(file_lines) < 50 else 0.6

        # --- Instruction clarification: normalize casual/vague language ---
        instruction = self._clarify_instruction(instruction, file_source, ext)

        # Compute multi_func_hint once, always defined
        # Triggers on: multiple `word()` patterns, multiple quoted instructions,
        # numbered steps, or multiple "X function" / "X method" mentions
        _func_call_matches = re.findall(r"\b\w+\(\)", instruction)
        _quoted_instructions = re.findall(r'"[^"]{10,}"', instruction)
        _numbered_steps = re.findall(r'(?:^|\n)\s*\d+[\.\)]\s+\S', instruction)
        _func_name_mentions = re.findall(r'\b(?:the\s+)?(\w+)\s+(?:function|method)\b', instruction, re.IGNORECASE)
        multi_func_hint = (
            len(_func_call_matches) >= 2
            or len(_quoted_instructions) >= 2
            or len(_numbered_steps) >= 2
            or len(set(_func_name_mentions)) >= 2
        )

        # --- Build syntax/runtime error notes ---
        syntax_errors = collect_syntax_errors(file_source, path) if is_python else []
        syntax_error_note = build_syntax_error_note(syntax_errors, file_lines) if syntax_errors else ""
        if syntax_errors:
            logger.info("Found %s syntax error(s) in %s", len(syntax_errors), path)

        # Build all context (runtime errors, RAG, history, edit log) in one call
        runtime_error_note, rag_context, history_context, edit_history_text, rag_citations = build_all_contexts(
            path=path,
            ext=ext,
            is_python=is_python,
            file_lines=file_lines,
            instruction=instruction,
            use_rag=use_rag,
            session_chat_history=session_chat_history,
            edit_log=self.edit_log,
            rerank_method=rerank_method,
            file_source=file_source,
            max_chunks=max_chunks,
            search_method=search_method
        )

        # Only build runtime errors if no syntax errors were found
        if syntax_errors:
            runtime_error_note = ""

        # --- Session memory context ---
        memory_context = ""
        if session_id:
            try:
                memory_context = self.memory.build_context_block(session_id, instruction, k=4)
                if memory_context:
                    logger.info("[AGENT] Session memory: injecting %d chars of context", len(memory_context))
            except Exception as e:
                logger.debug("[AGENT] Session memory recall failed: %s", e)

        combined_context = history_context + rag_context + edit_history_text + memory_context

        if multi_func_hint and runtime_error_note:
            runtime_error_note = (
                "NOTE: The file has a compiler error — it is ONE of the bugs listed in the instruction above. "
                "Fix ALL bugs in the instruction, including this one.\n"
            )

        extra_syntax = ""
        if syntax_error_note:
            extra_syntax += "\n\n" + syntax_error_note
        if runtime_error_note:
            extra_syntax += "\n\n" + runtime_error_note

        # --- Lint checks (deeper static analysis beyond compilation) ---
        lint_note = run_lint_checks(path, file_source, ext)
        if lint_note:
            extra_syntax += "\n\n" + lint_note
            logger.info("[AGENT] Lint issues found for %s", path)

        # extracted repeated `combined_context + extra_syntax` into one variable
        context_block = combined_context + extra_syntax

        # --- Cap context to avoid bloated prompts that stall the LLM ---
        _MAX_CONTEXT_CHARS = 8000  # ~2000 tokens — keep small for speed
        if len(context_block) > _MAX_CONTEXT_CHARS:
            logger.warning(
                "[AGENT] Context block too large (%d chars), truncating to %d",
                len(context_block), _MAX_CONTEXT_CHARS,
            )
            context_block = context_block[:_MAX_CONTEXT_CHARS] + "\n... (context truncated)\n"

        # --- Choose snippet for prompt ---
        func_start, func_end = 0, len(file_lines)

        if syntax_errors:
            err_line = syntax_errors[0].get('lineno') or 1
            original_snippet, _ = get_focused_context(file_source, err_line, window=20)
            original_header = f"ORIGINAL FILE TO EDIT (focused around line {err_line}):\n"
        elif multi_func_hint:
            # Multi-function — LLM needs to see the whole file
            original_snippet = file_source
            original_header = "ORIGINAL FILE TO EDIT (full file — fix all functions listed in the instruction):\n"
            func_start, func_end = 0, len(file_lines)
        else:
            focused, func_start, func_end = extract_function(file_source, instruction)
            has_call_sites = focused and '// ── CALL SITE' in focused
            if focused and len(focused.splitlines()) < 120:
                original_snippet = focused
                if has_call_sites:
                    original_header = "ORIGINAL FILE TO EDIT (function + its call sites — you may need SEARCH/REPLACE blocks for BOTH the function AND its callers):\n"
                else:
                    original_header = "ORIGINAL FILE TO EDIT (relevant function):\n"
            else:
                original_snippet = file_source
                original_header = "ORIGINAL FILE TO EDIT:\n"
                func_start, func_end = 0, len(file_lines)

        FORMAT_BLOCK = (
            f"```{lang_fence}\n"
            "<<<<<<< SEARCH\n"
            "(exact lines of old code here)\n"
            "=======\n"
            "(new edited code here)\n"
            ">>>>>>> REPLACE\n"
            "```"
        )

        # --- Cap original_snippet to prevent enormous prompts ---
        _MAX_SNIPPET_CHARS = 12000  # ~3000 tokens — shorter = faster LLM response
        if len(original_snippet) > _MAX_SNIPPET_CHARS:
            logger.warning(
                "[AGENT] File snippet too large (%d chars), truncating to %d",
                len(original_snippet), _MAX_SNIPPET_CHARS,
            )
            original_snippet = original_snippet[:_MAX_SNIPPET_CHARS] + "\n// ... (file truncated — remaining code omitted)\n"

        # Log total prompt budget for debugging
        _total_prompt_est = len(original_snippet) + len(context_block) + len(instruction) + 500
        logger.info(
            "[AGENT] Prompt budget: snippet=%d ctx=%d instruction=%d total=~%d chars (~%d tok)",
            len(original_snippet), len(context_block), len(instruction),
            _total_prompt_est, _total_prompt_est // 4,
        )

        llm = make_llm(temperature=0.0)

        if task_mode == "solve":
            is_implement = True
            logger.info("[AGENT] task_mode=solve — build mode active")
        elif task_mode == "fix":
            is_implement = False
            logger.info("[AGENT] task_mode=fix — fix mode active")
        else:
            is_implement = self._is_implement_task(instruction)
            if is_implement:
                logger.info("[AGENT] task_mode=auto detected implementation task — switching to build mode")

        # --- Planner step for implement tasks ---
        plan_context = ""
        if is_implement:
            try:
                plan = plan_task(instruction, file_source, ext, rag_context=rag_context, llm=llm)
                plan_context = format_plan_for_prompt(plan)
                logger.info("[AGENT] Planner produced %d steps, approach: %s", len(plan.get('steps', [])), plan.get('approach', '')[:200])
                for i, step in enumerate(plan.get('steps', []), 1):
                    logger.info("[AGENT]   Step %d: %s", i, step[:300])
            except Exception as e:
                logger.debug("[AGENT] Planner failed, continuing without plan: %s", e)

        # --- Detect crash-type errors for targeted analysis ---
        _crash_keywords = {
            'stack overflow', 'stack-overflow', 'buffer overflow', 'buffer-overflow',
            'out of bounds', 'out-of-bounds', 'segfault', 'segmentation fault',
            'heap-buffer-overflow', 'stack-buffer-overflow', 'addresssanitizer',
            'asan', 'runtime error', 'access violation', 'index out of range',
            'array index', 'undefined behavior', 'undefined behaviour',
        }
        instruction_lower_crash = instruction.lower()
        is_crash_bug = any(kw in instruction_lower_crash for kw in _crash_keywords)
        crash_analysis_hint = ""
        if is_crash_bug:
            crash_analysis_hint = (
                "\n\nCRASH BUG ANALYSIS CHECKLIST — the error is a runtime crash, NOT a logic error:\n"
                "1. Check EVERY array/vector/string index expression. For each one:\n"
                "   - What is the maximum value the index can reach?\n"
                "   - What is the array's actual length?\n"
                "   - Does the index ALWAYS stay in [0, length-1] for ALL loop iterations?\n"
                "2. Check for off-by-one errors in loop bounds.\n"
                "3. If patterns/arrays are length N, any index like `N + i` is OUT OF BOUNDS.\n"
                "   Use `(N + i) % N` or `(N + i - 1) % N` to wrap around.\n"
                "4. Check for integer overflow in index arithmetic.\n"
                "5. 'Stack overflow' on LeetCode usually means MEMORY ACCESS violation, not recursion.\n"
            )
            logger.info("[AGENT] Crash-type bug detected — injecting bounds-checking analysis hints")

        # --- Pattern classification pre-pass ---
        pattern_classification = ""
        if is_implement or is_crash_bug or "fix" in instruction.lower():
            pattern_classification = self._classify_problem_pattern(
                instruction, file_source, ext, llm
            )
        pattern_block = ""
        if pattern_classification:
            pattern_block = (
                f"\nALGORITHM CLASSIFICATION (use this to guide your solution):\n"
                f"{pattern_classification}\n\n"
            )

        # --- Two-stage: analysis (reasoning, temp=0.1) then edit (coding, temp=0.0) ---
        # Use lower num_predict for analysis to keep it concise and fast
        reasoning_llm = make_llm(temperature=0.1, num_predict=2048)
        logger.info("Waiting for LLM to analyze the issue...")
        try:
            if is_implement:
                plan_section = f"\nPLAN:\n{plan_context}\n" if plan_context else ""
                analysis_prompt = (
                    f"You are solving a programming task. Read the requirement carefully and plan the solution.\n\n"
                    f"{lang_directive}"
                    f"TASK: {instruction}\n\n"
                    f"{plan_section}"
                    f"{pattern_block}"
                    f"{original_header}```{lang_fence}\n{original_snippet}\n```\n\n"
                    "Describe step-by-step what algorithm or logic you will implement to satisfy the requirement. "
                    "Be specific about data structures, loop structure, and edge cases.\n\n"
                    f"{context_block}"
                )
            else:
                analysis_prompt = (
                    f"Look at this code and describe what needs to change to fix: {instruction}\n\n"
                    f"{lang_directive}"
                    f"{pattern_block}"
                    f"{original_header}```{lang_fence}\n{original_snippet}\n```\n\n"
                    f"{crash_analysis_hint}"
                    f"{context_block}"
                )
            analysis = reasoning_llm.invoke(analysis_prompt).strip()

            logger.info("Analysis received; requesting SEARCH/REPLACE from LLM...")
            
            block_instruction = self._build_block_instruction(
                multi_func_hint, instruction, lang_fence
            )

            if is_implement:
                edit_prompt = (
                    f"Given this plan:\n{analysis}\n\n"
                    f"{lang_directive}"
                    f"{block_instruction}"
                    "You are IMPLEMENTING a complete solution inside an existing stub function.\n"
                    "1. SEARCH must EXACTLY match the existing stub lines character-for-character including indentation.\n"
                    "2. REPLACE must contain the FULL working implementation — do not leave the body empty.\n"
                    "3. Include a few surrounding lines in SEARCH to make it unique.\n"
                    "4. Output ONLY the block(s) — no explanatory text.\n\n"
                    "CRITICAL: Write a real, complete algorithm in REPLACE. Do not output a placeholder or pass.\n\n"
                    "PRESERVATION RULE: Only modify the specific function(s) relevant to the task. "
                    "Do NOT delete, rewrite, or omit any other functions, classes, or code in the file. "
                    "If the file contains other code that is unrelated to this task, leave it COMPLETELY untouched. "
                    "Your SEARCH block should target ONLY the specific function being implemented.\n\n"
                    f"{original_header}```{lang_fence}\n{original_snippet}\n```\n\n"
                    f"TASK: {instruction}\n\n"
                    f"{context_block}"
                )
            else:
                edit_prompt = (
                    f"Given this analysis:\n{analysis}\n\n"
                    f"{lang_directive}"
                    f"{block_instruction}"
                    "1. SEARCH must EXACTLY match existing code character-for-character including indentation.\n"
                    "2. Include a few lines of surrounding context so SEARCH is unique in the file.\n"
                    "3. When ADDING code, keep the surrounding lines in REPLACE so they are not deleted.\n"
                    "4. Output ONLY the blocks — no explanatory text, no comments.\n\n"
                    "CRITICAL RULE: Your SEARCH blocks must contain ONLY the specific lines being changed (max 20-30 lines each). "
                    "Output all blocks back-to-back with no commentary between them.\n\n"
                    "PRESERVATION RULE: Only modify code directly relevant to the instruction. "
                    "Do NOT delete, rewrite, or omit any unrelated functions, classes, or code in the file.\n\n"
                    f"{original_header}```{lang_fence}\n{original_snippet}\n```\n\n"
                    f"INSTRUCTION: {instruction}\n\n"
                    f"{context_block}"
                )
            raw_output = llm.invoke(edit_prompt).strip()
        except Exception as e:
            logger.error("LLM invocation failed: %s", e)
            logger.debug(traceback.format_exc())
            return file_source

        # --- Parse blocks with auto-retry ---
        blocks, raw_output = parse_blocks_with_retry(raw_output)

        for i, (s, r) in enumerate(blocks):
            logger.debug("Block %s SEARCH:\n%s", i + 1, s)
            logger.debug("Block %s REPLACE:\n%s", i + 1, r)

        # --- Auto-retry if no valid blocks found ---
        # Skip expensive format retry chain — go straight to whole-function rewrite
        format_failed = False
        if not blocks:
            logger.warning("Bad format from LLM: SEARCH/REPLACE not found. Skipping to whole-function rewrite.")
            format_failed = True

        if format_failed:
            # Jump to whole-function rewrite — extract_function needs to run first
            verbatim, func_start, func_end = extract_function(file_source, instruction, hint_blocks=None)
            new_source = file_source

        # --- Apply blocks sequentially (skip if format failed) ---
        if not format_failed:
            source_lines = _apply_repair_blocks(blocks, file_lines[:], prefix="Block")

            post_errors = collect_syntax_errors("\n".join(source_lines), path) if is_python else []

            if post_errors and not syntax_errors:
                logger.warning("LLM edits introduced %s new syntax error(s) in %s", len(post_errors), path)
                current_source = "\n".join(source_lines)
                current_lines = current_source.split('\n')
                post_error_note = build_post_error_note(post_errors, current_lines)
                reinvoke_prompt = (
                    f"CURRENT FILE STATE:\n```{lang_fence}\n{current_source}\n```\n\n"
                    "Your previous edits introduced new syntax errors listed below. "
                    "Fix them without reverting the intended changes.\n\n"
                    f"{post_error_note}"
                    "Output ONE SEARCH/REPLACE block per error, top to bottom.\n"
                    "Each SEARCH block must match the CURRENT FILE STATE exactly."
                )
                try:
                    raw_output = llm.invoke(reinvoke_prompt).strip()
                    repair_blocks, raw_output = parse_blocks_with_retry(raw_output)
                    source_lines = _apply_repair_blocks(
                        repair_blocks, source_lines, prefix="Self-repair Block"
                    )
                except Exception as e:
                    logger.error("Self-repair LLM call failed: %s", e)
                    logger.debug(traceback.format_exc())

            new_source = "\n".join(source_lines)

        # --- Structural validation (all languages: brace balance, unreachable code, indentation) ---
        if not format_failed:
            structural_issues = validate_code_structure(new_source, ext)
            if structural_issues and new_source != file_source:
                logger.warning("Structural issues after edit: %s", [i['msg'] for i in structural_issues])
                structural_note = build_structural_issue_note(structural_issues, source_lines)
                try:
                    repair_prompt = (
                        f"CURRENT FILE STATE:\n```{lang_fence}\n{new_source}\n```\n\n"
                        "Your previous edits introduced structural issues. "
                        "Fix them without reverting the intended changes.\n\n"
                        f"{structural_note}"
                        "Output ONE SEARCH/REPLACE block per issue.\n"
                        "Each SEARCH block must match the CURRENT FILE STATE exactly."
                    )
                    raw_repair = llm.invoke(repair_prompt).strip()
                    repair_blocks, _ = parse_blocks_with_retry(raw_repair)
                    source_lines = _apply_repair_blocks(
                        repair_blocks, source_lines, prefix="Structural-repair"
                    )
                    new_source = "\n".join(source_lines)
                except Exception as e:
                    logger.warning("Structural repair LLM call failed: %s", e)

        if not format_failed and new_source == file_source and blocks:
            logger.warning("Blocks parsed but no changes landed — triggering verbatim retry")

            try:
                largest_block = max(len(s.strip().split('\n')) for s, _ in blocks)
            except Exception:
                largest_block = 0

            # reuse pre-computed file_lines instead of re-splitting
            file_line_count = len(file_lines)

            if largest_block > file_line_count * MAX_BLOCK_RATIO:
                logger.warning("Full-file replace detected — retrying with surgical edit instruction")
                verbatim, func_start, func_end = extract_function(file_source, instruction, hint_blocks=blocks)
                verbatim_hint = (
                    "Your previous response replaced the entire file. That is not allowed.\n"
                    "You must output a SEARCH/REPLACE block targeting ONLY the specific function or lines to change.\n"
                    "Here is the EXACT function content to use in your SEARCH block:\n"
                    f"```\n{verbatim}\n```\n"
                    "Output ONE targeted SEARCH/REPLACE block for this function only.\n\n"
                    f"FORMAT:\n{FORMAT_BLOCK}\n\n"
                )
            else:
                verbatim, func_start, func_end = extract_function(file_source, instruction, hint_blocks=None)
                mismatch_idx, mismatch_line = find_first_mismatch(
                    file_lines,  # reuse file_lines
                    blocks[0][0].strip().split('\n')
                )
                mismatch_msg = (
                    f"\nSpecifically, line {mismatch_idx + 1} of your SEARCH block:\n"
                    f"  `{mismatch_line}`\n"
                    "does not exist verbatim in the file.\n"
                ) if mismatch_idx >= 0 else ""

                verbatim_hint = (
                    "Your previous SEARCH block did not match the file."
                    + mismatch_msg +
                    "Here is the EXACT current content — use these lines verbatim in your SEARCH block:\n"
                    f"```\n{verbatim}\n```\n"
                    f"Re-output your SEARCH/REPLACE block.\n\n"
                    f"FORMAT:\n{FORMAT_BLOCK}\n\n"
                    "Previous LLM output:\n" + raw_output[:8000]
                )

            try:
                raw_output = llm.invoke(verbatim_hint).strip()
                retry_blocks, raw_output = parse_blocks_with_retry(raw_output)
                if retry_blocks:
                    source_lines = file_lines[:]
                    source_lines = _apply_repair_blocks(
                        retry_blocks, source_lines, prefix="Retry Block"
                    )
                    new_source = "\n".join(source_lines)
            except Exception as e:
                logger.error("Verbatim retry LLM invocation failed: %s", e)
                logger.debug(traceback.format_exc())

        # --- Whole-function rewrite fallback ---
        if new_source == file_source and (blocks or format_failed):
            if format_failed:
                logger.warning("Format failed — skipping retry chain, going direct to whole-function rewrite")
            else:
                logger.warning("Verbatim retry also failed — falling back to whole-function rewrite")
            # Strong language reminder at end of prompt (7B models lose early context)
            lang_reminder = f"\nREMINDER: Output {lang_name} code ONLY. The file extension is {ext}. Do NOT write Python or any other language.\n"
            # Prevent LLM from regenerating file-level declarations
            no_package_rule = (
                "5. Do NOT output 'package', 'import', '#include', or any file-level declarations. "
                "Output ONLY the function body/definition — the file already has its own package and imports.\n"
            )

            # If verbatim is empty, the instruction is about a NEW function
            # that doesn't exist yet — generate it and APPEND to the file.
            _is_new_function = not verbatim.strip()

            if _is_new_function:
                rewrite_prompt = (
                    f"{lang_directive}"
                    f"Write a NEW function to satisfy the following task.\n\n"
                    f"TASK: {instruction}\n\n"
                    f"EXISTING FILE:\n```{lang_fence}\n{file_source[:3000]}\n```\n\n"
                    "Rules:\n"
                    "1. Output ONLY the new function — no explanation, no markdown fence.\n"
                    "2. Do NOT reproduce any existing functions from the file.\n"
                    "3. Match the coding style and indentation of the existing file.\n"
                    f"{no_package_rule}"
                    f"{context_block}"
                    f"{lang_reminder}"
                )
            elif is_implement:
                rewrite_prompt = (
                    f"{lang_directive}"
                    f"Implement a complete solution for the following task inside the function stub below.\n\n"
                    f"TASK: {instruction}\n\n"
                    f"FUNCTION STUB:\n```{lang_fence}\n{verbatim}\n```\n\n"
                    "Rules:\n"
                    "1. Output ONLY the complete rewritten function — no explanation, no markdown fence.\n"
                    "2. Write a real, working algorithm. Do NOT leave the body empty or return a placeholder.\n"
                    "3. Keep exact indentation of the original.\n"
                    "4. Output ONLY this function — do NOT include other functions from the file.\n"
                    f"{no_package_rule}"
                    f"{context_block}"
                    f"{lang_reminder}"
                )
            else:
                rewrite_prompt = (
                    f"{lang_directive}"
                    f"Rewrite the following code to apply this change: {instruction}\n\n"
                    f"CURRENT CODE:\n```{lang_fence}\n{verbatim}\n```\n\n"
                    "Rules:\n"
                    "1. Output ONLY the rewritten code — no explanation, no markdown fence.\n"
                    "2. Preserve ALL existing logic except what the instruction explicitly changes.\n"
                    "3. Keep exact indentation of the original.\n"
                    "4. Do NOT remove, rewrite, or omit any code outside the targeted function.\n"
                    f"{context_block}"
                    f"{lang_reminder}"
                )
            try:
                rewritten = llm.invoke(rewrite_prompt).strip()
                rewritten = strip_markdown(rewritten) if "```" in rewritten else rewritten

                # Strip package/import declarations that LLM should not have generated
                rewritten = self._strip_file_level_declarations(rewritten, ext)

                if rewritten.strip():
                    # Language guard: hard-fail if LLM generated the wrong language
                    if not source_matches_ext(rewritten, ext):
                        lang_name = LANG_NAMES.get(ext, ext)
                        logger.warning(
                            "Whole-function rewrite rejected — LLM generated wrong language "
                            "(expected %s). Retrying with explicit language constraint.", ext
                        )
                        # Structured retry: explicitly demand the correct language
                        retry_prompt = (
                            f"YOUR PREVIOUS RESPONSE WAS IN THE WRONG LANGUAGE.\n"
                            f"You MUST respond ONLY in {lang_name} ({ext}). "
                            f"Do NOT use any other programming language.\n\n"
                            f"{rewrite_prompt}"
                        )
                        rewritten = llm.invoke(retry_prompt).strip()
                        rewritten = strip_markdown(rewritten) if "```" in rewritten else rewritten
                        rewritten = self._strip_file_level_declarations(rewritten, ext)
                        if not rewritten.strip() or not source_matches_ext(rewritten, ext):
                            logger.error(
                                "Language retry failed — still wrong language for %s. Keeping original.", ext
                            )
                            rewritten = None  # force skip to keep original
                    if rewritten and rewritten.strip():
                        # Safety check: reject if content is destroyed (rewritten loses >30% of lines on large files)
                        original_span_lines = file_lines[func_start:func_end]
                        rewritten_lines = rewritten.strip().split('\n')
                        line_ratio = len(rewritten_lines) / max(1, len(original_span_lines))

                        # Guard: if the span covers the entire file and has multiple functions,
                        # the rewrite would destroy unrelated code — refuse and keep original.
                        is_full_file = func_start == 0 and func_end >= len(file_lines)
                        func_count = count_top_level_functions(file_lines, ext)
                        span_func_count = count_top_level_functions(original_span_lines, ext)
                        rewritten_func_count = count_top_level_functions(rewritten_lines, ext)
                        if is_full_file and func_count > 1:
                            logger.warning(
                                "Whole-function rewrite rejected — span covers the entire file "
                                "(%d functions detected). Refusing to destroy unrelated code.",
                                func_count
                            )
                        elif span_func_count > 1 and rewritten_func_count < span_func_count:
                            logger.warning(
                                "Whole-function rewrite rejected — span has %d functions but "
                                "rewrite only has %d. Would destroy sibling functions.",
                                span_func_count, rewritten_func_count
                            )
                        elif line_ratio < 0.70 and len(original_span_lines) > 30:
                            logger.warning(
                                "Whole-function rewrite rejected — rewritten has only %.0f%% of original lines "
                                "(%d vs %d), likely data loss. Refusing to apply.",
                                line_ratio * 100, len(rewritten_lines), len(original_span_lines)
                            )
                        else:
                            if _is_new_function:
                                # Append the new function to the file instead of replacing
                                new_source = file_source.rstrip() + "\n\n" + rewritten.strip() + "\n"
                                logger.info("New function appended to file")
                            else:
                                new_source = whole_function_replace(file_source, rewritten, func_start, func_end)
                                logger.info("Whole-function rewrite applied (lines %s–%s)", func_start, func_end)

                        # Go file safety: ensure package declaration survived the rewrite
                        if ext == '.go' and new_source != file_source:
                            if not any(ln.strip().startswith('package ') for ln in new_source.split('\n')[:5]):
                                # Recover package line from original source
                                for orig_line in file_source.split('\n'):
                                    if orig_line.strip().startswith('package '):
                                        new_source = orig_line + '\n\n' + new_source
                                        logger.info("[SAFETY] Restored missing Go package declaration")
                                        break
                else:
                    logger.error("Whole-function rewrite returned empty output. Giving up.")
            except Exception as e:
                logger.error("Whole-function rewrite LLM call failed: %s", e)
                logger.debug(traceback.format_exc())

        # --- Targeted line-edit fallback ---
        # When all SEARCH/REPLACE attempts fail, try to extract error line numbers
        # from the instruction (e.g., AddressSanitizer, compiler errors, user mentions)
        # and ask the LLM to fix just those specific lines.
        if new_source == file_source:
            error_lines = extract_error_lines_from_text(instruction)
            if error_lines:
                logger.info("[AGENT] Targeted line-edit fallback: error lines %s extracted from instruction", error_lines)
                # Build focused context around each error line
                focus_snippets = []
                for err_line in error_lines[:3]:  # limit to 3 error sites
                    if 1 <= err_line <= len(file_lines):
                        start = max(0, err_line - 5)
                        end = min(len(file_lines), err_line + 5)
                        snippet = '\n'.join(file_lines[start:end])
                        focus_snippets.append(
                            f"Around line {err_line}:\n```{lang_fence}\n{snippet}\n```"
                        )

                if focus_snippets:
                    focused_context = '\n\n'.join(focus_snippets)
                    targeted_prompt = (
                        f"The user reported an error in this {lang_name} code.\n\n"
                        f"{lang_directive}"
                        f"ERROR DESCRIPTION: {instruction}\n\n"
                        f"FULL FILE:\n```{lang_fence}\n{file_source}\n```\n\n"
                        f"ERROR LOCATIONS:\n{focused_context}\n\n"
                        f"Fix ONLY the lines causing the error. Output a SEARCH/REPLACE block.\n"
                        f"The SEARCH block must match the EXACT current lines character-for-character.\n"
                        f"FORMAT:\n{FORMAT_BLOCK}\n"
                    )
                    try:
                        raw_output = llm.invoke(targeted_prompt).strip()
                        targeted_blocks, _ = parse_blocks_with_retry(raw_output)
                        if targeted_blocks:
                            source_lines = file_lines[:]
                            source_lines = _apply_repair_blocks(
                                targeted_blocks, source_lines, prefix="Targeted"
                            )
                            new_source = "\n".join(source_lines)
                    except Exception as e:
                        logger.warning("[AGENT] Targeted line-edit fallback failed: %s", e)

        # --- Self-correction ---
        new_source = self._self_correct_output(
            llm, new_source, file_source, is_implement, instruction, lang_fence, FORMAT_BLOCK
        )

        # --- Post-edit function count safety check ---
        # If the edit decreased the number of top-level functions, something
        # went wrong (likely the whole-function rewrite destroyed siblings).
        # Reject the edit and keep the original.
        if new_source != file_source:
            orig_func_count = count_top_level_functions(file_lines, ext)
            new_func_count = count_top_level_functions(new_source.split('\n'), ext)
            if orig_func_count > 1 and new_func_count < orig_func_count:
                logger.warning(
                    "[SAFETY] Edit rejected — function count dropped from %d to %d. "
                    "Restoring original to prevent data loss.",
                    orig_func_count, new_func_count,
                )
                new_source = file_source

        # --- Finalize: diff, explanation, write ---
        return self._finalize_edit(
            file_source, new_source, path, ext, dry_run,
            instruction, is_implement, llm, rag_citations, session_chat_history,
            session_id=session_id, rag_context=rag_context
        )

    # ─── Extracted helper methods for edit_code ─────────────────────────────

    @staticmethod
    def _build_block_instruction(multi_func_hint, instruction, lang_fence):
        """Build single or multi-block SEARCH/REPLACE instruction for the edit prompt."""
        if not multi_func_hint:
            return "Now produce a SEARCH/REPLACE block. Rules:\n"

        func_matches = re.findall(r'\b(\w+)\(\)', instruction)
        quoted_items = re.findall(r'"([^"]{5,})"', instruction)
        func_name_mentions = re.findall(
            r'\b(?:the\s+)?(\w+)\s+(?:function|method)\b', instruction, re.IGNORECASE
        )

        if quoted_items:
            task_labels = quoted_items
        elif func_matches:
            task_labels = [f"{n}()" for n in func_matches]
        elif func_name_mentions:
            task_labels = list(dict.fromkeys(func_name_mentions))
        else:
            task_labels = ["each fix listed in the instruction"]

        task_count = len(task_labels)
        task_list = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(task_labels))

        return (
            f"Now produce ONE SEARCH/REPLACE block for EACH of the {task_count} tasks:\n{task_list}\n\n"
            f"You MUST output {task_count} separate blocks — one COMPLETE block per task.\n"
            "For EACH task, output BOTH the SEARCH and REPLACE sections back-to-back "
            "with no text between them.\n\n"
            "TEMPLATE TO REPEAT FOR EACH TASK:\n"
            f"```{lang_fence}\n"
            "<<<<<<< SEARCH\n(exact old code for this fix)\n=======\n"
            "(exact new code for this fix)\n>>>>>>> REPLACE\n```\n\n"
            f"Output all {task_count} blocks sequentially with no commentary between them.\n"
        )

    @staticmethod
    def _self_correct_output(llm, new_source, file_source, is_implement,
                             instruction, lang_fence, FORMAT_BLOCK):
        """Self-correction: LLM grades its own output and fixes logical errors."""
        if not (is_implement and new_source != file_source):
            return new_source

        logger.info("[AGENT] Self-correction: grading generated code...")
        try:
            grade_prompt = (
                f"You just wrote code to satisfy this requirement:\n"
                f"REQUIREMENT: {instruction}\n\n"
                f"YOUR CODE:\n```{lang_fence}\n{new_source}\n```\n\n"
                "Grade this code CRITICALLY. You MUST do the following:\n"
                "1. Does the code implement the CORRECT function matching the requirement?\n"
                "   (e.g., if the requirement says 'getHappyString', does the code define 'getHappyString'?)\n"
                "2. TRACE TEST: Pick a concrete input from the requirement (or make one up if none given).\n"
                "   Walk through YOUR code line by line with that input.\n"
                "   Show the variable values at each step.\n"
                "   Does it produce the expected output? If not, what goes wrong and where?\n"
                "3. Does it handle edge cases (empty input, single element, max values)?\n"
                "4. Were any existing unrelated functions accidentally modified or removed?\n\n"
                "You MUST show the trace before giving your verdict.\n\n"
                "After the trace, respond with EXACTLY one of these two lines:\n"
                "  CORRECT: <brief reason why it fully satisfies the requirement>\n"
                "  ISSUE: <specific description of the bug — reference the trace step where it fails>\n\n"
                "Be STRICT. If the trace shows wrong output, say ISSUE."
            )
            grade_result = llm.invoke(grade_prompt).strip()
            logger.info("[AGENT] Self-grade: %s", grade_result[-300:])

            # The LLM outputs a trace first, then CORRECT: or ISSUE: — search the whole response
            import re as _re
            issue_match = _re.search(r'^ISSUE:\s*(.+)', grade_result, _re.MULTILINE | _re.IGNORECASE)
            correct_match = _re.search(r'^CORRECT:\s*(.+)', grade_result, _re.MULTILINE | _re.IGNORECASE)

            if issue_match and not correct_match:
                issue = issue_match.group(1).strip()
                logger.info("[AGENT] Self-correction triggered: %s", issue)
                correction_prompt = (
                    f"Your previous code has a logical error:\n"
                    f"ISSUE: {issue}\n\n"
                    f"ORIGINAL REQUIREMENT: {instruction}\n\n"
                    f"CURRENT CODE:\n```{lang_fence}\n{new_source}\n```\n\n"
                    "Fix ONLY the described issue. Output a SEARCH/REPLACE block.\n"
                    "1. SEARCH must match the current code character-for-character.\n"
                    "2. REPLACE must contain the corrected logic.\n"
                    "3. Output ONLY the block — no explanation.\n\n"
                    f"FORMAT:\n{FORMAT_BLOCK}"
                )
                raw = llm.invoke(correction_prompt).strip()
                blocks, _ = parse_blocks_with_retry(raw)
                if blocks:
                    lines = new_source.split('\n')
                    lines = _apply_repair_blocks(blocks, lines, prefix="Self-correction")
                    new_source = "\n".join(lines)
                    logger.info("[AGENT] Self-correction complete")
                else:
                    logger.warning("[AGENT] Self-correction produced no valid blocks.")
            else:
                logger.info("[AGENT] Code graded as correct.")
        except Exception as e:
            logger.warning("[AGENT] Self-correction failed: %s", e)
            logger.debug(traceback.format_exc())

        return new_source

    def _finalize_edit(self, file_source, new_source, path, ext, dry_run,
                       instruction, is_implement, llm, rag_citations,
                       session_chat_history, session_id=None, rag_context=""):
        """Generate diff, critic review, explanation, and optionally write the edited file."""
        diff = show_diff(file_source, new_source)
        logger.info("--- DIFF PREVIEW ---")
        logger.info('\n%s', diff or "No changes detected.")
        logger.info("--------------------")

        lang_fence = LANG_FENCE.get(ext, ext.lstrip('.') if ext else 'text')

        # --- Critic review ---
        critic_note = ""
        if new_source != file_source:
            try:
                critique = critique_code(
                    instruction=instruction,
                    original_source=file_source,
                    new_source=new_source,
                    ext=ext,
                    rag_context=rag_context,
                    llm=llm,
                )
                logger.info("[AGENT] Critic verdict: %s (confidence: %.1f)", critique["verdict"], critique["confidence"])
                if critique["verdict"] != "PASS" and critique["issues"]:
                    critic_note = build_critic_feedback_note(critique)
                    logger.info("[AGENT] Critic found issues, attempting fix...")
                    # One-shot fix based on critic feedback
                    try:
                        fix_prompt = (
                            f"CURRENT CODE:\n```{lang_fence}\n{new_source}\n```\n\n"
                            f"{critic_note}\n\n"
                            "Output ONE SEARCH/REPLACE block per issue. "
                            "Each SEARCH block must match the CURRENT CODE exactly."
                        )
                        raw_fix = llm.invoke(fix_prompt).strip()
                        fix_blocks, _ = parse_blocks_with_retry(raw_fix)
                        if fix_blocks:
                            lines = new_source.split('\n')
                            lines = _apply_repair_blocks(fix_blocks, lines, prefix="Critic fix")
                            new_source = "\n".join(lines)
                            diff = show_diff(file_source, new_source)
                    except Exception as e:
                        logger.debug("[AGENT] Critic fix failed: %s", e)
            except Exception as e:
                logger.debug("[AGENT] Critic review failed: %s", e)

        # --- Generate teaching-style explanation ---
        explanation = ""
        if new_source != file_source:
            try:
                explain_prompt = (
                    f"You just {'implemented' if is_implement else 'fixed'} code for this task:\n"
                    f"TASK: {instruction[:400]}\n\n"
                    f"YOUR CODE:\n```{lang_fence}\n{new_source[:2000]}\n```\n\n"
                    "Write an explanation for someone learning to code. Use this EXACT format:\n\n"
                    "## Key Concept\n"
                    "Name the core algorithm, data structure, or technique used (e.g. 'Two-Pointer Technique', "
                    "'Hash Map Lookup', 'Dummy Node Pattern'). One sentence explaining what it is.\n\n"
                    "## How It Works (Step-by-Step)\n"
                    "Walk through the code logic step by step using the ACTUAL variable names and values "
                    "from the code. Show what happens with a concrete example input.\n"
                    "Number each step. Max 5 steps.\n\n"
                    "## Why This Approach\n"
                    "One sentence on time/space complexity. One sentence on why this is better than a naive approach.\n\n"
                    "## Edge Cases\n"
                    "List 2-3 edge cases the code handles (or should handle), with brief explanation.\n\n"
                    "Keep it clear and beginner-friendly. Use the actual code to illustrate each point."
                )
                explanation = llm.invoke(explain_prompt).strip()
                logger.info("[AGENT] Teaching explanation: %s", explanation[:200])

                # Store explanation as a learning for tutor mode
                try:
                    from .tutor import store_agent_learning
                    lang_name_for_learning = LANG_NAMES.get(ext, ext.lstrip('.').upper() if ext else '')
                    store_agent_learning(
                        explanation=explanation,
                        topic=instruction[:100],
                        language=lang_name_for_learning,
                        file_path=path,
                    )
                except Exception:
                    pass  # don't let tutor bridge failure affect the agent

            except Exception as e:
                logger.debug("Explanation generation failed: %s", e)

        # --- Store in session memory ---
        if session_id and new_source != file_source:
            try:
                self.memory.add_turn(session_id, "user", instruction, {"file_path": path})
                self.memory.add_agent_action(session_id, instruction, diff, path)
            except Exception as e:
                logger.debug("[AGENT] Session memory store failed: %s", e)

        if dry_run:
            logger.info("Dry run mode; no file written.")
            return {
                "new_source": new_source,
                "diff": diff,
                "changed": new_source != file_source,
                "explanation": explanation,
                "citations": rag_citations,
            }

        try:
            write_file(path, new_source)
            try:
                run_git_command(["diff", "--", path])
            except Exception:
                pass
        except Exception as e:
            logger.error("Failed to write file %s: %s", path, e)
            return file_source

        if ext == ".go":
            try:
                subprocess.run(["gofmt", "-w", path], timeout=5, check=True)
                logger.info("gofmt applied to %s", path)
            except Exception:
                pass

        try:
            self.edit_log.setdefault(path, []).append({
                "instruction": instruction,
                "diff": diff,
                "timestamp": datetime.now().isoformat()
            })
        except Exception:
            pass

        if session_chat_history is not None:
            session_chat_history.append({"role": "User", "content": instruction})
            session_chat_history.append({"role": "Assistant", "content": f"Edited {path}: {instruction}"})

        logger.info("Wrote file %s", path)
        return new_source

    def _classify_problem_pattern(
        self, instruction: str, code: str, ext: str, llm
    ) -> str:
        """Classify the algorithm pattern to guide solution generation.
        
        Returns a structured analysis including:
        - Pattern type (binary search, DP, two-pointer, etc.)
        - Key invariants to maintain
        - Edge cases to handle
        - Time/space complexity expectations
        """
        prompt = (
            f"Analyze this {LANG_NAMES.get(ext, 'code')} problem and identify the algorithm pattern:\n\n"
            f"PROBLEM:\n{instruction}\n\n"
            f"CURRENT CODE (if any):\n```{LANG_FENCE.get(ext, 'text')}\n{code[:1500]}\n```\n\n"
            "Classify the algorithm pattern and provide:\n"
            "1. PATTERN: Name the algorithm type (e.g. 'binary search', 'dynamic programming', "
            "'two-pointer', 'sliding window', 'DFS backtracking', 'BFS', 'greedy', 'topological sort', etc.)\n"
            "2. CORE INVARIANTS: What conditions must ALWAYS remain true for the solution to work?\n"
            "3. EDGE CASES: List 3-5 specific edge cases this pattern commonly fails on.\n"
            "4. COMPLEXITY: Expected time and space complexity.\n"
            "5. SOLUTION TEMPLATE: 2-3 lines describing the high-level algorithm structure.\n\n"
            "Format:\n"
            "PATTERN: [type]\n"
            "INVARIANTS: [list]\n"
            "EDGE CASES: [list]\n"
            "COMPLEXITY: [O(?) time, O(?) space]\n"
            "TEMPLATE: [description]\n"
        )

        try:
            analysis = llm.invoke(prompt).strip()
            logger.info("[CLASSIFY_PATTERN] Identified: %s", analysis.split('\n')[0][:100])
            return analysis
        except Exception as e:
            logger.debug("[CLASSIFY_PATTERN] Failed: %s", e)
            return ""

    def _strategy_pivot(
        self,
        llm,
        instruction: str,
        current_source: str,
        ext: str,
        test_cases: list,
        test_results: list,
        failed_approaches: list,
    ) -> Optional[str]:
        """Rewrite the function from scratch using a fundamentally different algorithm.

        Called when the identical-error loop detector fires, meaning small tweaks
        to the current approach cannot fix the problem. Instead of retrying the
        same logic, this asks the LLM to pick a completely new strategy.
        """
        lang_fence = LANG_FENCE.get(ext, ext.lstrip('.') if ext else 'text')
        lang_name = LANG_NAMES.get(ext, ext.lstrip('.').upper() if ext else 'code')

        failures_block = ""
        for r in test_results:
            if not r["passed"]:
                got = r["actual"] if r["actual"] is not None else f"ERROR: {r.get('error', 'unknown')}"
                failures_block += f"  input={r['input']}  ->  expected={r['expected']},  got={got}\n"

        failed_block = ""
        if failed_approaches:
            failed_block = "APPROACHES ALREADY TRIED AND FAILED (do NOT reuse):\n"
            for fa in failed_approaches:
                failed_block += f"  - {fa}\n"
            failed_block += "\n"

        prompt = (
            f"LANGUAGE: You MUST write {lang_name} code. The file is {ext}. Do NOT use any other language.\n\n"
            f"Multiple fix attempts have failed with the same test errors, suggesting the "
            f"current approach has a structural issue that small edits cannot resolve.\n\n"
            f"PROBLEM: {instruction}\n\n"
            f"CURRENT CODE:\n```{lang_fence}\n{current_source}\n```\n\n"
            f"FAILING TESTS:\n{failures_block}\n"
            f"{failed_block}"
            f"Think step by step:\n"
            f"1. Re-read the problem statement carefully — what is actually being asked?\n"
            f"2. What specific logical error in the current approach causes these test failures?\n"
            f"3. What is the correct algorithm? (It may be a corrected version of the same approach, "
            f"or a different approach entirely — choose whatever produces correct results.)\n"
            f"4. Trace through your solution with the first failing test input to verify it works.\n\n"
            f"Write the COMPLETE rewritten function in {lang_name}.\n"
            f"Output ONLY the function code — no explanation, no markdown fences.\n"
            f"REMINDER: The output MUST be valid {lang_name} ({ext}) — not Python or any other language.\n"
        )

        try:
            # Use higher temperature + expanded context for creative divergence
            from brain.config import LLM_NUM_CTX_HARD, LLM_NUM_PREDICT_HARD, FALLBACK_LLM_MODEL
            
            # Route to stronger model if available and we've failed multiple approaches
            use_fallback = FALLBACK_LLM_MODEL and len(failed_approaches) >= 1
            model_override = FALLBACK_LLM_MODEL if use_fallback else None
            if use_fallback:
                logger.info("[STRATEGY_PIVOT] Escalating to stronger model: %s", FALLBACK_LLM_MODEL)
            
            pivot_llm = make_llm(
                model=model_override,
                temperature=0.3,
                num_ctx=LLM_NUM_CTX_HARD,
                num_predict=LLM_NUM_PREDICT_HARD,
            )
            rewritten = pivot_llm.invoke(prompt).strip()
            rewritten = strip_markdown(rewritten) if "```" in rewritten else rewritten
            rewritten = self._strip_file_level_declarations(rewritten, ext)

            if not rewritten.strip():
                logger.warning("[STRATEGY_PIVOT] LLM returned empty output")
                return None

            # Language guard: reject wrong-language pivots
            if not source_matches_ext(rewritten, ext):
                logger.warning("[STRATEGY_PIVOT] LLM generated wrong language (expected %s). Rejecting.", ext)
                return None

            # Apply via whole-function replace
            _, func_start, func_end = extract_function(current_source, instruction)
            new_source = whole_function_replace(current_source, rewritten, func_start, func_end)
            logger.info("[STRATEGY_PIVOT] New approach applied")
            return new_source
        except Exception as e:
            logger.error("[STRATEGY_PIVOT] Failed: %s", e)
            return None

    def fix_with_tests(
        self,
        path: str,
        instruction: str,
        test_cases: list,
        max_retries: int = 3,
        task_mode: str = "solve",
        session_id: Optional[str] = None,
    ) -> dict:
        """Iteratively fix code until all test cases pass or max_retries is exhausted.

        Features:
          - Debug tracing: runs failing tests with instrumented code to capture
            actual runtime variable values, giving the LLM concrete execution data
          - Self-reasoning: LLM reads execution trace to identify root cause
          - Diff context: shows LLM what changed in previous attempt (so it doesn't repeat)
          - Graduated escalation: targeted fix → debug-informed fix → strategy pivot
          - Degradation detector: stops if pass count drops between iterations
          - Temperature escalation: bumps temperature when stuck at same pass count

        Returns dict with final_source, test_results, all_passed, attempts, diff,
               explanation, citations.
        """
        from .test_runner import run_tests, build_test_failure_note, run_debug_trace, run_step_verification, format_step_verification_for_prompt

        llm = make_llm(temperature=0.0)
        original_source = read_file(path)
        source = original_source
        ext = os.path.splitext(path)[1].lower()
        all_results: list = []
        attempts = 0
        best_pass_count = 0
        best_source = source
        explanation_parts: list[str] = []
        citations: list[str] = []
        current_temperature = 0.0
        prev_error_signature = None  # Track repeated errors to break loops
        failed_approaches: list[str] = []  # Track failed strategies so pivots avoid them
        stagnant_count = 0  # How many consecutive attempts with no improvement
        previous_source = None  # Track previous source for diff context

        for attempt in range(max_retries + 1):
            logger.info("[FIX_WITH_TESTS] Attempt %d — running %d test case(s)...", attempt, len(test_cases))
            results = run_tests(source, path, test_cases, llm)
            all_results = results

            pass_count = sum(1 for r in results if r["passed"])
            failures = [r for r in results if not r["passed"]]

            # --- Structural validation ---
            structural_issues = validate_code_structure(source, ext)
            if structural_issues:
                logger.warning("[FIX_WITH_TESTS] Structural issues: %s",
                               [i['msg'] for i in structural_issues])

            # --- Identical-error loop detector + graduated escalation ---
            error_sig = frozenset(
                tuple((r['input'], str(r.get('actual', r.get('error', '')))))
                for r in results if not r['passed']
            ) | frozenset(i['msg'] for i in structural_issues)
            if error_sig and error_sig == prev_error_signature:
                stagnant_count += 1
                # Capture current approach description for the failed_approaches list
                try:
                    approach_desc = llm.invoke(
                        f"Describe in ONE sentence the algorithm/strategy used in this code:\n"
                        f"```\n{source[:2000]}\n```\n"
                        f"Output ONLY one sentence — no extra text."
                    ).strip()
                    if approach_desc and approach_desc not in failed_approaches:
                        failed_approaches.append(approach_desc)
                except Exception:
                    pass

                if stagnant_count >= 3:
                    logger.warning(
                        "[FIX_WITH_TESTS] Identical errors repeating after debug + pivot — stopping"
                    )
                    explanation_parts.append(
                        "Same errors repeating after debug trace and strategy pivot. Stopped to avoid infinite loop."
                    )
                    source = best_source
                    all_results = run_tests(source, path, test_cases, llm)
                    break

                # --- Stagnant count 1: Debug-trace-informed fix (before pivot) ---
                if stagnant_count == 1:
                    logger.info(
                        "[FIX_WITH_TESTS] Identical errors — running debug trace for deeper insight (attempt %d)",
                        attempt
                    )
                    explanation_parts.append(
                        f"Attempt {attempt}: Same errors detected — running debug trace to capture "
                        "actual runtime values for more targeted fix."
                    )
                    # Don't pivot yet — let debug trace inform the next fix attempt
                    # The debug trace will be picked up below in the failure_note builder

                # --- Stagnant count 2: Clean-slate rewrite with pattern classification ---
                elif stagnant_count >= 2:
                    logger.info(
                        "[FIX_WITH_TESTS] Identical errors persisting after debug fix — "
                        "forcing clean-slate rewrite with pattern classification (attempt %d)",
                        attempt
                    )
                    explanation_parts.append(
                        f"Attempt {attempt}: Debug-informed fix didn't resolve — "
                        "clean-slate rewrite with pattern classification."
                    )

                    # Step 1: Classify the problem pattern
                    pattern_class = self._classify_problem_pattern(
                        instruction, source, ext, llm
                    )

                    # Step 2: Gather trace data from failures for concrete context
                    trace_context = ""
                    if failures:
                        first_fail = failures[0]
                        for tc in failures[:3]:
                            trace_context += (
                                f"  input={tc['input']}  ->  expected={tc['expected']}"
                                f",  got={tc.get('actual', tc.get('error', 'CRASH'))}\n"
                            )

                    # Step 3: Force a from-scratch rewrite ignoring the broken code
                    lang_fence = LANG_FENCE.get(ext, ext.lstrip('.') if ext else 'text')
                    lang_name = LANG_NAMES.get(ext, ext.lstrip('.').upper() if ext else 'code')

                    failed_block = ""
                    if failed_approaches:
                        failed_block = "APPROACHES ALREADY TRIED AND FAILED (do NOT reuse):\n"
                        for fa in failed_approaches:
                            failed_block += f"  - {fa}\n"
                        failed_block += "\n"

                    pattern_section = ""
                    if pattern_class:
                        pattern_section = (
                            f"\nALGORITHM CLASSIFICATION:\n{pattern_class}\n\n"
                            "Use this classification to guide your implementation.\n"
                        )

                    rewrite_prompt = (
                        f"LANGUAGE: You MUST write {lang_name} code. The file is {ext}.\n\n"
                        f"The previous code was fundamentally broken. Do NOT patch it — "
                        f"write a COMPLETE new solution from scratch.\n\n"
                        f"PROBLEM: {instruction}\n\n"
                        f"{pattern_section}"
                        f"CONCRETE TEST DATA:\n{trace_context}\n"
                        f"{failed_block}"
                        f"Requirements:\n"
                        f"1. Write the COMPLETE function from scratch using the classified pattern.\n"
                        f"2. Trace through each failing test case mentally to verify correctness.\n"
                        f"3. Handle ALL edge cases identified in the classification.\n"
                        f"4. Output ONLY the function code — no explanation, no markdown fences.\n"
                        f"REMINDER: Valid {lang_name} ({ext}) only.\n"
                    )

                    try:
                        from brain.config import LLM_NUM_CTX_HARD, LLM_NUM_PREDICT_HARD
                        rewrite_llm = make_llm(
                            temperature=0.2,
                            num_ctx=LLM_NUM_CTX_HARD,
                            num_predict=LLM_NUM_PREDICT_HARD,
                        )
                        rewritten = rewrite_llm.invoke(rewrite_prompt).strip()
                        rewritten = strip_markdown(rewritten) if "```" in rewritten else rewritten
                        rewritten = self._strip_file_level_declarations(rewritten, ext)

                        if rewritten.strip() and source_matches_ext(rewritten, ext):
                            _, func_start, func_end = extract_function(source, instruction)
                            new_source = whole_function_replace(source, rewritten, func_start, func_end)
                            if new_source and new_source != source:
                                previous_source = source
                                source = new_source
                                write_file(path, source)
                                logger.info("[FIX_WITH_TESTS] Clean-slate rewrite applied — re-running tests")
                                continue
                    except Exception as e:
                        logger.error("[FIX_WITH_TESTS] Clean-slate rewrite failed: %s", e)

                    # Fallback: try old strategy pivot if clean-slate failed
                    pivot_source = self._strategy_pivot(
                        llm, instruction, source, ext, test_cases, results, failed_approaches
                    )
                    if pivot_source and pivot_source != source:
                        previous_source = source
                        source = pivot_source
                        write_file(path, source)
                        logger.info("[FIX_WITH_TESTS] Strategy pivot applied — re-running tests")
                        continue
                    else:
                        logger.warning("[FIX_WITH_TESTS] Strategy pivot produced no change — stopping")
                        source = best_source
                        all_results = run_tests(source, path, test_cases, llm)
                        break
            else:
                stagnant_count = 0
            prev_error_signature = error_sig

            if not failures:
                logger.info("[FIX_WITH_TESTS] All tests passed on attempt %d ✓", attempt)
                explanation_parts.append(f"Attempt {attempt}: All {len(results)} tests passed.")
                best_source = source
                break

            explanation_parts.append(
                f"Attempt {attempt}: {pass_count}/{len(results)} tests passing."
            )

            # --- Degradation detector ---
            if attempt > 0 and pass_count < best_pass_count:
                logger.warning(
                    "[FIX_WITH_TESTS] Degradation detected: %d passing → %d passing. "
                    "Reverting to best state and stopping.",
                    best_pass_count, pass_count
                )
                explanation_parts.append(
                    f"Degradation detected ({best_pass_count} → {pass_count} passing). "
                    "Reverted to best state and stopped."
                )
                source = best_source
                # Re-run tests on best source to get accurate final results
                all_results = run_tests(source, path, test_cases, llm)
                break

            if pass_count > best_pass_count:
                best_pass_count = pass_count
                best_source = source

            if attempt >= max_retries:
                logger.warning(
                    "[FIX_WITH_TESTS] Reached max retries (%d) with %d test(s) still failing.",
                    max_retries, len(failures)
                )
                break

            # --- Temperature + token escalation ---
            # If pass count is stagnant (same as last iteration), bump temperature
            # AND give the LLM more context/generation room for harder reasoning
            if attempt > 0 and pass_count == best_pass_count and current_temperature < 0.4:
                current_temperature = min(current_temperature + 0.15, 0.4)
                from brain.config import LLM_NUM_CTX_HARD, LLM_NUM_PREDICT_HARD
                llm = make_llm(
                    temperature=current_temperature,
                    num_ctx=LLM_NUM_CTX_HARD,
                    num_predict=LLM_NUM_PREDICT_HARD,
                )
                logger.info(
                    "[FIX_WITH_TESTS] Pass count stagnant — escalating temperature to %.2f, "
                    "num_ctx=%d, num_predict=%d",
                    current_temperature, LLM_NUM_CTX_HARD, LLM_NUM_PREDICT_HARD
                )
                explanation_parts.append(
                    f"Escalated temperature to {current_temperature:.2f} with expanded context "
                    f"(ctx={LLM_NUM_CTX_HARD}, predict={LLM_NUM_PREDICT_HARD}) for harder reasoning."
                )

            # --- Self-reasoning with debug trace: actual runtime values ---
            # Run debug trace on the first failing test to get real execution data
            debug_trace = None
            if failures:
                first_fail = failures[0]
                has_output = first_fail.get("actual") is not None
                has_crash = not has_output and first_fail.get("error")

                if has_output:
                    # Got a wrong output — run debug trace to capture runtime values
                    try:
                        logger.info("[FIX_WITH_TESTS] Running debug trace for failing test input=%s...",
                                    first_fail["input"][:60])
                        debug_trace = run_debug_trace(
                            source, path, first_fail["input"], first_fail["expected"], llm
                        )
                        if debug_trace:
                            logger.info("[FIX_WITH_TESTS] Debug trace captured (%d chars)", len(debug_trace))
                            explanation_parts.append(
                                f"Attempt {attempt}: Ran debug trace — captured runtime variable values."
                            )
                    except Exception as e:
                        logger.debug("[FIX_WITH_TESTS] Debug trace failed: %s", e)

                elif has_crash:
                    # Code crashed (segfault, OOB, stack overflow) — use the error message
                    # as the debug trace. ASan reports from test_runner will be included.
                    crash_error = first_fail["error"]
                    logger.info("[FIX_WITH_TESTS] Crash detected — using error as debug context (%d chars)",
                                len(crash_error))
                    debug_trace = (
                        "CRASH REPORT (code did not produce output — it crashed at runtime):\n"
                        f"{crash_error}\n\n"
                        "This is NOT a wrong-answer bug. The code CRASHES before producing any output.\n"
                        "Common causes: array/buffer out-of-bounds access, stack overflow from "
                        "unbounded recursion, null pointer dereference, integer overflow.\n"
                        "Look carefully at ALL array index expressions — check that every index "
                        "stays within [0, array_length-1] for ALL possible loop iterations."
                    )
                    explanation_parts.append(
                        f"Attempt {attempt}: Runtime crash detected — analyzing crash report."
                    )

            # --- Step-by-step assertion verification ---
            # Run structured verification to pinpoint EXACTLY which logical step fails.
            # This gives the edit agent concrete "Step 3 FAILED: got X, expected Y" data
            # instead of just "wrong output".
            step_verification_note = ""
            if failures and not all(r.get("actual") is None for r in failures):
                # Only run step verification when we have actual (wrong) output, not crashes
                first_fail = failures[0]
                if first_fail.get("actual") is not None:
                    try:
                        logger.info("[FIX_WITH_TESTS] Running step verification for input=%s...",
                                    first_fail["input"][:60])
                        step_result = run_step_verification(
                            source, path, first_fail["input"], first_fail["expected"],
                            instruction, llm
                        )
                        if step_result and step_result.get("steps"):
                            step_verification_note = format_step_verification_for_prompt(step_result)
                            n_pass = sum(1 for s in step_result["steps"] if s.get("passed"))
                            n_total = len(step_result["steps"])
                            logger.info("[FIX_WITH_TESTS] Step verification: %d/%d steps pass", n_pass, n_total)
                            if step_result.get("first_failure"):
                                fail_step = step_result["first_failure"]
                                explanation_parts.append(
                                    f"Attempt {attempt}: Step verification — bug at step {fail_step['step']}"
                                    f"{' (' + fail_step['description'] + ')' if fail_step.get('description') else ''}"
                                    f": got {fail_step['values'][:80]}"
                                )
                            else:
                                explanation_parts.append(
                                    f"Attempt {attempt}: Step verification — {n_pass}/{n_total} steps pass."
                                )
                    except Exception as e:
                        logger.debug("[FIX_WITH_TESTS] Step verification failed: %s", e)

            # Build diff from previous attempt to show LLM what already didn't work
            previous_diff = None
            if previous_source and previous_source != source:
                previous_diff = show_diff(previous_source, source)
                if previous_diff and len(previous_diff) > 4000:
                    previous_diff = previous_diff[:4000] + "\n... (truncated)"

            failure_note = build_test_failure_note(
                results, instruction, source=source, llm=llm,
                attempt=attempt, failed_approaches=failed_approaches,
                debug_trace=debug_trace, previous_diff=previous_diff,
                step_verification=step_verification_note,
            )
            augmented_instruction = f"{instruction}\n\n{failure_note}"

            # --- RAG-assisted error resolution ---
            # Query RAG specifically for the error pattern / algorithm to give
            # the edit agent relevant reference material (not just generic docs).
            if attempt >= 1 and failures:
                try:
                    first_fail = failures[0]
                    error_context = first_fail.get("error", "") or ""
                    actual = first_fail.get("actual", "") or ""
                    # Build a targeted query from the error + instruction
                    rag_error_query = f"{instruction[:200]} {error_context[:100]} {actual[:50]}".strip()
                    from brain.fast_search import fast_topic_search
                    from brain.config import EXT_TO_TOPIC
                    _topics = ["algorithms", "clean-code"]
                    _lang_topic = EXT_TO_TOPIC.get(ext)
                    if _lang_topic:
                        _topics.append(_lang_topic)
                    rag_error_results = fast_topic_search(
                        rag_error_query,
                        top_k=3,
                        rerank_method="keyword",
                        topic_filter=_topics,
                    )
                    if rag_error_results:
                        rag_error_text = "\n\n".join(
                            doc.page_content[:500] for doc in rag_error_results[:2]
                        )
                        augmented_instruction += (
                            f"\n\nREFERENCE (relevant algorithm/pattern from docs):\n"
                            f"{rag_error_text}\n"
                        )
                        logger.info("[FIX_WITH_TESTS] RAG error resolution: injected %d chars of relevant docs",
                                    len(rag_error_text))
                except Exception as e:
                    logger.debug("[FIX_WITH_TESTS] RAG error resolution failed: %s", e)

            # Include structural issues in the instruction so LLM can fix them
            if structural_issues:
                structural_note = build_structural_issue_note(structural_issues, source.split('\n'))
                augmented_instruction += f"\n\n{structural_note}"

            attempts += 1

            # Track current source before edit for next iteration's diff
            previous_source = source

            # Write current source so edit_code reads the right state
            write_file(path, source)

            edit_result = self.edit_code(
                path=path,
                instruction=augmented_instruction,
                dry_run=True,
                use_rag=True,
                task_mode=task_mode,
                session_id=session_id,
            )

            if isinstance(edit_result, dict) and edit_result.get("changed"):
                new_source_candidate = edit_result.get("new_source", source)
                # Language guard: reject if edit_code generated the wrong language
                if not source_matches_ext(new_source_candidate, ext):
                    logger.warning(
                        "[FIX_WITH_TESTS] edit_code generated wrong language (expected %s) — "
                        "discarding and stopping.", ext
                    )
                    explanation_parts.append(
                        f"Attempt {attempt}: LLM generated wrong language — discarded."
                    )
                    break
                source = new_source_candidate
                # Go file safety: ensure package declaration survived
                if ext == '.go' and not any(ln.strip().startswith('package ') for ln in source.split('\n')[:5]):
                    for orig_line in original_source.split('\n'):
                        if orig_line.strip().startswith('package '):
                            source = orig_line + '\n\n' + source
                            logger.info("[FIX_WITH_TESTS] Restored missing Go package declaration")
                            break
                logger.info("[FIX_WITH_TESTS] Code updated on attempt %d, re-running tests...", attempt)
            else:
                logger.warning("[FIX_WITH_TESTS] LLM made no changes on attempt %d — stopping.", attempt)
                explanation_parts.append(f"Attempt {attempt}: No code changes produced — stopped.")
                break

        # Use best source if final source is worse
        final_pass_count = sum(1 for r in all_results if r["passed"])
        if final_pass_count < best_pass_count:
            source = best_source
            all_results = run_tests(source, path, test_cases, llm)

        # Persist final source
        write_file(path, source)
        diff = show_diff(original_source, source)

        # --- Critic verification on final output ---
        critic_verdict = ""
        if source != original_source:
            try:
                critique = critique_code(
                    instruction=instruction,
                    original_source=original_source,
                    new_source=source,
                    ext=ext,
                    test_results=all_results,
                    llm=llm,
                )
                critic_verdict = f"Critic: {critique['verdict']} (confidence: {critique['confidence']:.1f})"
                if critique["issues"]:
                    critic_verdict += " | Issues: " + "; ".join(critique["issues"][:3])
                logger.info("[FIX_WITH_TESTS] %s", critic_verdict)
            except Exception as e:
                logger.debug("[FIX_WITH_TESTS] Critic failed: %s", e)

        # --- Store in session memory ---
        if session_id:
            try:
                self.memory.add_agent_action(
                    session_id, instruction, diff, path,
                    passed_tests=all(r["passed"] for r in all_results),
                )
            except Exception as e:
                logger.debug("[FIX_WITH_TESTS] Memory store failed: %s", e)

        # Build explanation summary
        explanation = (
            f"Agent worked on: {instruction[:120]}{'...' if len(instruction) > 120 else ''}\n"
            + "\n".join(explanation_parts)
        )
        if critic_verdict:
            explanation += f"\n{critic_verdict}"

        return {
            "final_source": source,
            "test_results": all_results,
            "all_passed": all(r["passed"] for r in all_results),
            "attempts": attempts,
            "diff": diff,
            "explanation": explanation,
            "citations": citations,
        }