import subprocess
import re
import logging
import sys
import traceback
from datetime import datetime
from types import SimpleNamespace
import os

from langchain_ollama import OllamaLLM
from typing import List, Dict, Optional

from brain.config import LLM_MODEL, LANG_NAMES, LANG_FENCE, IMPLEMENT_KEYWORDS
from brain.fast_search import fast_topic_search
from db.db_reader import get_code_documents
from .tools import read_file, write_file, run_git_command, show_diff, run_python_file, run_shell_command, list_files
from .code_editing_helpers import (
    collect_syntax_errors, parse_compiler_errors, get_focused_context,
    detect_indent_char, extract_function, apply_block_to_lines,
    is_oversized_block, find_first_mismatch, whole_function_replace,
    strip_markdown, parse_multiple_blocks, build_syntax_error_note,
    build_runtime_error_note, build_post_error_note,
    validate_code_structure, build_structural_issue_note,
    FUNC_PATTERN, _C_FUNC_PATTERN
)
from .code_context_builders import build_all_contexts
from .session_memory import SessionMemory
from .orchestration import plan_task, format_plan_for_prompt, critique_code, build_critic_feedback_note

logger = logging.getLogger("code_agent")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Shared session memory instance (persists across requests within a server lifetime)
_session_memory: Optional[SessionMemory] = None

def get_session_memory() -> SessionMemory:
    """Lazy-init singleton session memory."""
    global _session_memory
    if _session_memory is None:
        _session_memory = SessionMemory()
    return _session_memory


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
            llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)
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
        MAX_BLOCK_RATIO = 0.6

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

        # extracted repeated `combined_context + extra_syntax` into one variable
        context_block = combined_context + extra_syntax

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

        llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)

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
                logger.info("[AGENT] Planner produced %d steps, approach: %s", len(plan.get('steps', [])), plan.get('approach', '')[:100])
            except Exception as e:
                logger.debug("[AGENT] Planner failed, continuing without plan: %s", e)

        # --- Two-stage: analysis then edit ---
        logger.info("Waiting for LLM to analyze the issue...")
        try:
            if is_implement:
                plan_section = f"\nPLAN:\n{plan_context}\n" if plan_context else ""
                analysis_prompt = (
                    f"You are solving a programming task. Read the requirement carefully and plan the solution.\n\n"
                    f"{lang_directive}"
                    f"TASK: {instruction}\n\n"
                    f"{plan_section}"
                    f"{original_header}```{lang_fence}\n{original_snippet}\n```\n\n"
                    "Describe step-by-step what algorithm or logic you will implement to satisfy the requirement. "
                    "Be specific about data structures, loop structure, and edge cases.\n\n"
                    f"{context_block}"
                )
            else:
                analysis_prompt = (
                    f"Look at this code and describe what needs to change to fix: {instruction}\n\n"
                    f"{lang_directive}"
                    f"{original_header}```{lang_fence}\n{original_snippet}\n```\n\n"
                    f"{context_block}"
                )
            analysis = llm.invoke(analysis_prompt).strip()

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
        blocks = parse_multiple_blocks(raw_output)
        if not blocks:
            raw_output = strip_markdown(raw_output)
            blocks = parse_multiple_blocks(raw_output)

        for i, (s, r) in enumerate(blocks):
            logger.debug("Block %s SEARCH:\n%s", i + 1, s)
            logger.debug("Block %s REPLACE:\n%s", i + 1, r)

        # --- Auto-retry if no valid blocks found ---
        retry_count = 0
        max_retries = 2
        while not blocks and retry_count < max_retries:
            logger.warning("Bad format from LLM: SEARCH/REPLACE not found. Retry %s/%s", retry_count + 1, max_retries)
            followup = (
                f"You must ONLY output a SEARCH/REPLACE block in this exact format (no explanation):\n"
                f"{FORMAT_BLOCK}\n\n"
                "The SEARCH block must include the exact surrounding lines from the target file "
                "so it matches character-for-character.\n"
                "Previous LLM output:\n" + raw_output[:8000] + "\n\n"
                "Re-output the corrected SEARCH/REPLACE block now."
            )
            try:
                raw_output = llm.invoke(followup).strip()
            except Exception as e:
                logger.error("LLM re-invocation failed: %s", e)
                logger.debug(traceback.format_exc())
                break
            blocks = parse_multiple_blocks(raw_output)
            if not blocks:
                raw_output = strip_markdown(raw_output)
                blocks = parse_multiple_blocks(raw_output)
            retry_count += 1

        if not blocks:
            logger.error("Bad format from LLM after retries: SEARCH/REPLACE not found.")
            return file_source

        # --- Apply blocks sequentially ---
        source_lines = file_lines[:]  # copy pre-computed list instead of re-splitting

        for idx, (search_text, replace_text) in enumerate(blocks):
            if is_oversized_block(search_text, source_lines, idx + 1, prefix="Block"):
                continue

            match_start_idx, replace_lines, matched_len = apply_block_to_lines(source_lines, search_text, replace_text)
            if match_start_idx == -1:
                logger.error("Block %s: Unable to locate SEARCH block. Skipping.", idx + 1)
                continue

            logger.debug("Applying block %s at line %s replacing %s lines", idx + 1, match_start_idx, matched_len)
            source_lines = (
                source_lines[:match_start_idx]
                + replace_lines
                + source_lines[match_start_idx + matched_len:]
            )

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
                repair_blocks = parse_multiple_blocks(raw_output)
                if not repair_blocks:
                    raw_output = strip_markdown(raw_output)
                    repair_blocks = parse_multiple_blocks(raw_output)
                for search_text, replace_text in repair_blocks:
                    if is_oversized_block(search_text, source_lines, None, prefix="Self-repair Block"):
                        continue
                    match_start_idx, replace_lines, matched_len = apply_block_to_lines(source_lines, search_text, replace_text)
                    if match_start_idx != -1:
                        logger.debug("Applying self-repair block at %s replacing %s lines", match_start_idx, matched_len)
                        source_lines = (
                            source_lines[:match_start_idx]
                            + replace_lines
                            + source_lines[match_start_idx + matched_len:]
                        )
                    else:
                        logger.warning("Self-repair block could not be located. Skipping.")
            except Exception as e:
                logger.error("Self-repair LLM call failed: %s", e)
                logger.debug(traceback.format_exc())

        new_source = "\n".join(source_lines)

        # --- Structural validation (all languages: brace balance, unreachable code, indentation) ---
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
                repair_blocks = parse_multiple_blocks(raw_repair)
                if not repair_blocks:
                    repair_blocks = parse_multiple_blocks(strip_markdown(raw_repair))
                for search_text, replace_text in repair_blocks:
                    if is_oversized_block(search_text, source_lines, None, prefix="Structural-repair"):
                        continue
                    match_start_idx, replace_lines, matched_len = apply_block_to_lines(
                        source_lines, search_text, replace_text
                    )
                    if match_start_idx != -1:
                        logger.info("Structural repair block applied at line %s", match_start_idx)
                        source_lines = (
                            source_lines[:match_start_idx]
                            + replace_lines
                            + source_lines[match_start_idx + matched_len:]
                        )
                    else:
                        logger.warning("Structural repair block could not be located. Skipping.")
                new_source = "\n".join(source_lines)
            except Exception as e:
                logger.warning("Structural repair LLM call failed: %s", e)

        if new_source == file_source and blocks:
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
                retry_blocks = parse_multiple_blocks(raw_output)
                if not retry_blocks:
                    raw_output = strip_markdown(raw_output)
                    retry_blocks = parse_multiple_blocks(raw_output)
                if retry_blocks:
                    source_lines = file_lines[:]  # copy pre-computed list
                    for idx, (search_text, replace_text) in enumerate(retry_blocks):
                        if is_oversized_block(search_text, source_lines, idx + 1, prefix="Retry Block"):
                            continue
                        match_start_idx, replace_lines, matched_len = apply_block_to_lines(
                            source_lines, search_text, replace_text
                        )
                        if match_start_idx == -1:
                            logger.error("Retry Block %s: Unable to locate SEARCH block. Skipping.", idx + 1)
                            continue
                        logger.debug("Applying retry block %s at %s replacing %s lines", idx + 1, match_start_idx, matched_len)
                        source_lines = (
                            source_lines[:match_start_idx]
                            + replace_lines
                            + source_lines[match_start_idx + matched_len:]
                        )
                    new_source = "\n".join(source_lines)
            except Exception as e:
                logger.error("Verbatim retry LLM invocation failed: %s", e)
                logger.debug(traceback.format_exc())

        # --- Whole-function rewrite fallback ---
        if new_source == file_source and blocks:
            logger.warning("Verbatim retry also failed — falling back to whole-function rewrite")
            if is_implement:
                rewrite_prompt = (
                    f"Implement a complete solution for the following task inside the function stub below.\n\n"
                    f"{lang_directive}"
                    f"TASK: {instruction}\n\n"
                    f"FUNCTION STUB:\n```{lang_fence}\n{verbatim}\n```\n\n"
                    "Rules:\n"
                    "1. Output ONLY the complete rewritten function — no explanation, no markdown fence.\n"
                    "2. Write a real, working algorithm. Do NOT leave the body empty or return a placeholder.\n"
                    "3. Keep exact indentation of the original.\n"
                    "4. Output ONLY this function — do NOT include other functions from the file.\n"
                    f"{context_block}"
                )
            else:
                rewrite_prompt = (
                    f"Rewrite the following code to apply this change: {instruction}\n\n"
                    f"{lang_directive}"
                    f"CURRENT CODE:\n```{lang_fence}\n{verbatim}\n```\n\n"
                    "Rules:\n"
                    "1. Output ONLY the rewritten code — no explanation, no markdown fence.\n"
                    "2. Preserve ALL existing logic except what the instruction explicitly changes.\n"
                    "3. Keep exact indentation of the original.\n"
                    "4. Do NOT remove, rewrite, or omit any code outside the targeted function.\n"
                    f"{context_block}"
                )
            try:
                rewritten = llm.invoke(rewrite_prompt).strip()
                rewritten = strip_markdown(rewritten) if "```" in rewritten else rewritten
                if rewritten.strip():
                    # Safety check: reject if content is destroyed (rewritten loses >30% of lines on large files)
                    original_span_lines = file_lines[func_start:func_end]
                    rewritten_lines = rewritten.strip().split('\n')
                    line_ratio = len(rewritten_lines) / max(1, len(original_span_lines))

                    # Guard: if the span covers the entire file and has multiple functions,
                    # the rewrite would destroy unrelated code — refuse and keep original.
                    is_full_file = func_start == 0 and func_end >= len(file_lines)
                    func_count = sum(1 for l in file_lines if FUNC_PATTERN.match(l) or _C_FUNC_PATTERN.match(l))
                    if is_full_file and func_count > 1:
                        logger.warning(
                            "Whole-function rewrite rejected — span covers the entire file "
                            "(%d functions detected). Refusing to destroy unrelated code.",
                            func_count
                        )
                    elif line_ratio < 0.70 and len(original_span_lines) > 30:
                        logger.warning(
                            "Whole-function rewrite rejected — rewritten has only %.0f%% of original lines "
                            "(%d vs %d), likely data loss. Refusing to apply.",
                            line_ratio * 100, len(rewritten_lines), len(original_span_lines)
                        )
                    else:
                        new_source = whole_function_replace(file_source, rewritten, func_start, func_end)
                        logger.info("Whole-function rewrite applied (lines %s–%s)", func_start, func_end)
                else:
                    logger.error("Whole-function rewrite returned empty output. Giving up.")
            except Exception as e:
                logger.error("Whole-function rewrite LLM call failed: %s", e)
                logger.debug(traceback.format_exc())

        # --- Self-correction ---
        new_source = self._self_correct_output(
            llm, new_source, file_source, is_implement, instruction, lang_fence, FORMAT_BLOCK
        )

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
                "Does this code correctly and completely satisfy the requirement?\n"
                "Respond with EXACTLY one of these two formats:\n"
                "  CORRECT: <brief reason>\n"
                "  ISSUE: <specific description of the bug or missing logic>\n\n"
                "Only output one line. No other text."
            )
            grade_result = llm.invoke(grade_prompt).strip()
            logger.info("[AGENT] Self-grade: %s", grade_result[:200])

            if grade_result.upper().startswith("ISSUE:"):
                issue = grade_result[len("ISSUE:"):].strip()
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
                blocks = parse_multiple_blocks(raw)
                if not blocks:
                    blocks = parse_multiple_blocks(strip_markdown(raw))
                if blocks:
                    lines = new_source.split('\n')
                    for s_text, r_text in blocks:
                        m_idx, r_lines, m_len = apply_block_to_lines(lines, s_text, r_text)
                        if m_idx != -1:
                            lines = lines[:m_idx] + r_lines + lines[m_idx + m_len:]
                            logger.info("[AGENT] Self-correction block applied")
                        else:
                            logger.warning("[AGENT] Self-correction block could not be located.")
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
                        fix_blocks = parse_multiple_blocks(raw_fix)
                        if not fix_blocks:
                            fix_blocks = parse_multiple_blocks(strip_markdown(raw_fix))
                        if fix_blocks:
                            lines = new_source.split('\n')
                            for s_text, r_text in fix_blocks:
                                m_idx, r_lines, m_len = apply_block_to_lines(lines, s_text, r_text)
                                if m_idx != -1:
                                    lines = lines[:m_idx] + r_lines + lines[m_idx + m_len:]
                                    logger.info("[AGENT] Critic fix block applied")
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
                failures_block += f"  input={r['input']}  →  expected={r['expected']},  got={got}\n"

        failed_block = ""
        if failed_approaches:
            failed_block = "APPROACHES ALREADY TRIED AND FAILED (do NOT reuse):\n"
            for fa in failed_approaches:
                failed_block += f"  - {fa}\n"
            failed_block += "\n"

        prompt = (
            f"The current implementation of the function is FUNDAMENTALLY WRONG. "
            f"Small fixes have been attempted multiple times and all failed with the same errors.\n\n"
            f"PROBLEM: {instruction}\n\n"
            f"CURRENT BROKEN CODE:\n```{lang_fence}\n{current_source}\n```\n\n"
            f"FAILING TESTS:\n{failures_block}\n"
            f"{failed_block}"
            f"You MUST solve this using a COMPLETELY DIFFERENT algorithm or approach.\n"
            f"Think step by step:\n"
            f"1. Re-read the problem statement carefully — what is actually being asked?\n"
            f"2. Why does the current approach fundamentally fail?\n"
            f"3. What alternative algorithm solves this correctly?\n\n"
            f"Write the COMPLETE rewritten function in {lang_name}.\n"
            f"Output ONLY the function code — no explanation, no markdown fences.\n"
        )

        try:
            # Use slightly higher temperature to encourage creative divergence
            pivot_llm = OllamaLLM(model=LLM_MODEL, temperature=0.3)
            rewritten = pivot_llm.invoke(prompt).strip()
            rewritten = strip_markdown(rewritten) if "```" in rewritten else rewritten

            if not rewritten.strip():
                logger.warning("[STRATEGY_PIVOT] LLM returned empty output")
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
          - Self-reasoning: LLM traces through failing tests to identify root cause
          - Degradation detector: stops if pass count drops between iterations
          - Temperature escalation: bumps temperature when stuck at same pass count
          - Explanation + citations: returns what the agent did and which RAG docs it used

        Returns dict with final_source, test_results, all_passed, attempts, diff,
               explanation, citations.
        """
        from .test_runner import run_tests, build_test_failure_note

        llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)
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

            # --- Identical-error loop detector + strategy pivot ---
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

                if stagnant_count >= 2:
                    logger.warning(
                        "[FIX_WITH_TESTS] Identical errors repeating after pivot — stopping"
                    )
                    explanation_parts.append(
                        "Same errors repeating after strategy pivot. Stopped to avoid infinite loop."
                    )
                    source = best_source
                    all_results = run_tests(source, path, test_cases, llm)
                    break

                # --- Strategy Pivot: rewrite from scratch with a different approach ---
                logger.info(
                    "[FIX_WITH_TESTS] Identical errors — triggering strategy pivot (attempt %d)",
                    attempt
                )
                explanation_parts.append(
                    f"Attempt {attempt}: Same errors detected — pivoting to a different algorithm."
                )
                pivot_source = self._strategy_pivot(
                    llm, instruction, source, ext, test_cases, results, failed_approaches
                )
                if pivot_source and pivot_source != source:
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

            # --- Temperature escalation ---
            # If pass count is stagnant (same as last iteration), bump temperature
            if attempt > 0 and pass_count == best_pass_count and current_temperature < 0.4:
                current_temperature = min(current_temperature + 0.15, 0.4)
                llm = OllamaLLM(model=LLM_MODEL, temperature=current_temperature)
                logger.info(
                    "[FIX_WITH_TESTS] Pass count stagnant — escalating temperature to %.2f",
                    current_temperature
                )
                explanation_parts.append(
                    f"Escalated temperature to {current_temperature:.2f} for creative problem-solving."
                )

            # --- Self-reasoning: trace through test to find root cause ---
            failure_note = build_test_failure_note(
                results, instruction, source=source, llm=llm,
                attempt=attempt, failed_approaches=failed_approaches
            )
            augmented_instruction = f"{instruction}\n\n{failure_note}"

            # Include structural issues in the instruction so LLM can fix them
            if structural_issues:
                structural_note = build_structural_issue_note(structural_issues, source.split('\n'))
                augmented_instruction += f"\n\n{structural_note}"

            attempts += 1

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
                source = edit_result.get("new_source", source)
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

    def list_db_code(self, limit: int = 20):
        docs = get_code_documents(limit=limit)
        return [d["text"] for d in docs]

    def test_file(self, path: str) -> str:
        return run_python_file(path)

    def run_shell_command(self, command: str) -> str:
        return run_shell_command(command)

    def list_files(self) -> list[str]:
        return list_files(self.repo_path)