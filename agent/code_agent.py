import subprocess
import re
import logging
import sys
import traceback
from datetime import datetime
import os

from langchain_ollama import OllamaLLM
from typing import List, Dict, Optional

from brain.config import LLM_MODEL
from brain.fast_search import fast_topic_search
from db.db_reader import get_code_documents
from .tools import read_file, write_file, run_git_command, show_diff, run_python_file, run_shell_command, list_files
from .code_editing_helpers import (
    collect_syntax_errors, parse_compiler_errors, get_focused_context,
    detect_indent_char, extract_function, apply_block_to_lines,
    is_oversized_block, find_first_mismatch, whole_function_replace,
    strip_markdown, parse_multiple_blocks, build_syntax_error_note,
    build_runtime_error_note, build_post_error_note
)
from .code_context_builders import build_all_contexts

#recognize function/class definitions in multiple languages for better context extraction
FUNC_PATTERN = re.compile(r"\s*(def|class|func)\s+\w+")

logger = logging.getLogger("code_agent")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class CodeAgent:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.edit_log: dict[str, list] = {}

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

    def edit_code(
        self,
        path: str,
        instruction: str,
        dry_run: bool = True,
        use_rag: bool = False,
        session_chat_history: Optional[List[Dict[str, str]]] = None,
        rerank_method: str = "cross_encoder",
        max_chunks: Optional[int] = None
    ) -> str:
        """Edit code in a file based on instruction using LLM.
        
        Args:
            path: File path to edit
            instruction: Editing instruction
            dry_run: If True, only show diff without writing
            use_rag: If True, include RAG context from code search
            session_chat_history: Optional conversation history for context
            rerank_method: Reranking method for RAG - "cross_encoder", "keyword", or "none"
        """
        file_source = read_file(path)
        ext = os.path.splitext(path)[1].lower()
        is_python = ext == ".py"
        lang_fence = "python" if is_python else (ext.lstrip('.') if ext else "text")

        # Resolve "auto" rerank method based on instruction complexity
        rerank_method = self._resolve_rerank_method(instruction, rerank_method)

        file_lines = file_source.split('\n')
        MAX_BLOCK_RATIO = 0.6

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
        runtime_error_note, rag_context, history_context, edit_history_text = build_all_contexts(
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
            max_chunks=max_chunks
        )

        # Only build runtime errors if no syntax errors were found
        if syntax_errors:
            runtime_error_note = ""

        combined_context = history_context + rag_context + edit_history_text

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
            if focused and len(focused.splitlines()) < 60:
                original_snippet = focused
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

        # --- Two-stage: analysis then edit ---
        logger.info("Waiting for LLM to analyze the issue...")
        try:
            analysis_prompt = (
                f"Look at this code and describe what needs to change to fix: {instruction}\n\n"
                f"{original_header}```{lang_fence}\n{original_snippet}\n```\n\n"
                f"{context_block}" 
            )
            analysis = llm.invoke(analysis_prompt).strip()

            logger.info("Analysis received; requesting SEARCH/REPLACE from LLM...")
            
            # When fixing multiple functions, instruct LLM to output multiple blocks
            if multi_func_hint:
                # Build a descriptive label list from whatever hint form was detected
                func_matches = re.findall(r'\b(\w+)\(\)', instruction)
                quoted_items = re.findall(r'"([^"]{5,})"', instruction)
                func_name_mentions = re.findall(r'\b(?:the\s+)?(\w+)\s+(?:function|method)\b', instruction, re.IGNORECASE)

                if quoted_items:
                    task_labels = quoted_items
                elif func_matches:
                    task_labels = [f"{n}()" for n in func_matches]
                elif func_name_mentions:
                    task_labels = list(dict.fromkeys(func_name_mentions))  # deduplicated, preserving order
                else:
                    task_labels = ["each fix listed in the instruction"]

                task_count = len(task_labels)
                task_list = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(task_labels))

                block_instruction = (
                    f"Now produce ONE SEARCH/REPLACE block for EACH of the {task_count} tasks:\n{task_list}\n\n"
                    f"You MUST output {task_count} separate blocks — one COMPLETE block per task.\n"
                    "For EACH task, output BOTH the SEARCH and REPLACE sections back-to-back with no text between them.\n\n"
                    "TEMPLATE TO REPEAT FOR EACH TASK:\n"
                    f"```{lang_fence}\n"
                    "<<<<<<< SEARCH\n"
                    "(exact old code for this fix)\n"
                    "=======\n"
                    "(exact new code for this fix)\n"
                    ">>>>>>> REPLACE\n"
                    "```\n\n"
                    f"Output all {task_count} blocks sequentially with no commentary between them.\n"
                )
            else:
                block_instruction = "Now produce a SEARCH/REPLACE block. Rules:\n"
            
            edit_prompt = (
                f"Given this analysis:\n{analysis}\n\n"
                f"{block_instruction}"
                "1. SEARCH must EXACTLY match existing code character-for-character including indentation.\n"
                "2. Include a few lines of surrounding context so SEARCH is unique in the file.\n"
                "3. When ADDING code, keep the surrounding lines in REPLACE so they are not deleted.\n"
                "4. Output ONLY the blocks — no explanatory text, no comments.\n\n"
                "CRITICAL RULE: Your SEARCH blocks must contain ONLY the specific lines being changed (max 20-30 lines each). "
                "Output all blocks back-to-back with no commentary between them.\n\n"
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
            rewrite_prompt = (
                f"Rewrite the following code to apply this change: {instruction}\n\n"
                f"CURRENT CODE:\n```{lang_fence}\n{verbatim}\n```\n\n"
                "Rules:\n"
                "1. Output ONLY the rewritten code — no explanation, no markdown fence.\n"
                "2. Preserve ALL existing logic except what the instruction explicitly changes.\n"
                "3. Keep exact indentation of the original.\n"
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
                    if line_ratio < 0.70 and len(original_span_lines) > 30:
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

        diff = show_diff(file_source, new_source)
        logger.info("--- DIFF PREVIEW ---")
        logger.info('\n%s', diff or "No changes detected.")
        logger.info("--------------------")

        if dry_run:
            logger.info("Dry run mode enabled; no file written.")
            return {"new_source": new_source, "diff": diff, "changed": new_source != file_source}

        try:
            write_file(path, new_source)
            try:
                run_git_command(["diff", "--", path])
            except Exception as e:
                logger.debug("git diff failed: %s", e)
        except Exception as e:
            logger.error("Failed to write file %s: %s", path, e)
            logger.debug(traceback.format_exc())
            return file_source

        if ext == ".go":
            try:
                subprocess.run(["gofmt", "-w", path], timeout=5, check=True)
                logger.info("gofmt applied to %s", path)
            except Exception as e:
                logger.debug("gofmt failed: %s", e)

        try:
            self.edit_log.setdefault(path, []).append({
                "instruction": instruction,
                "diff": diff,
                "timestamp": datetime.now().isoformat()
            })
        except Exception:
            logger.debug("Failed to append to edit_log for %s", path)

        if session_chat_history is not None:
            session_chat_history.append({"role": "User", "content": instruction})
            session_chat_history.append({"role": "Assistant", "content": f"Edited {path}: {instruction}"})

        logger.info("Wrote file %s", path)
        return new_source

    def list_db_code(self, limit: int = 20):
        docs = get_code_documents(limit=limit)
        return [d["text"] for d in docs]

    def test_file(self, path: str) -> str:
        return run_python_file(path)

    def run_shell_command(self, command: str) -> str:
        return run_shell_command(command)

    def list_files(self) -> list[str]:
        return list_files(self.repo_path)