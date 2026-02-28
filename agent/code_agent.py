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

#recognize function/class definitions in multiple languages for better context extraction
FUNC_PATTERN = re.compile(r"\s*(def|class|func)\s+\w+")

LANG_CHECK_CMD = {
    ".go":  ["go", "build"],
    ".rs":  ["rustc", "--edition", "2021"],
    ".cpp": ["g++", "-fsyntax-only"],
    ".c":   ["gcc", "-fsyntax-only"],
    ".ts":  ["tsc", "--noEmit"],
}

logger = logging.getLogger("code_agent")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class CodeAgent:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.edit_log: dict[str, list] = {}

    def edit_code(
        self,
        path: str,
        instruction: str,
        dry_run: bool = True,
        use_rag: bool = False,
        session_chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:

        file_source = read_file(path)
        ext = os.path.splitext(path)[1].lower()
        is_python = ext == ".py"
        lang_fence = "python" if is_python else (ext.lstrip('.') if ext else "text")
        # removed dead `indent_char = '\t' if '\t' in file_source else ' '`
        #          detect_indent_char() inside apply_block_to_lines already handles this.

        # pre-compute once; all `file_source.split('\n')` below use this.
        file_lines = file_source.split('\n')

        blocks = []
        verbatim = ""
        MAX_BLOCK_RATIO = 0.6

        # --- Collect ALL syntax errors ---
        def collect_syntax_errors(source: str, path: str) -> list:
            errors = []
            temp = source
            lines = temp.split('\n')
            seen_lines = set()
            for _ in range(10):
                try:
                    compile(temp, path, 'exec')
                    break
                except SyntaxError as se:
                    se_line = getattr(se, 'lineno', None)
                    if se_line in seen_lines:
                        break
                    seen_lines.add(se_line)
                    errors.append({
                        'msg': se.msg,
                        'lineno': se_line,
                        'offset': getattr(se, 'offset', None),
                        'text': (getattr(se, 'text', '') or '').strip()
                    })
                    if se_line and se_line <= len(lines):
                        lines[se_line - 1] = '# SYNTAX_ERROR_PLACEHOLDER'
                        temp = '\n'.join(lines)
            return errors

        def parse_compiler_errors(stderr: str) -> list:
            """Works for go, rustc, gcc, clang, tsc — all use file:line:col: message."""
            errors = []
            for match in re.finditer(r"^(.+?):(\d+):(\d+):\s+(.+)$", stderr, re.MULTILINE):
                errors.append({
                    'file': match.group(1),
                    'lineno': int(match.group(2)),
                    'col': int(match.group(3)),
                    'msg': match.group(4)
                })
            return errors

        # --- Focused context helper ---
        def get_focused_context(source: str, error_lineno: int, window: int = 20):
            lines = source.split('\n')
            start = max(0, error_lineno - window)
            end = min(len(lines), error_lineno + window)
            raw = '\n'.join(lines[start:end])
            numbered = '\n'.join(f"{start + i + 1}: {l}" for i, l in enumerate(lines[start:end]))
            return raw, numbered

        def detect_indent_char(source: str) -> str:
            for line in source.split('\n'):
                if line and line[0] == '\t':
                    return '\t'
            return ' '

        # --- Extract enclosing function/class ---
        def extract_function(source: str, instruction: str, hint_blocks=None) -> tuple[str, int, int]:
            lines = source.split('\n')
            target_idx = None

            if hint_blocks:
                try:
                    first_search = hint_blocks[0][0]
                    for line in first_search.split('\n'):
                        if not line.strip():
                            continue
                        pat = line.strip()
                        for i, l in enumerate(lines):
                            if l.strip() == pat:
                                target_idx = i
                                break
                        if target_idx is not None:
                            break
                except Exception:
                    target_idx = None

            if target_idx is None:
                m = re.search(
                    r"\b([\w_]+)\s+function\b"
                    r"|function\s+([\w_]+)\b"
                    r"|def\s+([\w_]+)\b"
                    r"|func\s+([\w_]+)\b"
                    r"|\b([\w_]+)\s*\(\)",
                    instruction, re.I
                )
                name = next((g for g in m.groups() if g), None) if m else None
                if name:
                    for i, l in enumerate(lines):
                        if re.match(rf"\s*(def|func)\s+{re.escape(name)}\b", l, re.I):
                            target_idx = i
                            break

            if target_idx is None:
                start_idx = 0
            else:
                start_idx = target_idx
                while start_idx > 0 and not FUNC_PATTERN.match(lines[start_idx]):
                    start_idx -= 1
                if not FUNC_PATTERN.match(lines[start_idx]) and start_idx != 0:
                    for i in range(target_idx, -1, -1):
                        if FUNC_PATTERN.match(lines[i]):
                            start_idx = i
                            break

            indent = (
                len(lines[start_idx]) - len(lines[start_idx].lstrip())
                if start_idx < len(lines) and FUNC_PATTERN.match(lines[start_idx])
                else 0
            )

            end_idx = start_idx + 1
            while end_idx < len(lines):
                line = lines[end_idx]
                if FUNC_PATTERN.match(line) and (len(line) - len(line.lstrip())) <= indent:
                    break
                end_idx += 1

            if target_idx is None and start_idx == 0:
                return "", 0, len(lines)
            return '\n'.join(lines[start_idx:end_idx]), start_idx, end_idx

        # removed `indent_char` param and dead tab branch.
        # `get_replace` already returns early for tab files and reindent_block is now space-only.
        # reached for space-indented (Python) files.
        def reindent_block(replace_lines: list, file_indent: int, llm_base_indent: int) -> list:
            result = []
            for line in replace_lines:
                if not line.strip():
                    result.append("")
                elif llm_base_indent == 0:
                    llm_spaces = len(line) - len(line.lstrip())
                    result.append(' ' * file_indent + ' ' * llm_spaces + line.lstrip())
                else:
                    delta = file_indent - llm_base_indent
                    current = len(line) - len(line.lstrip())
                    new_indent = max(0, current + delta)
                    result.append(' ' * new_indent + line.lstrip())
            return result

        def apply_block_to_lines(source_lines, search_text, replace_text):
            search_lines = search_text.split('\n')
            while search_lines and not search_lines[0].strip():
                search_lines.pop(0)
            while search_lines and not search_lines[-1].strip():
                search_lines.pop()

            replace_lines = replace_text.strip('\n').split('\n')
            n = len(search_lines)
            if n == 0:
                return -1, None, 0

            indent_char = detect_indent_char('\n'.join(source_lines))

            def get_replace(match_idx):
                # Tab-indented files: return as-is; LLM output is already correct.
                if indent_char == '\t':
                    return replace_lines
                file_indent = len(source_lines[match_idx]) - len(source_lines[match_idx].lstrip())
                first_non_empty = next((l for l in replace_lines if l.strip()), "")
                llm_base_indent = len(first_non_empty) - len(first_non_empty.lstrip()) if first_non_empty else 0
                # CLEANED: no indent_char arg — reindent_block is space-only now
                return reindent_block(replace_lines, file_indent, llm_base_indent)

            # 1. Exact match (trailing whitespace normalized)
            for i in range(len(source_lines) - n + 1):
                if all(source_lines[i+j].rstrip() == search_lines[j].rstrip() for j in range(n)):
                    return i, get_replace(i), n

            # 2. Whitespace-insensitive match
            for i in range(len(source_lines) - n + 1):
                if all(source_lines[i+j].strip() == search_lines[j].strip() for j in range(n)):
                    return i, get_replace(i), n

            # 3. Anchor + sliding window tolerant match
            anchors = [s.strip() for s in search_lines if s.strip()]
            if anchors:
                first_anchor = anchors[0]
                best_idx, best_score = -1, 0
                for i, line in enumerate(source_lines):
                    if line.strip() == first_anchor:
                        score = sum(
                            1 for j in range(min(n, len(source_lines) - i))
                            if source_lines[i+j].strip() == search_lines[j].strip()
                        )
                        if score > best_score:
                            best_score, best_idx = score, i
                if best_score >= max(2, n // 2):
                    matched = 0
                    for j in range(min(n, len(source_lines) - best_idx)):
                        if source_lines[best_idx + j].strip() == search_lines[j].strip():
                            matched += 1
                        else:
                            break
                    return best_idx, get_replace(best_idx), max(1, matched)

            # 4. Fuzzy difflib fallback
            try:
                from difflib import SequenceMatcher
                source_str = '\n'.join(source_lines)
                search_str = '\n'.join(s.rstrip() for s in search_lines)
                sm = SequenceMatcher(None, source_str, search_str, autojunk=False)
                match = sm.find_longest_match(0, len(source_str), 0, len(search_str))
                if match.size > 40:
                    start_line = source_str[:match.a].count('\n')
                    matched_substr = source_str[match.a: match.a + match.size]
                    matched_lines = matched_substr.count('\n') + 1
                    return start_line, get_replace(start_line), matched_lines
            except Exception:
                logger.debug("Fuzzy matching failed: %s", traceback.format_exc())

            return -1, None, 0

        def is_oversized_block(search_text: str, source_lines: list, idx: Optional[int] = None, prefix: str = "Block") -> bool:
            try:
                search_line_count = len(search_text.strip('\n').split('\n'))
                file_len = len(source_lines)
                if search_line_count > file_len * MAX_BLOCK_RATIO:
                    if idx is not None:
                        logger.warning("%s %s rejected — SEARCH spans %s/%s lines. Skipping.", prefix, idx, search_line_count, file_len)
                    else:
                        logger.warning("%s rejected — SEARCH spans %s/%s lines. Skipping.", prefix, search_line_count, file_len)
                    return True
            except Exception:
                logger.debug("Oversize check failed: %s", traceback.format_exc())
            return False

        def find_first_mismatch(source_lines: list, search_lines: list):
            for j, sl in enumerate(search_lines):
                if sl.strip() and not any(sl.strip() == src.strip() for src in source_lines):
                    return j, sl
            return -1, None

        # --- Build syntax/runtime error notes ---
        syntax_errors = collect_syntax_errors(file_source, path) if is_python else []
        syntax_error_note = ""

        if syntax_errors:
            broken_lines = file_lines  # reuse file_lines
            syntax_error_note = "SYNTAX_ERRORS_DETECTED (fix ALL of them):\n\n"
            for err in syntax_errors:
                se_line = err['lineno']
                start = max(0, se_line - 3)
                end = min(len(broken_lines), se_line + 2)
                snippet = '\n'.join(f"{start+i+1}: {l}" for i, l in enumerate(broken_lines[start:end]))
                syntax_error_note += (
                    f"ERROR: {err['msg']} at Line {se_line}, Col {err['offset']}\n"
                    f"CONTEXT (line numbers are reference only — never copy them into SEARCH/REPLACE):\n"
                    f"```\n{snippet}\n```\n"
                    f"RAW LINES FOR SEARCH (no line numbers):\n"
                    f"```\n{chr(10).join(broken_lines[start:end])}\n```\n\n"
                )
            syntax_error_note += (
                "Output ONE SEARCH/REPLACE block per error, in order top to bottom.\n"
                "Each SEARCH block must match the exact broken lines shown above."
            )
            logger.info("Found %s syntax error(s) in %s", len(syntax_errors), path)
        

        runtime_error_note = ""
        if not syntax_errors:
            # single clean conditional instead of double-assignment
            if is_python:
                cmd = [sys.executable, path]
            else:
                base = LANG_CHECK_CMD.get(ext)
                cmd = base + [path] if base else None

            if cmd:
                try:
                    run_result = subprocess.run(
                        cmd, capture_output=True, text=True,
                        timeout=10, stdin=subprocess.DEVNULL
                    )
                    if run_result.returncode != 0 and run_result.stderr:
                        compiler_errors = parse_compiler_errors(run_result.stderr)
                        if compiler_errors:
                            runtime_error_note = "RUNTIME_ERRORS_DETECTED (fix ALL of them):\n\n"
                            broken_lines = file_lines 
                            for err in compiler_errors:
                                lineno = err['lineno']
                                start = max(0, lineno - 3)
                                end = min(len(broken_lines), lineno + 2)
                                snippet = '\n'.join(
                                    f"{start+i+1}: {l}"
                                    for i, l in enumerate(broken_lines[start:end])
                                )
                                runtime_error_note += (
                                    f"ERROR: {err['msg']} at Line {lineno}, Col {err['col']}\n"
                                    f"CONTEXT (line numbers are reference only — never copy them into SEARCH/REPLACE):\n"
                                    f"```\n{snippet}\n```\n"
                                    f"RAW LINES FOR SEARCH (no line numbers):\n"
                                    f"```\n{chr(10).join(broken_lines[start:end])}\n```\n\n"
                                )
                            runtime_error_note += "Output ONE SEARCH/REPLACE block per error, top to bottom."
                        else:
                            runtime_error_note = (
                                "RUNTIME_ERRORS_DETECTED:\n"
                                f"```\n{run_result.stderr[:2000]}\n```\n\n"
                                "Output ONE SEARCH/REPLACE block per error fix."
                            )
                        logger.info("Compiler errors detected in %s", path)
                except subprocess.TimeoutExpired:
                    runtime_error_note = "RUNTIME_WARNING: Build timed out — check for infinite loops.\n"
                    logger.warning("File %s timed out during check", path)
                except Exception as e:
                    logger.debug("Compiler check failed: %s", e)

        # --- RAG Context ---
        rag_context = ""
        if use_rag:
            print("\n[FAST_SEARCH] BM25 for code context...")
            try:
                results = fast_topic_search(instruction)
                if results:
                    rag_context = "REFERENCE DOCS (Guide your edit):\n"
                    for i, doc in enumerate(results[:3]):
                        score = doc.metadata.get('bm25_score', 'N/A')
                        doc_source = doc.metadata.get('source', 'Unknown')
                        score_str = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)
                        rag_context += f"[{i+1}] {doc_source} (BM25:{score_str})\n{doc.page_content[:300]}...\n\n"
                    print(f"Found {len(results)} chunks!")
                else:
                    rag_context = "No exact matches—use general principles.\n"
            except Exception as e:
                print(f"RAG error: {e}")

        # --- Conversation History Context ---
        history_context = ""
        if session_chat_history:
            recent = session_chat_history[-6:]
            history_context = "\nCONVERSATION CONTEXT (follow-up edit):\n"
            for msg in recent:
                role = "USER: " if msg.get("role", "").lower() in ["user", "human"] else "AGENT: "
                content = msg.get('content', '')
                content = (content[:200] + "...") if len(content) > 200 else content
                history_context += f"{role}{content}\n"

        edit_history_text = ""
        try:
            recent_edits = self.edit_log.get(path, [])[-3:]
            if recent_edits:
                edit_history_text = "RECENT EDITS (most recent last):\n"
                for e in recent_edits:
                    edit_history_text += f"- {e.get('timestamp','')}: {e.get('instruction','')}\nDiff:\n{e.get('diff','')[:1000]}\n\n"
        except Exception:
            edit_history_text = ""

        combined_context = history_context + rag_context + edit_history_text

        # detect if the instruction references many functions (e.g., "Fix A, B, C")
        import re as _re
        multi_func_hint = len(_re.findall(r"\b\w+\(\)", instruction)) >= 3

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

        #--- Determine original snippet for LLM prompt ---
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
                
        # --- Choose snippet for prompt ---
        func_start, func_end = 0, len(file_lines)

        if syntax_errors:
            err_line = syntax_errors[0].get('lineno') or 1
            original_snippet, _ = get_focused_context(file_source, err_line, window=20)
            original_header = f"ORIGINAL FILE TO EDIT (focused around line {err_line}):\n"
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
            edit_prompt = (
                f"Given this analysis:\n{analysis}\n\n"
                "Now produce a SEARCH/REPLACE block. Rules:\n"
                "1. SEARCH must EXACTLY match existing code character-for-character including indentation.\n"
                "2. Include a few lines of surrounding context so SEARCH is unique in the file.\n"
                "3. When ADDING code, keep the surrounding lines in REPLACE so they are not deleted.\n"
                "4. Output ONLY the block — no conversational text.\n\n"
                "CRITICAL RULE: Your SEARCH block must contain ONLY the specific lines being changed (max 20-30 lines). "
                "If your SEARCH block contains the entire file, it will be REJECTED. Output targeted, surgical edits only.\n\n"
                f"FORMAT:\n{FORMAT_BLOCK}\n\n"
                f"{original_header}```{lang_fence}\n{original_snippet}\n```\n\n"
                f"INSTRUCTION: {instruction}\n\n"
                f"{context_block}"  # purpose: provide RAG and history context to guide the rewrite, even in this fallback scenario
            )
            raw_output = llm.invoke(edit_prompt).strip()
        except Exception as e:
            logger.error("LLM invocation failed: %s", e)
            logger.debug(traceback.format_exc())
            return file_source

        def whole_function_replace(source: str, func_text: str, start_idx: int, end_idx: int) -> str:
            source_lines = source.split('\n')
            new_func_lines = func_text.strip('\n').split('\n')
            return '\n'.join(source_lines[:start_idx] + new_func_lines + source_lines[end_idx:])

        def strip_markdown(code: str) -> str:
            try:
                m = re.search(r"```(?:[a-zA-Z0-9_+\-]*)\s*\n(.*?)\n```", code, re.DOTALL)
                if m:
                    return m.group(1)
            except Exception:
                logger.warning("Failed to strip markdown fences; using raw output")
            return code

        raw_output = strip_markdown(raw_output)

        def parse_multiple_blocks(text: str):
            pattern = re.compile(
                r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE",
                re.DOTALL
            )
            return [(m.group(1), m.group(2)) for m in pattern.finditer(text)]

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
        

        # change syntax_errors away and define multi_func_hint here since it's only used in this block now
        if post_errors and not syntax_errors:
            logger.warning("LLM edits introduced %s new syntax error(s) in %s", len(post_errors), path)
            current_source = "\n".join(source_lines)
            broken_lines = current_source.split('\n')
            post_error_note = "NEWLY INTRODUCED SYNTAX ERRORS (fix these only):\n\n"
            for err in post_errors:
                se_line = err['lineno']
                start = max(0, se_line - 3)
                end = min(len(broken_lines), se_line + 2)
                snippet = '\n'.join(f"{start+i+1}: {l}" for i, l in enumerate(broken_lines[start:end]))
                post_error_note += (
                    f"ERROR: {err['msg']} at Line {se_line}, Col {err['offset']}\n"
                    f"EXACT BROKEN SNIPPET:\n```\n{snippet}\n```\n\n"
                )
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
                f"Rewrite the following function to apply this change: {instruction}\n\n"
                f"CURRENT FUNCTION:\n```{lang_fence}\n{verbatim}\n```\n\n"
                "Rules:\n"
                "1. Output ONLY the rewritten function — no explanation, no markdown fence.\n"
                "2. Preserve ALL existing logic except what the instruction explicitly changes.\n"
                "3. Keep exact indentation of the original.\n"
                f"{context_block}"  # purpose: provide RAG and history context to guide the rewrite, even in this fallback scenario
            )
            try:
                rewritten = llm.invoke(rewrite_prompt).strip()
                rewritten = strip_markdown(rewritten) if "```" in rewritten else rewritten
                if rewritten.strip():
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
            return new_source

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
