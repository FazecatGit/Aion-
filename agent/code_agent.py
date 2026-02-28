import re
import logging
import traceback
from langchain_ollama import OllamaLLM
from typing import List, Dict, Optional

from brain.config import LLM_MODEL
from brain.fast_search import fast_topic_search
from db.db_reader import get_code_documents
from .tools import read_file, write_file, run_git_command, show_diff, run_python_file, run_shell_command, list_files

logger = logging.getLogger("code_agent")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class CodeAgent:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def edit_code(
        self,
        path: str,
        instruction: str,
        dry_run: bool = True,
        use_rag: bool = False,
        session_chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:

        file_source = read_file(path)

        # --- Collect ALL syntax errors
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

        # --- Build syntax error note for ALL errors ---
        syntax_errors = collect_syntax_errors(file_source, path)
        syntax_error_note = ""

        if syntax_errors:
            broken_lines = file_source.split('\n')
            syntax_error_note = "SYNTAX_ERRORS_DETECTED (fix ALL of them):\n\n"

            for err in syntax_errors:
                se_line = err['lineno']
                start = max(0, se_line - 3)
                end = min(len(broken_lines), se_line + 2)
                snippet = '\n'.join(
                    f"{start + i + 1}: {l}"
                    for i, l in enumerate(broken_lines[start:end])
                )
                syntax_error_note += (
                    f"ERROR: {err['msg']} at Line {se_line}, Col {err['offset']}\n"
                    f"EXACT BROKEN SNIPPET (use VERBATIM in SEARCH block):\n"
                    f"```\n{snippet}\n```\n\n"
                )

            syntax_error_note += (
                "Output ONE SEARCH/REPLACE block per error, in order top to bottom.\n"
                "Each SEARCH block must match the exact broken lines shown above."
            )
            logger.info("Found %s syntax error(s) in %s", len(syntax_errors), path)
        
        runtime_error_note = ""
        if not syntax_errors:  # only run if file compiles cleanly
            try:
                import subprocess, sys
                run_result = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                    text=True,
                    timeout=5  # prevent infinite loops from hanging forever
                )
                if run_result.returncode != 0 and run_result.stderr:
                    runtime_error_note = (
                        "RUNTIME_ERRORS_DETECTED (fix these too):\n"
                        f"```\n{run_result.stderr[:2000]}\n```\n\n"
                        "Output ONE SEARCH/REPLACE block per error fix."
                    )
                    logger.info("Runtime errors detected in %s", path)
            except subprocess.TimeoutExpired:
                runtime_error_note = (
                    "RUNTIME_WARNING: File timed out during execution — "
                    "likely contains an infinite loop. Check loop conditions.\n"
                )
                logger.warning("File %s timed out during runtime check", path)
            except Exception as e:
                logger.debug("Runtime check failed: %s", e)

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
                rag_context = ""

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

        combined_context = history_context + rag_context
        extra_syntax = ""
        if syntax_error_note:
            extra_syntax += "\n\n" + syntax_error_note
        if runtime_error_note:
            extra_syntax += "\n\n" + runtime_error_note

        # --- Prompt ---
        prompt = (
            f"ORIGINAL FILE TO EDIT:\n```python\n{file_source}\n```\n\n"
            "You are an expert AI coding agent. I need you to edit files as well with follow up edits.\n"
            "You must output a SEARCH/REPLACE block that specifies exactly what code to remove, and what code to insert.\n\n"
            "1. The SEARCH block must EXACTLY match the existing code in the file, character for character, including indentation.\n"
            "2. Include enough context (a few lines before and after) so the SEARCH block is unique in the file.\n"
            "3. If you are ADDING new code, you MUST include the original surrounding code inside the REPLACE block so it doesn't get deleted.\n"
            "4. INTERPRET INSTRUCTIONS LITERALLY: If asked to add a specific print statement, string, or command, type it EXACTLY as requested.\n"
            "5. Output ONLY the SEARCH/REPLACE block inside a single markdown block. No conversational text.\n\n"
            "If multiple independent fixes are required, output one SEARCH/REPLACE block per fix.\n"
            "Do NOT return the entire corrected file as a single replacement.\n\n"
            "FORMAT:\n"
            "```python\n"
            "<<<<<<< SEARCH\n"
            "(exact lines of old code here)\n"
            "=======\n"
            "(new edited code here)\n"
            ">>>>>>> REPLACE\n"
            "```\n\n"
            f"INSTRUCTION: {instruction}\n\n"
            f"{combined_context}"
            f"{extra_syntax}"
        )

        llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)
        logger.info("Waiting for LLM to calculate the exact edit...")
        try:
            raw_output = llm.invoke(prompt).strip()
        except Exception as e:
            logger.error("LLM invocation failed: %s", e)
            logger.debug(traceback.format_exc())
            return file_source

        # --- Strip markdown fences ---
        def strip_markdown(code: str) -> str:
            if '```python' in code:
                try:
                    # FIX: was .rsplit('```', 1)[1] — wrong index, should be [0]
                    return code.split('```python', 1)[1].rsplit('```', 1)[0]
                except Exception:
                    logger.warning("Failed to strip markdown fences; using raw output")
            return code

        raw_output = strip_markdown(raw_output)

        # --- Parse SEARCH/REPLACE blocks ---
        def parse_multiple_blocks(text: str):
            pattern = re.compile(
                r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\n(.*?)\n>>>>>>>\s*REPLACE",
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
            logger.warning(
                "Bad format from LLM: SEARCH/REPLACE not found. Retry %s/%s",
                retry_count + 1, max_retries
            )
            logger.debug("LLM raw output (truncated): %s", raw_output[:4000])

            followup = (
                "You must ONLY output a SEARCH/REPLACE block in this exact format (no explanation):\n"
                "```python\n<<<<<<< SEARCH\n(EXACT LINES TO SEARCH FOR)\n=======\n(REPLACEMENT LINES)\n>>>>>>> REPLACE\n```\n\n"
                "Important: The SEARCH block must include the exact surrounding lines from the target file so it matches character-for-character.\n"
                "If you previously returned a block that doesn't match, re-output it now with MORE surrounding context lines.\n\n"
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
            logger.debug("Final LLM raw output (truncated): %s", raw_output[:4000])
            return file_source

        # --- Apply blocks sequentially ---
        source_lines = file_source.split('\n')

        def apply_block_to_lines(source_lines, search_text, replace_text):
            search_lines = search_text.strip().split('\n')
            replace_lines = replace_text.strip('\n').split('\n')

            # Exact stripped match
            for i in range(len(source_lines) - len(search_lines) + 1):
                if all(
                    source_lines[i + j].strip() == search_lines[j].strip()
                    for j in range(len(search_lines))
                ):
                    return i, replace_lines

            # Anchor-based tolerant match
            anchors = [s.strip() for s in search_lines if s.strip()]
            if anchors:
                first_anchor = anchors[0]  # FIX: was anchors (the full list), not anchors[0]
                best_idx, best_score = -1, 0
                for i, line in enumerate(source_lines):
                    if line.strip() == first_anchor:
                        score = sum(
                            1 for j in range(min(len(search_lines), len(source_lines) - i))
                            if source_lines[i + j].strip() == search_lines[j].strip()
                        )
                        if score > best_score:
                            best_score, best_idx = score, i
                if best_score >= max(1, len(search_lines) // 3):
                    return best_idx, replace_lines

            # Fuzzy substring fallback
            try:
                from difflib import SequenceMatcher
                source_str = '\n'.join(source_lines)
                search_str = '\n'.join(s.strip() for s in search_lines)
                sm = SequenceMatcher(None, source_str, search_str)
                match = sm.find_longest_match(0, len(source_str), 0, len(search_str))
                if match.size > 40:
                    start_line = source_str[:match.a].count('\n')
                    return start_line, replace_lines
            except Exception:
                logger.debug("Fuzzy matching failed: %s", traceback.format_exc())

            return -1, None

        # Update chat history
        if session_chat_history is not None:
            session_chat_history.append({"role": "User", "content": instruction})
            session_chat_history.append({"role": "Assistant", "content": f"Edited {path}: {instruction}"})

        for idx, (search_text, replace_text) in enumerate(blocks):
            match_start_idx, replace_lines = apply_block_to_lines(source_lines, search_text, replace_text)
            if match_start_idx == -1:
                logger.error("Block %s: Unable to locate SEARCH block. Skipping.", idx + 1)
                continue

            original_indent = len(source_lines[match_start_idx]) - len(source_lines[match_start_idx].lstrip())
            indent_str = " " * original_indent

            first_non_empty = next((l for l in replace_lines if l.strip()), "")
            llm_base_indent = len(first_non_empty) - len(first_non_empty.lstrip())

            indented_replace = []
            for line in replace_lines:
                if not line.strip():
                    indented_replace.append("")
                elif llm_base_indent == 0:
                    llm_indent = len(line) - len(line.lstrip())
                    indented_replace.append(indent_str + (" " * llm_indent) + line.lstrip())
                else:
                    indented_replace.append(line)

            source_lines = (
                source_lines[:match_start_idx]
                + indented_replace
                + source_lines[match_start_idx + len(search_text.strip().split('\n')):]
            )

        new_source = "\n".join(source_lines)

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
