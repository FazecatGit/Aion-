"""
Helper functions for code editing operations.
Extracted from code_agent.py to improve readability and maintainability.
"""

import re
import logging
import traceback
from difflib import SequenceMatcher

logger = logging.getLogger("code_agent")

# --- Pattern Recognition ---
FUNC_PATTERN = re.compile(r"\s*(def|class|func)\s+\w+")


def collect_syntax_errors(source: str, path: str) -> list:
    """Collect all syntax errors from Python source code."""
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
    """Parse compiler errors from stderr (works for go, rustc, gcc, clang, tsc)."""
    errors = []
    for match in re.finditer(r"^(.+?):(\d+):(\d+):\s+(.+)$", stderr, re.MULTILINE):
        errors.append({
            'file': match.group(1),
            'lineno': int(match.group(2)),
            'col': int(match.group(3)),
            'msg': match.group(4)
        })
    return errors


def get_focused_context(source: str, error_lineno: int, window: int = 20) -> tuple:
    """Extract focused context around error line with numbering."""
    lines = source.split('\n')
    start = max(0, error_lineno - window)
    end = min(len(lines), error_lineno + window)
    raw = '\n'.join(lines[start:end])
    numbered = '\n'.join(f"{start + i + 1}: {l}" for i, l in enumerate(lines[start:end]))
    return raw, numbered


def detect_indent_char(source: str) -> str:
    """Detect if file uses tabs or spaces for indentation."""
    for line in source.split('\n'):
        if line and line[0] == '\t':
            return '\t'
    return ' '


def extract_function(source: str, instruction: str, hint_blocks=None) -> tuple:
    """Extract the relevant function/class from source based on instruction."""
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


def reindent_block(replace_lines: list, file_indent: int, llm_base_indent: int) -> list:
    """Reindent a block of code to match file indentation."""
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
    """Match and replace a code block within source lines."""
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
        if indent_char == '\t':
            return replace_lines
        file_indent = len(source_lines[match_idx]) - len(source_lines[match_idx].lstrip())
        first_non_empty = next((l for l in replace_lines if l.strip()), "")
        llm_base_indent = len(first_non_empty) - len(first_non_empty.lstrip()) if first_non_empty else 0
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
            return best_idx, get_replace(best_idx), n

    # 4. Fuzzy difflib fallback
    try:
        source_str = '\n'.join(source_lines)
        search_str = '\n'.join(s.rstrip() for s in search_lines)
        sm = SequenceMatcher(None, source_str, search_str, autojunk=False)
        match = sm.find_longest_match(0, len(source_str), 0, len(search_str))
        if match.size > 40:
            start_line = source_str[:match.a].count('\n')
            return start_line, get_replace(start_line), n
    except Exception:
        logger.debug("Fuzzy matching failed: %s", traceback.format_exc())

    return -1, None, 0


def is_oversized_block(search_text: str, source_lines: list, idx=None, prefix: str = "Block", max_ratio: float = 0.6) -> bool:
    """Check if a search block is too large (exceeds MAX_BLOCK_RATIO of file)."""
    try:
        search_line_count = len(search_text.strip('\n').split('\n'))
        file_len = len(source_lines)
        if search_line_count > file_len * max_ratio:
            if idx is not None:
                logger.warning("%s %s rejected — SEARCH spans %s/%s lines. Skipping.", prefix, idx, search_line_count, file_len)
            else:
                logger.warning("%s rejected — SEARCH spans %s/%s lines. Skipping.", prefix, search_line_count, file_len)
            return True
    except Exception:
        logger.debug("Oversize check failed: %s", traceback.format_exc())
    return False


def find_first_mismatch(source_lines: list, search_lines: list) -> tuple:
    """Find the first line in search_lines that doesn't exist in source_lines."""
    for j, sl in enumerate(search_lines):
        if sl.strip() and not any(sl.strip() == src.strip() for src in source_lines):
            return j, sl
    return -1, None


def whole_function_replace(source: str, func_text: str, start_idx: int, end_idx: int) -> str:
    """Replace a function in the source, keeping everything before and after."""
    source_lines = source.split('\n')
    new_func_lines = func_text.strip('\n').split('\n')
    return '\n'.join(source_lines[:start_idx] + new_func_lines + source_lines[end_idx:])


def strip_markdown(code: str) -> str:
    """Remove markdown code fences from output."""
    try:
        m = re.search(r"```(?:[a-zA-Z0-9_+\-]*)\s*\n(.*?)\n```", code, re.DOTALL)
        if m:
            return m.group(1)
    except Exception:
        logger.warning("Failed to strip markdown fences; using raw output")
    return code


def parse_multiple_blocks(text: str) -> list:
    """Parse one or more SEARCH/REPLACE blocks from LLM output."""
    pattern = re.compile(
        r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE",
        re.DOTALL
    )
    return [(m.group(1), m.group(2)) for m in pattern.finditer(text)]


def build_syntax_error_note(syntax_errors: list, file_lines: list) -> str:
    """Build a formatted note describing syntax errors."""
    if not syntax_errors:
        return ""
    
    note = "SYNTAX_ERRORS_DETECTED (fix ALL of them):\n\n"
    for err in syntax_errors:
        se_line = err['lineno']
        start = max(0, se_line - 3)
        end = min(len(file_lines), se_line + 2)
        snippet = '\n'.join(f"{start+i+1}: {l}" for i, l in enumerate(file_lines[start:end]))
        note += (
            f"ERROR: {err['msg']} at Line {se_line}, Col {err['offset']}\n"
            f"CONTEXT (line numbers are reference only — never copy them into SEARCH/REPLACE):\n"
            f"```\n{snippet}\n```\n"
            f"RAW LINES FOR SEARCH (no line numbers):\n"
            f"```\n{chr(10).join(file_lines[start:end])}\n```\n\n"
        )
    note += (
        "Output ONE SEARCH/REPLACE block per error, in order top to bottom.\n"
        "Each SEARCH block must match the exact broken lines shown above."
    )
    return note


def build_runtime_error_note(compiler_errors: list, file_lines: list) -> str:
    """Build a formatted note describing runtime/compiler errors."""
    if not compiler_errors:
        return ""
    
    note = "RUNTIME_ERRORS_DETECTED (fix ALL of them):\n\n"
    for err in compiler_errors:
        lineno = err['lineno']
        start = max(0, lineno - 3)
        end = min(len(file_lines), lineno + 2)
        snippet = '\n'.join(
            f"{start+i+1}: {l}"
            for i, l in enumerate(file_lines[start:end])
        )
        note += (
            f"ERROR: {err['msg']} at Line {lineno}, Col {err['col']}\n"
            f"CONTEXT (line numbers are reference only — never copy them into SEARCH/REPLACE):\n"
            f"```\n{snippet}\n```\n"
            f"RAW LINES FOR SEARCH (no line numbers):\n"
            f"```\n{chr(10).join(file_lines[start:end])}\n```\n\n"
        )
    note += "Output ONE SEARCH/REPLACE block per error, top to bottom."
    return note


def build_post_error_note(post_errors: list, file_lines: list) -> str:
    """Build a formatted note for syntax errors introduced by LLM edits."""
    if not post_errors:
        return ""
    
    note = "NEWLY INTRODUCED SYNTAX ERRORS (fix these only):\n\n"
    for err in post_errors:
        se_line = err['lineno']
        start = max(0, se_line - 3)
        end = min(len(file_lines), se_line + 2)
        snippet = '\n'.join(f"{start+i+1}: {l}" for i, l in enumerate(file_lines[start:end]))
        note += (
            f"ERROR: {err['msg']} at Line {se_line}, Col {err['offset']}\n"
            f"EXACT BROKEN SNIPPET:\n```\n{snippet}\n```\n\n"
        )
    return note
