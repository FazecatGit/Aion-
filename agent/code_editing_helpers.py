"""
Code Editing Helpers - parsing, matching, and structural validation for code edits.

Provides the low-level machinery that CodeAgent relies on:
  - Syntax / compiler error collection
  - SEARCH/REPLACE block parsing and application
  - Function extraction with call-site expansion
  - Language detection and extension matching
  - Structural validation (brace balance, unreachable code, indentation)
  - Formatted error notes for LLM prompts
"""

import re
import logging
import traceback
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from brain.config import BRACE_LANGUAGES

logger = logging.getLogger("code_agent")


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Python / Go-style function signature
FUNC_PATTERN = re.compile(r"\s*(def|class|func)\s+\w+")

# C/C++/Java-style function: type name(...) {
# Handles: void foo(, int main(, static std::vector<int> bar(, const char* baz(
_C_FUNC_PATTERN = re.compile(
    r"^\s*"
    r"(?:(?:static|inline|virtual|extern|const|volatile|unsigned|signed)\s+)*"
    r"(?:[\w:*&<>]+\s+)+"
    r"(\w+)\s*\("
)

_C_FAMILY_EXTS = frozenset({
    '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.java', '.cs', '.js', '.ts',
})

_LANG_INDICATORS: dict[str, re.Pattern] = {
    "python": re.compile(
        r'\bdef\s+\w+\s*\(|^import\s|^from\s.*import\s|if\s+__name__|^\s+return\s|:\s*$',
        re.MULTILINE,
    ),
    "go": re.compile(
        r'\bfunc\s+\w+\s*\(|^package\s|:=|fmt\.\w+|^import\s+\(',
        re.MULTILINE,
    ),
    "cpp": re.compile(
        r'#include\s|int\s+main\s*\(|std::|void\s+\w+\s*\(|->|cout\s*<<',
        re.MULTILINE,
    ),
    "rust": re.compile(
        r'\bfn\s+\w+|let\s+mut\s|println!\(|use\s+\w+::|impl\s+',
        re.MULTILINE,
    ),
}

_EXT_TO_LANG: dict[str, str] = {
    '.py': 'python', '.go': 'go',
    '.cpp': 'cpp', '.c': 'cpp', '.h': 'cpp', '.hpp': 'cpp',
    '.rs': 'rust',
}

_RETURN_KEYWORDS = frozenset({'return', 'break', 'continue'})


# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_source_language(source: str) -> str:
    """Detect the likely programming language of *source*.

    Returns one of: 'python', 'go', 'cpp', 'rust', 'unknown'.
    """
    first_lines = '\n'.join(source.strip().split('\n')[:20])
    scores = {
        lang: sum(1 for _ in pattern.finditer(first_lines))
        for lang, pattern in _LANG_INDICATORS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else 'unknown'


def source_matches_ext(source: str, ext: str) -> bool:
    """Return True if the detected language matches *ext* (or detection is uncertain)."""
    detected = detect_source_language(source)
    if detected == 'unknown':
        return True
    expected = _EXT_TO_LANG.get(ext)
    if not expected:
        return True
    return detected == expected


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

def collect_syntax_errors(source: str, path: str) -> list[dict]:
    """Collect all syntax errors from Python source code (up to 10)."""
    errors: list[dict] = []
    temp = source
    lines = temp.split('\n')
    seen_lines: set[int | None] = set()

    for _ in range(10):
        try:
            compile(temp, path, 'exec')
            break
        except SyntaxError as se:
            lineno = getattr(se, 'lineno', None)
            if lineno in seen_lines:
                break
            seen_lines.add(lineno)
            errors.append({
                'msg': se.msg,
                'lineno': lineno,
                'offset': getattr(se, 'offset', None),
                'text': (getattr(se, 'text', '') or '').strip(),
            })
            if lineno and lineno <= len(lines):
                lines[lineno - 1] = '# SYNTAX_ERROR_PLACEHOLDER'
                temp = '\n'.join(lines)
    return errors


def parse_compiler_errors(stderr: str) -> list[dict]:
    """Parse ``file:line:col: message`` errors from compiler stderr."""
    return [
        {'file': m.group(1), 'lineno': int(m.group(2)), 'col': int(m.group(3)), 'msg': m.group(4)}
        for m in re.finditer(r"^(.+?):(\d+):(\d+):\s+(.+)$", stderr, re.MULTILINE)
    ]


def extract_error_lines_from_text(text: str) -> list[int]:
    """Extract line numbers mentioned in error messages / user instructions.

    Handles compiler ``file:line:col``, Python traceback ``File "...", line N``,
    and generic ``line N`` / ``Line N:`` patterns.
    """
    lines: set[int] = set()
    for m in re.finditer(r'[\w./\\]+\.\w+:(\d+):\d+', text):
        lines.add(int(m.group(1)))
    for m in re.finditer(r'[Ff]ile\s+"[^"]+",\s*line\s+(\d+)', text):
        lines.add(int(m.group(1)))
    for m in re.finditer(r'\b[Ll]ine\s+(\d+)\b', text):
        lines.add(int(m.group(1)))
    return sorted(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# ERRORNOTE FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _ErrorNoteStyle:
    """Configuration for how an error note is rendered."""
    title: str
    lineno_key: str
    col_key: str
    include_raw: bool
    footer: str


_SYNTAX_STYLE = _ErrorNoteStyle(
    title="SYNTAX_ERRORS_DETECTED (fix ALL of them):",
    lineno_key="lineno",
    col_key="offset",
    include_raw=True,
    footer=(
        "Output ONE SEARCH/REPLACE block per error, in order top to bottom.\n"
        "Each SEARCH block must match the exact broken lines shown above."
    ),
)

_RUNTIME_STYLE = _ErrorNoteStyle(
    title="RUNTIME_ERRORS_DETECTED (fix ALL of them):",
    lineno_key="lineno",
    col_key="col",
    include_raw=True,
    footer="Output ONE SEARCH/REPLACE block per error, top to bottom.",
)

_POST_EDIT_STYLE = _ErrorNoteStyle(
    title="NEWLY INTRODUCED SYNTAX ERRORS (fix these only):",
    lineno_key="lineno",
    col_key="offset",
    include_raw=False,
    footer="",
)


def _build_error_note(errors: list[dict], file_lines: list[str], style: _ErrorNoteStyle) -> str:
    """Build a formatted prompt note from a list of error dicts.

    Centralises the logic previously spread across three separate builders.
    """
    if not errors:
        return ""

    parts = [f"{style.title}\n"]
    for err in errors:
        lineno = err.get(style.lineno_key, 0)
        col = err.get(style.col_key, '')
        start = max(0, lineno - 3)
        end = min(len(file_lines), lineno + 2)
        snippet = '\n'.join(f"{start + i + 1}: {l}" for i, l in enumerate(file_lines[start:end]))

        col_str = f", Col {col}" if col else ""
        parts.append(
            f"ERROR: {err['msg']} at Line {lineno}{col_str}\n"
            f"CONTEXT (line numbers are reference only  never copy them into SEARCH/REPLACE):\n"
            f"```\n{snippet}\n```\n"
        )
        if style.include_raw:
            raw_lines = '\n'.join(file_lines[start:end])
            parts.append(f"RAW LINES FOR SEARCH (no line numbers):\n```\n{raw_lines}\n```\n")
        parts.append("")

    if style.footer:
        parts.append(style.footer)
    return "\n".join(parts)


def build_syntax_error_note(syntax_errors: list, file_lines: list) -> str:
    """Build a formatted note describing syntax errors."""
    return _build_error_note(syntax_errors, file_lines, _SYNTAX_STYLE)


def build_runtime_error_note(compiler_errors: list, file_lines: list) -> str:
    """Build a formatted note describing runtime/compiler errors."""
    return _build_error_note(compiler_errors, file_lines, _RUNTIME_STYLE)


def build_post_error_note(post_errors: list, file_lines: list) -> str:
    """Build a formatted note for syntax errors introduced by LLM edits."""
    return _build_error_note(post_errors, file_lines, _POST_EDIT_STYLE)


#  ═══════════════════════════════════════════════════════════════════════════════
# TEXT & BLOCK PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def strip_markdown(code: str) -> str:
    """Remove markdown code fences from LLM output."""
    try:
        m = re.search(r"```(?:[a-zA-Z0-9_+\-]*)\s*\n(.*?)\n```", code, re.DOTALL)
        if m:
            return m.group(1)
    except Exception:
        logger.warning("Failed to strip markdown fences; using raw output")
    return code


def parse_multiple_blocks(text: str) -> list[tuple[str, str]]:
    """Parse one or more SEARCH/REPLACE blocks from LLM output."""
    pattern = re.compile(
        r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE",
        re.DOTALL,
    )
    return [(m.group(1), m.group(2)) for m in pattern.finditer(text)]


def parse_blocks_with_retry(raw_output: str) -> tuple[list[tuple[str, str]], str]:
    """Parse SEARCH/REPLACE blocks, stripping markdown on first failure.

    Returns ``(blocks, cleaned_output)`` so callers don't need to repeat
    the ``if not blocks: strip -> retry`` pattern themselves.
    """
    blocks = parse_multiple_blocks(raw_output)
    if not blocks:
        raw_output = strip_markdown(raw_output)
        blocks = parse_multiple_blocks(raw_output)
    return blocks, raw_output


# ═══════════════════════════════════════════════════════════════════════════════
# INDENTATION & CONTEXT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_focused_context(source: str, error_lineno: int, window: int = 20) -> tuple[str, str]:
    """Extract focused context around *error_lineno* with numbered lines."""
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


def reindent_block(replace_lines: list[str], file_indent: int, llm_base_indent: int) -> list[str]:
    """Re-indent a block of replacement lines to match the file indentation."""
    result: list[str] = []
    for line in replace_lines:
        if not line.strip():
            result.append("")
        elif llm_base_indent == 0:
            llm_spaces = len(line) - len(line.lstrip())
            result.append(' ' * file_indent + ' ' * llm_spaces + line.lstrip())
        else:
            delta = file_indent - llm_base_indent
            current = len(line) - len(line.lstrip())
            result.append(' ' * max(0, current + delta) + line.lstrip())
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTION EXTRACTION & CALL-SITE EXPANSION
# ═══════════════════════════════════════════════════════════════════════════════

def _find_function_range(lines: list[str], start_hint: int) -> tuple[int, int]:
    """Walk from *start_hint* to find the (start, end) of the enclosing function."""
    start_idx = start_hint
    while start_idx > 0:
        if FUNC_PATTERN.match(lines[start_idx]) or _C_FUNC_PATTERN.match(lines[start_idx]):
            break
        start_idx -= 1

    is_python_style = FUNC_PATTERN.match(lines[start_idx]) if start_idx < len(lines) else False

    indent = 0
    if start_idx < len(lines) and (FUNC_PATTERN.match(lines[start_idx]) or _C_FUNC_PATTERN.match(lines[start_idx])):
        indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())

    # -- Indent-based end detection (Python/Ruby) --
    if is_python_style:
        end_idx = start_idx + 1
        while end_idx < len(lines):
            line = lines[end_idx]
            if (FUNC_PATTERN.match(line) or _C_FUNC_PATTERN.match(line)) and (len(line) - len(line.lstrip())) <= indent:
                break
            end_idx += 1
        return start_idx, end_idx

    # -- Brace-based end detection (C/C++/Java/Go/JS) --
    brace_depth = 0
    seen_open = False
    for i in range(start_idx, len(lines)):
        for ch in lines[i]:
            if ch == '{':
                brace_depth += 1
                seen_open = True
            elif ch == '}':
                brace_depth -= 1
        if seen_open and brace_depth <= 0:
            return start_idx, i + 1

    # Fallback: walk to next top-level function
    end_idx = start_idx + 1
    while end_idx < len(lines):
        line = lines[end_idx]
        if (FUNC_PATTERN.match(line) or _C_FUNC_PATTERN.match(line)) and (len(line) - len(line.lstrip())) <= indent:
            break
        end_idx += 1
    return start_idx, end_idx


def _find_call_sites(
    lines: list[str], func_name: str, func_start: int, func_end: int, context_window: int = 5,
) -> list[tuple[int, int]]:
    """Find line ranges outside the function body that call *func_name*."""
    call_pattern = re.compile(rf'\b{re.escape(func_name)}\s*\(')
    regions: list[tuple[int, int]] = []

    for i, line in enumerate(lines):
        if func_start <= i < func_end:
            continue
        if call_pattern.search(line):
            regions.append(_find_function_range(lines, i))

    if not regions:
        return []
    regions.sort()
    merged = [regions[0]]
    for s, e in regions[1:]:
        if s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def extract_function(source: str, instruction: str, hint_blocks=None) -> tuple[str, int, int]:
    """Extract the relevant function/class from *source* based on *instruction*.

    Also includes call sites of the target function so the LLM can produce
    multi-site SEARCH/REPLACE blocks.
    """
    lines = source.split('\n')
    target_idx: Optional[int] = None
    func_name: Optional[str] = None

    # -- Hint-based discovery --
    if hint_blocks:
        try:
            first_search = hint_blocks[0][0]
            for search_line in first_search.split('\n'):
                stripped = search_line.strip()
                if not stripped:
                    continue
                for i, l in enumerate(lines):
                    if l.strip() == stripped:
                        target_idx = i
                        break
                if target_idx is not None:
                    break
        except Exception:
            target_idx = None

    # -- Instruction-based discovery --
    if target_idx is None:
        m = re.search(
            r"\b([\w_]+)\s+function\b"
            r"|function\s+([\w_]+)\b"
            r"|def\s+([\w_]+)\b"
            r"|func\s+([\w_]+)\b"
            r"|\b([\w_]+)\s*\(\)",
            instruction, re.I,
        )
        func_name = next((g for g in m.groups() if g), None) if m else None
        if func_name:
            target_idx = _locate_function_definition(lines, func_name)

    # -- Word-matching fallback: scan instruction for any function name in file --
    if target_idx is None:
        # Collect all function names defined in the file
        _file_func_names: list[tuple[str, int]] = []
        for i, l in enumerate(lines):
            pm = re.match(r"\s*(?:def|func)\s+(\w+)", l)
            if pm:
                _file_func_names.append((pm.group(1), i))
            else:
                cm = _C_FUNC_PATTERN.match(l)
                if cm:
                    _file_func_names.append((cm.group(1), i))
        # Check if any file function name appears in the instruction
        _instr_lower = instruction.lower()
        for _fn, _fi in _file_func_names:
            if _fn.lower() in _instr_lower:
                func_name = _fn
                target_idx = _fi
                break

    # -- LLM-based function targeting: ask the LLM which function the instruction refers to --
    if target_idx is None and _file_func_names:
        try:
            from langchain_ollama import OllamaLLM
            from brain.config import LLM_MODEL
            _target_llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)
            _sig_list = "\n".join(f"  {i+1}. {fn} (line {fi+1})" for i, (fn, fi) in enumerate(_file_func_names))
            _target_prompt = (
                f"A user wants to edit code. Their instruction:\n\"{instruction}\"\n\n"
                f"The file contains these functions:\n{_sig_list}\n\n"
                "Which function number should be edited? If the instruction is about "
                "adding a NEW function (not editing existing ones), respond NEW.\n"
                "Answer with ONLY the number or NEW."
            )
            _target_answer = _target_llm.invoke(_target_prompt).strip()
            if _target_answer.upper() != "NEW":
                _chosen_idx = int(re.search(r'\d+', _target_answer).group()) - 1
                if 0 <= _chosen_idx < len(_file_func_names):
                    func_name, target_idx = _file_func_names[_chosen_idx]
                    logger.info("[EXTRACT] LLM matched instruction to function '%s' (line %d)",
                                func_name, target_idx + 1)
            else:
                logger.info("[EXTRACT] LLM says instruction is for a NEW function, not an existing one")
        except Exception as e:
            logger.debug("[EXTRACT] LLM function targeting failed: %s", e)

    # -- Determine function range --
    if target_idx is None:
        start_idx = 0
    else:
        start_idx = target_idx
        while start_idx > 0 and not FUNC_PATTERN.match(lines[start_idx]) and not _C_FUNC_PATTERN.match(lines[start_idx]):
            start_idx -= 1
        if not FUNC_PATTERN.match(lines[start_idx]) and not _C_FUNC_PATTERN.match(lines[start_idx]) and start_idx != 0:
            for i in range(target_idx, -1, -1):
                if FUNC_PATTERN.match(lines[i]) or _C_FUNC_PATTERN.match(lines[i]):
                    start_idx = i
                    break

    func_start, func_end = _find_function_range(lines, start_idx)

    if target_idx is None and start_idx == 0:
        # Could not identify a specific function.  If the file has only one
        # function, return that function's range (safe).  Otherwise return
        # the range of just the *first* function to avoid handing back the
        # entire file which would cause the whole-function-rewrite fallback
        # to destroy unrelated code.
        first_func_start, first_func_end = _find_function_range(lines, 0)
        if first_func_end >= len(lines):
            # _find_function_range walked to EOF — check if there really is
            # only one function before returning full-file range.
            func_count = count_top_level_functions(lines)
            if func_count <= 1:
                return '\n'.join(lines), 0, len(lines)
            # Multiple functions but we can't tell which one — return the
            # first function only as a safe default.
            # Re-scan for first real function definition.
            for _i, _l in enumerate(lines):
                if FUNC_PATTERN.match(_l) or _C_FUNC_PATTERN.match(_l):
                    _fs, _fe = _find_function_range(lines, _i)
                    return '\n'.join(lines[_fs:_fe]), _fs, _fe
            # No function patterns at all — return full file as last resort.
            return '\n'.join(lines), 0, len(lines)
        return '\n'.join(lines[first_func_start:first_func_end]), first_func_start, first_func_end

    # -- Call-site expansion --
    call_regions = _find_call_sites(lines, func_name, func_start, func_end) if func_name else []

    if not call_regions:
        return '\n'.join(lines[func_start:func_end]), func_start, func_end

    return _build_composite_snippet(lines, func_name, func_start, func_end, call_regions)


def _locate_function_definition(lines: list[str], func_name: str) -> Optional[int]:
    """Find the line index of a function *definition* by name."""
    # Python/Go style
    for i, l in enumerate(lines):
        if re.match(rf"\s*(def|func)\s+{re.escape(func_name)}\b", l, re.I):
            return i
    # C/C++/Java style
    for i, l in enumerate(lines):
        cm = _C_FUNC_PATTERN.match(l)
        if cm and cm.group(1) == func_name:
            return i
        if re.search(rf'\b{re.escape(func_name)}\s*\(', l) and _looks_like_definition(l):
            return i
    return None


def _looks_like_definition(line: str) -> bool:
    """Heuristic: a line that contains ``name(`` is a definition (not a call)."""
    stripped = line.lstrip()
    return not any(stripped.startswith(kw) for kw in ('if', 'while', 'for', 'return'))


def _build_composite_snippet(
    lines: list[str],
    func_name: str,
    func_start: int,
    func_end: int,
    call_regions: list[tuple[int, int]],
) -> tuple[str, int, int]:
    """Build a composite snippet containing the function definition + its call sites."""
    parts = [
        f"// -- FUNCTION DEFINITION (lines {func_start + 1}-{func_end}) --",
        '\n'.join(lines[func_start:func_end]),
        "",
    ]
    emitted = 0
    for cs, ce in call_regions:
        if cs >= func_start and ce <= func_end:
            continue
        if cs < func_end and ce > func_end:
            cs = func_end
        parts.append(f"// -- CALL SITE (lines {cs + 1}-{ce}) --")
        parts.append('\n'.join(lines[cs:ce]))
        parts.append("")
        emitted += 1

    if emitted == 0:
        return '\n'.join(lines[func_start:func_end]), func_start, func_end

    logger.info(
        "[EXTRACT] Expanded snippet: function '%s' (%d-%d) + %d call-site region(s)",
        func_name, func_start + 1, func_end, emitted,
    )
    overall_end = max(func_end, max(ce for _, ce in call_regions))
    return '\n'.join(parts), func_start, overall_end


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def apply_block_to_lines(
    source_lines: list[str], search_text: str, replace_text: str,
) -> tuple[int, list[str] | None, int]:
    """Match and replace a SEARCH block within *source_lines*.

    Tries four strategies in order:
      1. Exact match (trailing-whitespace normalised)
      2. Whitespace-insensitive match
      3. Anchor + sliding-window tolerant match
      4. Fuzzy difflib fallback
    """
    search_lines = _strip_blank_edges(search_text.split('\n'))
    replace_lines = replace_text.strip('\n').split('\n')
    n = len(search_lines)
    if n == 0:
        return -1, None, 0

    indent_char = detect_indent_char('\n'.join(source_lines))

    def _get_replace(match_idx: int) -> list[str]:
        if indent_char == '\t':
            return replace_lines
        file_indent = len(source_lines[match_idx]) - len(source_lines[match_idx].lstrip())
        first_non_empty = next((l for l in replace_lines if l.strip()), "")
        llm_base = len(first_non_empty) - len(first_non_empty.lstrip()) if first_non_empty else 0
        return reindent_block(replace_lines, file_indent, llm_base)

    # Strategy 1 -- exact (trailing whitespace normalised)
    for i in range(len(source_lines) - n + 1):
        if all(source_lines[i + j].rstrip() == search_lines[j].rstrip() for j in range(n)):
            return i, _get_replace(i), n

    # Strategy 2 -- whitespace-insensitive
    for i in range(len(source_lines) - n + 1):
        if all(source_lines[i + j].strip() == search_lines[j].strip() for j in range(n)):
            return i, _get_replace(i), n

    # Strategy 3 -- anchor + sliding window
    result = _anchor_match(source_lines, search_lines, n, _get_replace)
    if result[0] != -1:
        return result

    # Strategy 4 -- fuzzy difflib
    return _fuzzy_match(source_lines, search_lines, n, _get_replace)


def _strip_blank_edges(lines: list[str]) -> list[str]:
    """Strip leading and trailing blank lines."""
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _anchor_match(
    source_lines: list[str],
    search_lines: list[str],
    n: int,
    get_replace,
) -> tuple[int, list[str] | None, int]:
    """Find the best anchor-based match."""
    anchors = [s.strip() for s in search_lines if s.strip()]
    if not anchors:
        return -1, None, 0

    first_anchor = anchors[0]
    best_idx, best_score = -1, 0
    for i, line in enumerate(source_lines):
        if line.strip() != first_anchor:
            continue
        score = sum(
            1 for j in range(min(n, len(source_lines) - i))
            if source_lines[i + j].strip() == search_lines[j].strip()
        )
        if score > best_score:
            best_score, best_idx = score, i

    if best_score >= max(3, int(n * 0.7)):
        return best_idx, get_replace(best_idx), n
    return -1, None, 0


def _fuzzy_match(
    source_lines: list[str],
    search_lines: list[str],
    n: int,
    get_replace,
) -> tuple[int, list[str] | None, int]:
    """Fuzzy difflib-based match as last resort.

    Only matches if the longest common substring covers a significant
    portion of the search text to avoid catastrophic mis-replacements
    that destroy unrelated functions.
    """
    try:
        source_str = '\n'.join(source_lines)
        search_str = '\n'.join(s.rstrip() for s in search_lines)
        sm = SequenceMatcher(None, source_str, search_str, autojunk=False)
        match = sm.find_longest_match(0, len(source_str), 0, len(search_str))
        # Require the match to cover at least 60% of the search text
        if match.size > max(60, int(len(search_str) * 0.6)):
            start_line = source_str[:match.a].count('\n')
            # Clamp replacement span: don't exceed available lines past the start
            safe_n = min(n, len(source_lines) - start_line)
            return start_line, get_replace(start_line), safe_n
    except Exception:
        logger.debug("Fuzzy matching failed: %s", traceback.format_exc())
    return -1, None, 0


def is_oversized_block(
    search_text: str,
    source_lines: list[str],
    idx: int | None = None,
    prefix: str = "Block",
    max_ratio: float = 0.6,
) -> bool:
    """Return True if a SEARCH block is too large relative to the file.

    For small files (< 50 lines) the ratio is relaxed to 0.95 because
    single-function files legitimately require replacing most of the content.
    """
    try:
        search_line_count = len(search_text.strip('\n').split('\n'))
        file_len = len(source_lines)
        effective_ratio = 0.95 if file_len < 50 else max_ratio
        if search_line_count > file_len * effective_ratio:
            tag = f"{prefix} {idx}" if idx is not None else prefix
            logger.warning("%s rejected -- SEARCH spans %s/%s lines. Skipping.", tag, search_line_count, file_len)
            return True
    except Exception:
        logger.debug("Oversize check failed: %s", traceback.format_exc())
    return False


def find_first_mismatch(source_lines: list[str], search_lines: list[str]) -> tuple[int, str | None]:
    """Find the first line in *search_lines* that has no verbatim match in *source_lines*."""
    for j, sl in enumerate(search_lines):
        if sl.strip() and not any(sl.strip() == src.strip() for src in source_lines):
            return j, sl
    return -1, None


def whole_function_replace(source: str, func_text: str, start_idx: int, end_idx: int) -> str:
    """Replace lines ``[start_idx:end_idx]`` in *source* with *func_text*.

    Includes a safety check: if the replacement would reduce the number of
    top-level functions, the original source is returned unchanged.
    """
    source_lines = source.split('\n')
    new_lines = source_lines[:start_idx] + func_text.strip('\n').split('\n') + source_lines[end_idx:]
    orig_count = count_top_level_functions(source_lines)
    new_count = count_top_level_functions(new_lines)
    if orig_count > 1 and new_count < orig_count:
        logger.warning(
            "whole_function_replace safety: function count dropped %d → %d. Keeping original.",
            orig_count, new_count,
        )
        return source
    return '\n'.join(new_lines)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTION COUNTING
# ═══════════════════════════════════════════════════════════════════════════════

def count_top_level_functions(file_lines: list[str], ext: str = "") -> int:
    """Count top-level function definitions (excludes class declarations in C-family)."""
    is_c_family = ext.lower() in _C_FAMILY_EXTS
    count = 0

    for line in file_lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if not stripped:
            continue
        if is_c_family:
            if stripped.startswith('class '):
                continue
            if _C_FUNC_PATTERN.match(line) and indent <= 4:
                count += 1
        else:
            if FUNC_PATTERN.match(line) and indent == 0:
                count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_string_literals(line: str) -> str:
    """Remove string/char literals from a line to avoid false-positives in brace counting."""
    result: list[str] = []
    in_string = False
    string_char: str | None = None
    escaped = False

    for ch in line:
        if escaped:
            escaped = False
            continue
        if ch == '\\':
            escaped = True
            if not in_string:
                result.append(ch)
            continue
        if in_string:
            if ch == string_char:
                in_string = False
            continue
        if ch in ('"', "'", '`'):
            in_string = True
            string_char = ch
            continue
        result.append(ch)
    return ''.join(result)


def validate_code_structure(source: str, ext: str) -> list[dict]:
    """Validate structural integrity: brace balance, unreachable code, indentation.

    Returns list of ``{'type', 'msg', 'lineno', 'severity'}`` dicts.
    """
    lines = source.split('\n')
    issues: list[dict] = []
    issues.extend(_check_brace_balance(lines, ext))
    issues.extend(_check_unreachable_code(lines))
    issues.extend(_check_mixed_indentation(lines))
    return issues


def _check_brace_balance(lines: list[str], ext: str) -> list[dict]:
    """Check for unmatched braces in brace-based languages."""
    if ext not in BRACE_LANGUAGES:
        return []

    issues: list[dict] = []
    brace_stack: list[int] = []

    for i, line in enumerate(lines):
        stripped = _strip_string_literals(line)
        comment_idx = stripped.find('//')
        if comment_idx != -1:
            stripped = stripped[:comment_idx]
        for ch in stripped:
            if ch == '{':
                brace_stack.append(i + 1)
            elif ch == '}':
                if brace_stack:
                    brace_stack.pop()
                else:
                    issues.append({
                        'type': 'extra_closing_brace',
                        'msg': f'Extra closing brace at line {i + 1} -- no matching opening brace',
                        'lineno': i + 1,
                        'severity': 'error',
                    })

    for open_line in brace_stack:
        issues.append({
            'type': 'unclosed_brace',
            'msg': f'Unclosed opening brace from line {open_line} -- missing closing brace',
            'lineno': open_line,
            'severity': 'error',
        })
    return issues


def _check_unreachable_code(lines: list[str]) -> list[dict]:
    """Detect statements immediately following ``return`` / ``break`` / ``continue``."""
    issues: list[dict] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        word = stripped.split('(')[0].split(' ')[0]
        if word not in _RETURN_KEYWORDS:
            continue
        indent = len(line) - len(line.lstrip())
        for j in range(i + 1, min(i + 10, len(lines))):
            nxt = lines[j]
            nxt_stripped = nxt.strip()
            if not nxt_stripped or nxt_stripped.startswith('//') or nxt_stripped.startswith('#'):
                continue
            if nxt_stripped in ('}', ')', ']'):
                break
            if FUNC_PATTERN.match(nxt):
                break
            if (len(nxt) - len(nxt.lstrip())) >= indent:
                issues.append({
                    'type': 'unreachable_code',
                    'msg': f'Unreachable code at line {j + 1} (after `{word}` at line {i + 1})',
                    'lineno': j + 1,
                    'severity': 'warning',
                })
            break
    return issues


def _check_mixed_indentation(lines: list[str]) -> list[dict]:
    """Flag files that mix tabs and spaces."""
    has_tabs = any(line and line[0] == '\t' for line in lines if line.strip())
    has_spaces = any(line.startswith('  ') for line in lines if line.strip())
    if has_tabs and has_spaces:
        return [{
            'type': 'mixed_indentation',
            'msg': 'File mixes tabs and spaces for indentation',
            'lineno': 0,
            'severity': 'warning',
        }]
    return []


def build_structural_issue_note(issues: list[dict], file_lines: list[str]) -> str:
    """Build a formatted prompt note describing structural code issues."""
    if not issues:
        return ""

    parts = ["STRUCTURAL ISSUES DETECTED (fix ALL of them):\n"]
    for issue in issues:
        lineno = issue.get('lineno', 0)
        if lineno > 0:
            start = max(0, lineno - 3)
            end = min(len(file_lines), lineno + 2)
            snippet = '\n'.join(f"{start + i + 1}: {l}" for i, l in enumerate(file_lines[start:end]))
            parts.append(
                f"  {issue['type'].upper()}: {issue['msg']}\n"
                f"  CONTEXT:\n```\n{snippet}\n```\n"
            )
        else:
            parts.append(f"  {issue['type'].upper()}: {issue['msg']}\n")

    parts.append(
        "Fix each structural issue with a SEARCH/REPLACE block.\n"
        "Common fixes: remove unreachable code after return, add/remove braces to balance, fix indentation.\n"
    )
    return "\n".join(parts)