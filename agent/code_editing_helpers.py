"""
Helper functions for code editing operations.
Extracted from code_agent.py to improve readability and maintainability.
"""

import re
import logging
import traceback
from difflib import SequenceMatcher

from brain.config import BRACE_LANGUAGES

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


# Also match C/C++/Java-style function definitions: type name(...) {
# Handles: void foo(, int main(, static std::vector<int> bar(, const char* baz(
_C_FUNC_PATTERN = re.compile(
    r"^\s*"
    r"(?:(?:static|inline|virtual|extern|const|volatile|unsigned|signed)\s+)*"  # optional qualifiers
    r"(?:[\w:*&<>]+\s+)+"   # return type tokens (at least one word + space)
    r"(\w+)\s*\("            # function name + opening paren
)

def _find_function_range(lines: list, start_hint: int) -> tuple:
    """Given a start line that's inside/at a function, find (start, end) of that function."""
    # Walk backwards to find the function signature
    start_idx = start_hint
    while start_idx > 0:
        if FUNC_PATTERN.match(lines[start_idx]) or _C_FUNC_PATTERN.match(lines[start_idx]):
            break
        start_idx -= 1

    is_python_style = FUNC_PATTERN.match(lines[start_idx]) if start_idx < len(lines) else False

    indent = (
        len(lines[start_idx]) - len(lines[start_idx].lstrip())
        if start_idx < len(lines) and (FUNC_PATTERN.match(lines[start_idx]) or _C_FUNC_PATTERN.match(lines[start_idx]))
        else 0
    )

    if is_python_style:
        # Indent-based end detection (Python, Ruby, etc.)
        end_idx = start_idx + 1
        while end_idx < len(lines):
            line = lines[end_idx]
            if (FUNC_PATTERN.match(line) or _C_FUNC_PATTERN.match(line)) and (len(line) - len(line.lstrip())) <= indent:
                break
            end_idx += 1
        return start_idx, end_idx

    # Brace-based end detection (C/C++/Java/Go/JS)
    end_idx = start_idx + 1
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
            end_idx = i + 1
            return start_idx, end_idx

    # Fallback: walk to next top-level function
    end_idx = start_idx + 1
    while end_idx < len(lines):
        line = lines[end_idx]
        if (FUNC_PATTERN.match(line) or _C_FUNC_PATTERN.match(line)) and (len(line) - len(line.lstrip())) <= indent:
            break
        end_idx += 1

    return start_idx, end_idx


def _find_call_sites(lines: list, func_name: str, func_start: int, func_end: int, context_window: int = 5) -> list:
    """
    Find lines outside (func_start, func_end) that call func_name.
    Returns list of (region_start, region_end) line ranges including surrounding context.
    """
    call_pattern = re.compile(rf'\b{re.escape(func_name)}\s*\(')
    regions = []
    for i, line in enumerate(lines):
        if func_start <= i < func_end:
            continue  # skip the function definition itself
        if call_pattern.search(line):
            # Find the enclosing function for this call site
            enc_start, enc_end = _find_function_range(lines, i)
            regions.append((enc_start, enc_end))

    # Merge strictly overlapping regions (not merely adjacent)
    if not regions:
        return []
    regions.sort()
    merged = [regions[0]]
    for s, e in regions[1:]:
        if s < merged[-1][1]:  # strict overlap only
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def extract_function(source: str, instruction: str, hint_blocks=None) -> tuple:
    """Extract the relevant function/class from source based on instruction.

    Now also includes call sites of the target function so the LLM can
    produce multi-site SEARCH/REPLACE blocks (e.g. fix a function signature
    AND update its callers).
    """
    lines = source.split('\n')
    target_idx = None
    func_name = None  # track the matched function name for call-site discovery

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
        func_name = next((g for g in m.groups() if g), None) if m else None
        if func_name:
            # Try Python/Go style first
            for i, l in enumerate(lines):
                if re.match(rf"\s*(def|func)\s+{re.escape(func_name)}\b", l, re.I):
                    target_idx = i
                    break
            # Try C/C++/Java style: type funcName(...)
            if target_idx is None:
                for i, l in enumerate(lines):
                    cm = _C_FUNC_PATTERN.match(l)
                    if cm and cm.group(1) == func_name:
                        target_idx = i
                        break
                    # Also match bare name at start like: void funcName(...)
                    if re.search(rf'\b{re.escape(func_name)}\s*\(', l) and '{' in source[sum(len(lines[j])+1 for j in range(i)):sum(len(lines[j])+1 for j in range(min(i+3, len(lines))))]:
                        # Verify this looks like a definition, not a call (has a type before it or is at top-level indent)
                        stripped = l.lstrip()
                        if not stripped.startswith('if') and not stripped.startswith('while') and not stripped.startswith('for') and not stripped.startswith('return'):
                            target_idx = i
                            break

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
        return "", 0, len(lines)

    # ── Call-site expansion ───────────────────────────────────────────
    # If we identified a function name, find its call sites outside the
    # function body and include them in the snippet so the LLM can
    # produce multi-site edits (signature + callers).
    if func_name:
        call_regions = _find_call_sites(lines, func_name, func_start, func_end)
    else:
        call_regions = []

    if call_regions:
        # Build a composite snippet: function definition + call-site regions
        # with clear section markers so the LLM knows what it's looking at.
        # NEVER merge across categories — keep func def and each call site separate.
        parts = []

        # 1. Always emit the function definition first
        parts.append(f"// ── FUNCTION DEFINITION (lines {func_start+1}-{func_end}) ──")
        parts.append('\n'.join(lines[func_start:func_end]))
        parts.append("")

        # 2. Emit each call-site region, skipping any that overlap with func def
        emitted_call_sites = 0
        for cs, ce in call_regions:
            # Skip if fully contained within function definition
            if cs >= func_start and ce <= func_end:
                continue
            # Trim overlap with function definition
            if cs < func_end and ce > func_end:
                cs = func_end
            parts.append(f"// ── CALL SITE (lines {cs+1}-{ce}) ──")
            parts.append('\n'.join(lines[cs:ce]))
            parts.append("")
            emitted_call_sites += 1

        if emitted_call_sites == 0:
            # All call sites were inside the function itself — no expansion needed
            return '\n'.join(lines[func_start:func_end]), func_start, func_end

        composite = '\n'.join(parts)
        overall_start = func_start
        overall_end = max(func_end, max(ce for _, ce in call_regions))
        logger.info("[EXTRACT] Expanded snippet: function '%s' (%d-%d) + %d call-site region(s)",
                    func_name, func_start+1, func_end, emitted_call_sites)
        return composite, overall_start, overall_end

    return '\n'.join(lines[func_start:func_end]), func_start, func_end


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


def count_top_level_functions(file_lines: list, ext: str = "") -> int:
    """Count top-level function definitions, excluding class declarations in C-family languages.

    For C/C++/Java/Go files, 'class' keyword should not count as a function.
    Only counts actual function definitions (def/func for Python/Go, type+name( for C-family).
    Also skips methods that are indented inside a class body — only counts truly top-level.
    """
    c_family_exts = {'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.java', '.cs', '.js', '.ts'}
    is_c_family = ext.lower() in c_family_exts

    count = 0
    for line in file_lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Skip blank lines
        if not stripped:
            continue

        # For C-family: only count top-level functions (indent <= 4 to allow some namespace indentation)
        # and skip 'class' keyword
        if is_c_family:
            if stripped.startswith('class '):
                continue  # class declarations are not functions
            if _C_FUNC_PATTERN.match(line) and indent <= 4:
                count += 1
        else:
            # Python/Go/Ruby: use FUNC_PATTERN but only at top-level (indent 0)
            if FUNC_PATTERN.match(line) and indent == 0:
                count += 1

    return count


def extract_error_lines_from_text(text: str) -> list:
    """Extract line numbers from error messages in user instructions.

    Handles common error report formats:
      - AddressSanitizer: 'solution.cpp:33:43'
      - GCC/Clang: 'file.cpp:10:5: error:'
      - Python traceback: 'File "foo.py", line 42'
      - Generic: 'line 33', 'Line 33:', 'at line 33'
      - User shorthand: 'Line 33:'
    """
    lines = set()

    # Standard compiler format: file:line:col
    for m in re.finditer(r'[\w./\\]+\.\w+:(\d+):\d+', text):
        lines.add(int(m.group(1)))

    # Python traceback: File "...", line N
    for m in re.finditer(r'[Ff]ile\s+"[^"]+",\s*line\s+(\d+)', text):
        lines.add(int(m.group(1)))

    # Generic "line N" patterns
    for m in re.finditer(r'\b[Ll]ine\s+(\d+)\b', text):
        lines.add(int(m.group(1)))

    return sorted(lines)


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


# --- Structural Validation (all languages) ---

def _strip_string_literals(line: str) -> str:
    """Remove string/char literals from a line to avoid false positives in brace counting."""
    result = []
    in_string = False
    string_char = None
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


def validate_code_structure(source: str, ext: str) -> list:
    """Validate structural integrity of code: brace balance, unreachable code, indentation.

    Works for all brace-based languages (Go, C, C++, Rust, JS, TS, Java, etc.).
    Returns list of issue dicts: {'type', 'msg', 'lineno', 'severity'}.
    """
    issues = []
    lines = source.split('\n')

    # --- 1. Brace balance ---
    if ext in BRACE_LANGUAGES:
        brace_stack = []  # stores (line_number, char) for each opening brace
        for i, line in enumerate(lines):
            stripped = _strip_string_literals(line)
            # Remove single-line comments
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
                            'msg': f'Extra closing brace at line {i + 1} — no matching opening brace',
                            'lineno': i + 1,
                            'severity': 'error',
                        })
        for open_line in brace_stack:
            issues.append({
                'type': 'unclosed_brace',
                'msg': f'Unclosed opening brace from line {open_line} — missing closing brace',
                'lineno': open_line,
                'severity': 'error',
            })

    # --- 2. Unreachable code detection ---
    _RETURN_KW = {'return', 'break', 'continue'}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        word = stripped.split('(')[0].split(' ')[0]
        if word not in _RETURN_KW:
            continue
        indent = len(line) - len(line.lstrip())
        # Look at the next non-empty, non-comment line
        for j in range(i + 1, min(i + 10, len(lines))):
            next_line = lines[j]
            next_stripped = next_line.strip()
            if not next_stripped or next_stripped.startswith('//') or next_stripped.startswith('#'):
                continue
            # Closing brace/bracket at same or lower indent is fine
            if next_stripped in ('}', ')', ']'):
                break
            # New function/class definition is fine
            if FUNC_PATTERN.match(next_line):
                break
            next_indent = len(next_line) - len(next_line.lstrip())
            if next_indent >= indent:
                issues.append({
                    'type': 'unreachable_code',
                    'msg': f'Unreachable code at line {j + 1} (after `{word}` at line {i + 1})',
                    'lineno': j + 1,
                    'severity': 'warning',
                })
            break

    # --- 3. Indentation consistency ---
    has_tabs = False
    has_spaces = False
    for line in lines:
        if not line or not line.strip():
            continue
        if line[0] == '\t':
            has_tabs = True
        elif line.startswith('  '):
            has_spaces = True
    if has_tabs and has_spaces:
        issues.append({
            'type': 'mixed_indentation',
            'msg': 'File mixes tabs and spaces for indentation',
            'lineno': 0,
            'severity': 'warning',
        })

    return issues


def build_structural_issue_note(issues: list, file_lines: list) -> str:
    """Build a formatted prompt note describing structural code issues for LLM repair."""
    if not issues:
        return ""

    note = "STRUCTURAL ISSUES DETECTED (fix ALL of them):\n\n"
    for issue in issues:
        lineno = issue.get('lineno', 0)
        if lineno > 0:
            start = max(0, lineno - 3)
            end = min(len(file_lines), lineno + 2)
            snippet = '\n'.join(f"{start+i+1}: {l}" for i, l in enumerate(file_lines[start:end]))
            note += (
                f"  {issue['type'].upper()}: {issue['msg']}\n"
                f"  CONTEXT:\n```\n{snippet}\n```\n\n"
            )
        else:
            note += f"  {issue['type'].upper()}: {issue['msg']}\n\n"

    note += (
        "Fix each structural issue with a SEARCH/REPLACE block.\n"
        "Common fixes: remove unreachable code after return, add/remove braces to balance, fix indentation.\n"
    )
    return note
