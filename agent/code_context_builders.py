"""
Context builders for code editing operations.
Handles construction of all context notes and information passed to the LLM.
"""

import sys
import subprocess
import logging
from typing import Optional, List, Dict

from brain.fast_search import fast_topic_search
from .code_editing_helpers import parse_compiler_errors

logger = logging.getLogger("code_agent")

LANG_CHECK_CMD = {
    ".go":  ["go", "build"],
    ".rs":  ["rustc", "--edition", "2021"],
    ".cpp": ["g++", "-fsyntax-only"],
    ".c":   ["gcc", "-fsyntax-only"],
    ".ts":  ["tsc", "--noEmit"],
}


def build_runtime_error_notes(
    path: str,
    ext: str,
    is_python: bool,
    file_lines: list
) -> str:
    """
    Build runtime/compiler error notes by running the code/compiler check.
    Returns error note string (empty if no errors).
    """
    runtime_error_note = ""
    
    if is_python:
        cmd = [sys.executable, path]
    else:
        base = LANG_CHECK_CMD.get(ext)
        cmd = base + [path] if base else None

    if not cmd:
        return runtime_error_note

    try:
        run_result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=10, stdin=subprocess.DEVNULL
        )
        if run_result.returncode != 0 and run_result.stderr:
            from .code_editing_helpers import build_runtime_error_note as build_formatted_note
            
            compiler_errors = parse_compiler_errors(run_result.stderr)
            if compiler_errors:
                runtime_error_note = build_formatted_note(compiler_errors, file_lines)
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

    return runtime_error_note


def assess_instruction_clarity(instruction: str, file_source: str) -> int:
    """
    Assess instruction clarity and return optimal max RAG chunks.
    
    Returns:
        0-3: Clear instruction (has comments, specific keywords)
        3-5: Medium clarity 
        5-7: Vague instruction (needs full context)
    """
    instruction_lower = instruction.lower()
    file_lower = file_source.lower()

    # If instruction defers to file comments as the spec, keep RAG minimal —
    # a couple of examples for style are fine, but too many chunks drown the comments out
    comment_deference_phrases = [
        "based on comment", "use the comment", "comments are available",
        "from the comment", "using the comment", "follow the comment",
        "as per comment", "implement from comment"
    ]
    if any(p in instruction_lower for p in comment_deference_phrases):
        return 3

    clarity_score = 0
    
    # Check for clarity indicators
    clear_keywords = {
        "implement", "fill", "complete", "body", "function", 
        "debug", "fix error", "fix bug", "solve", "based on comment"
    }
    
    if any(kw in instruction_lower for kw in clear_keywords):
        clarity_score += 2
    
    # Check for code comments in file (problem spec embedded)
    comment_ratio = (file_lower.count('//') + file_lower.count('#')) / max(1, len(file_source.split('\n')))
    if comment_ratio > 0.1:  # >10% of lines are comments
        clarity_score += 2
    
    # Check instruction length
    word_count = len(instruction.split())
    if word_count < 10:
        clarity_score += 1  # Short/specific instruction
    elif word_count > 30:
        clarity_score -= 1  # Long/vague instruction
    
    # Map score to chunk count
    if clarity_score >= 3:
        return 3  # Clear: 0-3 chunks
    elif clarity_score >= 1:
        return 5  # Medium: 3-5 chunks
    else:
        return 7  # Vague: 5-7 chunks


def build_rag_context(instruction: str, use_rag: bool, rerank_method: str = "cross_encoder", max_chunks: int = 7) -> str:
    """
    Build RAG (Retrieval-Augmented Generation) context from code search.
    Returns reference docs string (empty if use_rag is False or search fails).
    
    Args:
        instruction: The editing instruction to search for
        use_rag: Whether to enable RAG
        rerank_method: Reranking method - "cross_encoder" (default), "keyword", or "none"
        max_chunks: Maximum number of chunks to include (default 7, may be limited by clarity)
    """
    rag_context = ""
    
    if not use_rag:
        return rag_context
    
    print("\n[FAST_SEARCH] BM25 for code context...")
    try:
        results = fast_topic_search(instruction, rerank_method=rerank_method)
        if results:
            rag_context = "REFERENCE DOCS (Guide your edit):\n"
            limited_results = results[:min(max_chunks, len(results))]
            for i, doc in enumerate(limited_results):
                score = doc.metadata.get('bm25_score', 'N/A')
                doc_source = doc.metadata.get('source', 'Unknown')
                score_str = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)
                rag_context += f"[{i+1}] {doc_source} (BM25:{score_str})\n{doc.page_content[:300]}...\n\n"
            print(f"Found {len(results)} chunks, using {len(limited_results)}/{max_chunks}!")
        else:
            rag_context = "No exact matches—use general principles.\n"
    except Exception as e:
        logger.debug(f"RAG error: {e}")
        print(f"RAG error: {e}")

    return rag_context


def build_history_context(session_chat_history: Optional[List[Dict[str, str]]]) -> str:
    """
    Build conversation history context from recent chat messages.
    Returns history string (empty if no session history).
    """
    history_context = ""
    
    if not session_chat_history:
        return history_context
    
    recent = session_chat_history[-6:]
    history_context = "\nCONVERSATION CONTEXT (follow-up edit):\n"
    for msg in recent:
        role = "USER: " if msg.get("role", "").lower() in ["user", "human"] else "AGENT: "
        content = msg.get('content', '')
        content = (content[:200] + "...") if len(content) > 200 else content
        history_context += f"{role}{content}\n"

    return history_context


def build_edit_history(edit_log: dict, path: str) -> str:
    """
    Build recent edits history from edit log.
    Returns edit history string (empty if no prior edits for this file).
    """
    edit_history_text = ""
    
    try:
        recent_edits = edit_log.get(path, [])[-3:]
        if recent_edits:
            edit_history_text = "RECENT EDITS (most recent last):\n"
            for e in recent_edits:
                edit_history_text += f"- {e.get('timestamp','')}: {e.get('instruction','')}\nDiff:\n{e.get('diff','')[:1000]}\n\n"
    except Exception as e:
        logger.debug(f"Failed to build edit history: {e}")

    return edit_history_text


def build_all_contexts(
    path: str,
    ext: str,
    is_python: bool,
    file_lines: list,
    instruction: str,
    use_rag: bool,
    session_chat_history: Optional[List[Dict[str, str]]],
    edit_log: dict,
    rerank_method: str = "cross_encoder",
    file_source: str = "",
    max_chunks: Optional[int] = None
) -> tuple[str, str, str, str]:
    """
    Build all context types at once.
    Returns tuple of (runtime_error_note, rag_context, history_context, edit_history_text).
    
    Args:
        rerank_method: Reranking method for RAG - "cross_encoder", "keyword", or "none"
        file_source: Full source code (used to assess clarity for RAG chunking)
    """
    runtime_error_note = build_runtime_error_notes(path, ext, is_python, file_lines)
    
    # Assess instruction clarity and determine optimal RAG chunk limit (unless overridden)
    if max_chunks is None:
        max_chunks = assess_instruction_clarity(instruction, file_source) if file_source else 7
    rag_context = build_rag_context(instruction, use_rag, rerank_method=rerank_method, max_chunks=max_chunks)
    
    history_context = build_history_context(session_chat_history)
    edit_history_text = build_edit_history(edit_log, path)
    
    return runtime_error_note, rag_context, history_context, edit_history_text
