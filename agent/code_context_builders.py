import os
import re
import sys
import subprocess
import shutil
import tempfile
import logging
from typing import Optional, List, Dict
from operator import itemgetter

# On Windows, CREATE_NO_WINDOW prevents a console window flash and avoids
# interaction with asyncio's ProactorEventLoop IOCP handles from threads.
_SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

from brain.fast_search import fast_topic_search
from brain.config import (
    LLM_MODEL, CHROMA_DIR, EMBEDDING_MODEL,
    LANG_CHECK_CMD, LANG_LINT_CMD, LANG_FENCE, LANG_DOC_KEYWORDS,
    LANG_IRRELEVANT_DOC_KEYWORDS, LANG_QUERY_ENHANCEMENT,
    UNIVERSAL_DOC_KEYWORDS, UNIVERSAL_TOPICS, IMPLEMENT_KEYWORDS,
)
from .code_editing_helpers import parse_compiler_errors
from print_logger import get_logger

logger = get_logger("code_agent")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPILER / RUNTIME ERROR DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _go_compile_check(path: str, source: str, file_lines: list) -> str:
    """Run Go compilation check, handling standalone functions (no package declaration).

    LeetCode-style Go files often have no 'package main' — `go build` rejects them
    with 'expected package, found func'. We wrap in a temp package to catch REAL
    compile errors (type mismatches, undefined vars) without the false positive.
    """
    from .code_editing_helpers import build_runtime_error_note as build_formatted_note

    stripped = source.lstrip()
    needs_wrap = not stripped.startswith("package ")

    if not needs_wrap:
        # Normal Go file — direct compile check
        try:
            result = subprocess.run(
                ["go", "build", path],
                capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL,
                creationflags=_SUBPROCESS_FLAGS,
            )
            if result.returncode != 0 and result.stderr:
                errors = parse_compiler_errors(result.stderr)
                if errors:
                    logger.info("Compiler errors detected in %s", path)
                    return build_formatted_note(errors, file_lines)
        except Exception as e:
            logger.debug("Go compile check failed: %s", e)
        return ""

    # Standalone function — wrap in temp package for compilation
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapped = f"package main\n\n{source}\n\nfunc main() {{}}\n"
            tmp_path = os.path.join(tmpdir, "main.go")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(wrapped)

            result = subprocess.run(
                ["go", "build", "."],
                capture_output=True, text=True, timeout=10,
                cwd=tmpdir, stdin=subprocess.DEVNULL,
                creationflags=_SUBPROCESS_FLAGS,
            )
            if result.returncode != 0 and result.stderr:
                errors = parse_compiler_errors(result.stderr)
                # Adjust line numbers (offset by 2 for 'package main\n\n')
                for e in errors:
                    e["lineno"] = max(1, e["lineno"] - 2)
                # Filter out the dummy main() errors
                errors = [e for e in errors if "main redeclared" not in e.get("msg", "")]
                if errors:
                    logger.info("Compiler errors detected in %s (via temp wrap)", path)
                    return build_formatted_note(errors, file_lines)
    except Exception as e:
        logger.debug("Go compile check (wrapped) failed: %s", e)

    return ""


def build_runtime_error_notes(
    path: str,
    ext: str,
    is_python: bool,
    file_lines: list,
    file_source: str = "",
) -> str:
    """
    Build runtime/compiler error notes by running the code/compiler check.
    Returns error note string (empty if no errors).
    """
    # Go has special handling for standalone functions
    if ext == ".go":
        return _go_compile_check(path, file_source or "\n".join(file_lines), file_lines)

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
            timeout=10, stdin=subprocess.DEVNULL,
            creationflags=_SUBPROCESS_FLAGS,
        )
        if run_result.returncode != 0 and run_result.stderr:
            from .code_editing_helpers import build_runtime_error_note as build_formatted_note
            
            compiler_errors = parse_compiler_errors(run_result.stderr)
            if compiler_errors:
                runtime_error_note = build_formatted_note(compiler_errors, file_lines)
            else:
                runtime_error_note = (
                    "RUNTIME_ERRORS_DETECTED:\n"
                    f"```\n{run_result.stderr[:4000]}\n```\n\n"
                    "Output ONE SEARCH/REPLACE block per error fix."
                )
            logger.info("Compiler errors detected in %s", path)
    except subprocess.TimeoutExpired:
        runtime_error_note = "RUNTIME_WARNING: Build timed out — check for infinite loops.\n"
        logger.warning("File %s timed out during check", path)
    except Exception as e:
        logger.debug("Compiler check failed: %s", e)

    return runtime_error_note


# ═══════════════════════════════════════════════════════════════════════════════
# STATIC ANALYSIS / LINTING
# ═══════════════════════════════════════════════════════════════════════════════


def run_lint_checks(path: str, source: str, ext: str) -> str:
    """Run language-specific lint checks. Returns a prompt note or empty string.

    This complements the compiler/syntax checks with deeper analysis:
    - Go: `go vet` checks for suspicious constructs, printf format errors, etc.
    - Python: `ruff` checks for common mistakes (if installed).
    """
    lint_cmd_template = LANG_LINT_CMD.get(ext)
    if not lint_cmd_template:
        return ""

    # Check if lint binary is available
    binary = lint_cmd_template[0]
    if not shutil.which(binary):
        return ""

    try:
        if ext == ".go":
            # Go vet needs a valid package — wrap standalone functions
            stripped = source.lstrip()
            if not stripped.startswith("package "):
                with tempfile.TemporaryDirectory() as tmpdir:
                    wrapped = f"package main\n\n{source}\n\nfunc main() {{}}\n"
                    tmp_path = os.path.join(tmpdir, "main.go")
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        f.write(wrapped)
                    result = subprocess.run(
                        ["go", "vet", "."],
                        capture_output=True, text=True, timeout=10,
                        cwd=tmpdir, stdin=subprocess.DEVNULL,
                        creationflags=_SUBPROCESS_FLAGS,
                    )
            else:
                result = subprocess.run(
                    ["go", "vet", path],
                    capture_output=True, text=True, timeout=10,
                    stdin=subprocess.DEVNULL,
                    creationflags=_SUBPROCESS_FLAGS,
                )
        else:
            # Generic: run lint command on the file
            result = subprocess.run(
                lint_cmd_template + [path],
                capture_output=True, text=True, timeout=10,
                stdin=subprocess.DEVNULL,
                creationflags=_SUBPROCESS_FLAGS,
            )

        stderr_out = (result.stderr or "").strip()
        stdout_out = (result.stdout or "").strip()
        lint_output = stderr_out or stdout_out

        if result.returncode != 0 and lint_output:
            # Filter out noise from the dummy wrapper
            lines = [l for l in lint_output.split("\n")
                     if "main redeclared" not in l and l.strip()]
            if not lines:
                return ""
            # Adjust line numbers for Go standalone wrapping
            if ext == ".go" and not source.lstrip().startswith("package "):
                adjusted = []
                for l in lines:
                    m = re.match(r"^(.*?):(\d+):(\d+):\s*(.*)$", l)
                    if m:
                        lineno = max(1, int(m.group(2)) - 2)
                        adjusted.append(f"line {lineno}: {m.group(4)}")
                    else:
                        adjusted.append(l)
                lines = adjusted

            lint_text = "\n".join(lines[:10])  # cap at 10 issues
            logger.info("[LINT] Found %d issue(s) in %s", len(lines), path)
            return (
                f"LINT WARNINGS (from static analysis — these are NOT compiler errors, "
                f"but suspicious patterns that may indicate bugs):\n"
                f"```\n{lint_text}\n```\n\n"
                f"Fix any lint issues that relate to the task. Ignore unrelated warnings.\n"
            )
    except subprocess.TimeoutExpired:
        logger.debug("Lint check timed out for %s", path)
    except Exception as e:
        logger.debug("Lint check failed for %s: %s", path, e)

    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION CLARITY ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════


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

    # Implementation/solve tasks benefit from seeing example patterns in RAG
    if any(kw in instruction_lower for kw in IMPLEMENT_KEYWORDS):
        return 7  # Give implement tasks maximum RAG context for examples

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


# ═══════════════════════════════════════════════════════════════════════════════
# RAG RETRIEVAL & CHUNK GRADING
# ═══════════════════════════════════════════════════════════════════════════════


# ---------- Language pre-filter (cheap, no LLM) ----------


def _language_prefilter(chunks: list, file_ext: str) -> list:
    """Fast pre-filter: remove chunks from docs clearly written for a different language.
    
    Uses source filename matching — no LLM call needed.
    Language-agnostic docs (algorithm textbooks, math, clean code, PDFs about
    universal concepts) are always allowed through.
    If the file extension has no filter rules, all chunks pass through unchanged.
    """
    ext = file_ext.lower()
    irrelevant_keywords = LANG_IRRELEVANT_DOC_KEYWORDS.get(ext)
    if not irrelevant_keywords:
        return chunks  # No filter rules for this language — pass all through

    filtered = []
    for doc in chunks:
        source = doc.metadata.get('source', '').lower().replace(' ', '_').replace('-', '_')
        topic = doc.metadata.get('topic', '').lower()

        # Always allow docs whose topic is language-agnostic (algorithms, math, etc.)
        if topic in UNIVERSAL_TOPICS:
            filtered.append(doc)
            continue

        # Always allow docs matching universal keyword patterns (textbooks, etc.)
        if any(u in source for u in UNIVERSAL_DOC_KEYWORDS):
            filtered.append(doc)
            continue

        # Filter out docs matching irrelevant language keywords
        if any(kw.replace('-', '_') in source for kw in irrelevant_keywords):
            logger.info("[LANG_FILTER] Skipped wrong-language doc: %s", doc.metadata.get('source', 'Unknown'))
            continue
        
        filtered.append(doc)
    
    if len(filtered) < len(chunks):
        logger.info("[LANG_FILTER] Pre-filter: %d/%d chunks passed for %s file", len(filtered), len(chunks), ext)
    
    return filtered


# ---------- RRF for hybrid retrieval ----------

def _doc_key(doc) -> tuple:
    """Create a unique key for a doc (matching rag_brain.py implementation)."""
    meta = doc.metadata or {}
    source = meta.get("source")
    page = meta.get("page")
    if source is not None and page is not None:
        return (source, page)
    return (source, page, doc.page_content[:200])


def _assign_rank_scores(docs: list, k_param: int = 60) -> dict:
    """Assign RRF scores based on rank position."""
    return {
        _doc_key(doc): 1.0 / (k_param + rank)
        for rank, doc in enumerate(docs, start=1)
    }


def _merge_score_dicts(*score_dicts) -> dict:
    """Merge multiple score dictionaries by summing scores."""
    result = {}
    for scores in score_dicts:
        for key, score in scores.items():
            result[key] = result.get(key, 0) + score
    return result


def _rrf_fusion(semantic_docs: list, keyword_docs: list, k_param: int = 60) -> list:
    """Fuse semantic and keyword search results using Reciprocal Rank Fusion."""
    semantic_scores = _assign_rank_scores(semantic_docs, k_param)
    keyword_scores = _assign_rank_scores(keyword_docs, k_param)
    rrf_scores = _merge_score_dicts(semantic_scores, keyword_scores)
    
    # Build doc map from both sets
    doc_map = {_doc_key(doc): doc for doc in semantic_docs + keyword_docs}
    
    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores.items(), key=itemgetter(1), reverse=True)
    
    # Return docs in RRF order
    return [doc_map[key] for key, _ in sorted_keys]

def _grade_chunks(chunks: list, instruction: str, llm, file_ext: str = "") -> list:
    """Post-retrieval chunk grading with language-aware auto-pass.
    
    Chunks from documentation matching the target language are automatically kept
    (they already passed language pre-filter + cross-encoder reranking).
    Only chunks from ambiguous/unknown sources get LLM-graded.
    
    This avoids the problem where small LLMs (7b) can't distinguish 
    "useful C++ reference" from "irrelevant C++ reference" and mark everything IRRELEVANT.
    """
    if not chunks:
        return chunks
    
    source_keywords = LANG_DOC_KEYWORDS.get(file_ext.lower(), [])
    
    # Split chunks into auto-pass (correct language / universal) and needs-grading
    auto_pass = []
    needs_grading = []
    for doc in chunks:
        source = doc.metadata.get('source', '').lower().replace(' ', '_').replace('-', '_')
        topic = doc.metadata.get('topic', '').lower()
        
        # Auto-pass: correct language OR universal doc (by keyword or topic)
        if (source_keywords and any(kw.replace('-', '_') in source for kw in source_keywords)) or \
           any(u in source for u in UNIVERSAL_DOC_KEYWORDS) or \
           topic in UNIVERSAL_TOPICS:
            doc.metadata['graded'] = True
            doc.metadata['auto_pass'] = True
            auto_pass.append(doc)
        else:
            needs_grading.append(doc)
    
    if auto_pass:
        logger.info("[CHUNK_GRADING] Auto-passed %d/%d chunks (correct language / universal docs)", 
                    len(auto_pass), len(chunks))
    
    # Only LLM-grade the ambiguous chunks (wrong/unknown language sources)
    graded_ambiguous = []
    if needs_grading:
        target_lang = LANG_FENCE.get(file_ext.lower(), '')
        
        chunk_descriptions = []
        for i, doc in enumerate(needs_grading):
            source = doc.metadata.get('source', 'Unknown')
            preview = doc.page_content[:400].replace('\n', ' ')
            chunk_descriptions.append(f"  [{i+1}] {source}: {preview}...")
        
        chunks_block = "\n".join(chunk_descriptions)
        
        grade_prompt = (
            f"You are grading retrieved document chunks for usefulness to a coding task.\n\n"
            f"TASK: {instruction}\n"
            f"TARGET LANGUAGE: {target_lang or 'unknown'}\n\n"
            f"CHUNKS:\n{chunks_block}\n\n"
            f"For each chunk, output ONE line: [N] RELEVANT or [N] IRRELEVANT\n\n"
            f"RELEVANT = contains code, algorithms, data structures, or programming concepts "
            f"useful for the task. IRRELEVANT = completely unrelated topic or wrong language.\n"
            f"When in doubt, mark RELEVANT.\n\n"
            f"Output {len(needs_grading)} lines, one per chunk. No other text."
        )
        
        try:
            result = llm.invoke(grade_prompt).strip()
            lines = result.split('\n')
            
            for i, doc in enumerate(needs_grading):
                keep = True
                for line in lines:
                    if f"[{i+1}]" in line:
                        keep = "RELEVANT" in line.upper() and "IRRELEVANT" not in line.upper()
                        break
                if keep:
                    doc.metadata['graded'] = True
                    graded_ambiguous.append(doc)
                else:
                    logger.info("[CHUNK_GRADING] Filtered out ambiguous chunk %d: %s", i+1,
                              doc.metadata.get('source', 'Unknown'))
        except Exception as e:
            logger.warning("[CHUNK_GRADING] Grading failed, keeping ambiguous chunks: %s", e)
            graded_ambiguous = needs_grading
    
    final = auto_pass + graded_ambiguous
    logger.info("[CHUNK_GRADING] Final: %d/%d chunks kept (%d auto-pass, %d graded)", 
                len(final), len(chunks), len(auto_pass), len(graded_ambiguous))
    return final


def build_rag_context(instruction: str, use_rag: bool, rerank_method: str = "cross_encoder", max_chunks: int = 7, file_path: str = "", search_method: str = "both") -> tuple[str, list[str]]:
    """
    Build RAG context using RRF (Reciprocal Rank Fusion) hybrid retrieval.
    Combines BM25 (fast, keyword) and semantic (ChromaDB embeddings) search results.
    
    Returns (rag_context_string, citations_list).
    
    Args:
        instruction: The editing instruction to search for
        use_rag: Whether to enable RAG
        rerank_method: Reranking method - "cross_encoder" (default), "keyword", or "none"
        max_chunks: Maximum number of chunks to include (default 7)
        file_path: Path to the file being edited (for language-aware query enhancement)
        search_method: "bm25" (keyword only), "semantic" (ChromaDB only), or "both" (RRF fusion)
    """
    rag_context = ""
    citations: list[str] = []
    
    if not use_rag:
        return rag_context, citations
    
    # Language-aware query enhancement: inject language keywords to bias search
    enhanced_query = instruction
    ext = ""
    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in LANG_QUERY_ENHANCEMENT:
            enhanced_query = f"{LANG_QUERY_ENHANCEMENT[ext]} {instruction}"
            logger.info("[RAG] Enhanced query: %s", LANG_QUERY_ENHANCEMENT[ext])

    # Detect if this is an algorithm/DS problem to filter relevant PDFs
    _algo_keywords = [
        'algorithm', 'dynamic programming', 'dp', 'backtracking', 'greedy',
        'binary search', 'recursion', 'graph', 'tree', 'bfs', 'dfs',
        'sorting', 'heap', 'trie', 'hash', 'linked list', 'stack', 'queue',
        'sliding window', 'two pointer', 'divide and conquer', 'memoization',
        'knapsack', 'shortest path', 'topological', 'segment tree',
        'leetcode', 'competitive', 'time complexity', 'space complexity',
        'big o', 'data structure', 'algorithmic problem', 'coding challenge',
        'dijkstra', 'floyd warshall', 'bellman ford', 'union find', 'disjoint set',
        'fibonacci', 'longest common', 'edit distance', 'subarray', 'subsequence',
        'palindrome', 'anagram', 'permutation', 'combination', 'bit manipulation',
        'recurrence relation', 'catalan', 'branch and bound',
    ]
    _inst_lower = instruction.lower()
    algo_topic_filter = None
    if any(kw in _inst_lower for kw in _algo_keywords):
        algo_topic_filter = ["algorithms"]
        logger.info("[RAG] Detected algorithm-related instruction, adding topic filter")

        # Query expansion: use a single LLM call to extract precise algorithm
        # keywords from the problem description.  This turns vague descriptions
        # like "Fancy sequence" into "lazy evaluation modular inverse prefix sum"
        # so BM25 and semantic search can actually find the right textbook chunks.
        try:
            from brain.config import make_llm
            from brain.prompts import build_code_query_expansion_prompt
            expand_llm = make_llm(temperature=0.0)
            expansion_prompt = build_code_query_expansion_prompt(instruction)
            expanded_terms = expand_llm.invoke(expansion_prompt).strip()
            if expanded_terms and len(expanded_terms) < 300:
                enhanced_query = f"{enhanced_query} {expanded_terms}"
                logger.info("[RAG] Query expansion added: %s", expanded_terms)
        except Exception as e:
            logger.warning("[RAG] Query expansion failed (continuing without): %s", e)
    
    # --- BM25 Search (keyword-based) ---
    bm25_results = []
    if search_method in ("bm25", "both"):
        logger.info("[RAG] Running BM25 search...")
        try:
            results = fast_topic_search(enhanced_query, rerank_method=rerank_method, topic_filter=algo_topic_filter)
            if results:
                # Language pre-filter before limiting
                if file_path:
                    results = _language_prefilter(results, ext)
                
                # Don't grade BM25 yet; we'll grade after RRF fusion
                bm25_results = results[:min(max_chunks * 2, len(results))]
                logger.info("[RAG] BM25 returned %d results", len(bm25_results))
        except Exception as e:
            logger.warning("[RAG] BM25 search failed: %s", e)
    
    # --- ChromaDB Semantic Search ---
    chroma_results = []
    if search_method in ("semantic", "both"):
        logger.info("[RAG] Running ChromaDB semantic search...")
        try:
            from langchain_chroma import Chroma
            from langchain_ollama import OllamaEmbeddings
            
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
            
            # Truncate query to prevent embedding model overflow
            semantic_query = enhanced_query[:2000] if len(enhanced_query) > 2000 else enhanced_query
            # Get semantic results (more than max_chunks for RRF fusion)
            semantic_docs = vectorstore.similarity_search(semantic_query, k=max_chunks * 2)
            
            if semantic_docs and file_path:
                # Apply language pre-filter
                semantic_docs = _language_prefilter(semantic_docs, ext)
            
            chroma_results = semantic_docs[:min(max_chunks * 2, len(semantic_docs))]
            logger.info("[RAG] ChromaDB returned %d results", len(chroma_results))
        except Exception as e:
            logger.warning("[RAG] ChromaDB search failed: %s", e)
    
    # --- RRF Fusion (combine BM25 + semantic using Reciprocal Rank Fusion) ---
    final_results = []
    if search_method == "both" and bm25_results and chroma_results:
        logger.info("[RAG] Fusing BM25 and semantic results with RRF...")
        fused = _rrf_fusion(chroma_results, bm25_results, k_param=60)
        final_results = fused[:max_chunks * 2]
        logger.info("[RAG] RRF fused to %d results (before grading)", len(final_results))
    elif search_method == "both" and bm25_results:
        logger.info("[RAG] Using BM25 only (no semantic results)")
        final_results = bm25_results[:max_chunks * 2]
    elif search_method == "both" and chroma_results:
        logger.info("[RAG] Using ChromaDB only (no BM25 results)")
        final_results = chroma_results[:max_chunks * 2]
    elif search_method == "bm25":
        final_results = bm25_results[:max_chunks * 2]
    elif search_method == "semantic":
        final_results = chroma_results[:max_chunks * 2]
    
    # --- Cross-encoder rerank after fusion (better precision on top-k) ---
    if final_results and rerank_method == "cross_encoder":
        try:
            from brain.query_pipeline import rerank_documents
            from brain.config import CROSS_ENCODER_MODEL, RERANK_BATCH_SIZE
            pre_rerank = len(final_results)
            final_results = rerank_documents(
                docs=final_results,
                query=instruction,
                method="cross_encoder",
                cross_encoder_model=CROSS_ENCODER_MODEL,
                batch_size=RERANK_BATCH_SIZE,
            )
            logger.info("[RAG] Cross-encoder reranked %d → %d results", pre_rerank, len(final_results))
        except Exception as e:
            logger.warning("[RAG] Cross-encoder rerank failed (continuing without): %s", e)

    # --- Grade fused results for relevance ---
    if final_results:
        logger.info("[RAG] Grading %d fused chunks...", len(final_results))
        try:
            from brain.config import make_llm
            grade_llm = make_llm(temperature=0.0)
            file_ext = ext if file_path else ""
            graded_results = _grade_chunks(final_results, instruction, grade_llm, file_ext=file_ext)
            final_results = graded_results[:max_chunks]
        except Exception as e:
            logger.warning("[RAG] Chunk grading failed: %s", e)
            final_results = final_results[:max_chunks]
    
    # --- Format output ---
    if final_results:
        rag_context = "REFERENCE DOCS (Guide your edit — graded for relevance):\n"
        for i, doc in enumerate(final_results):
            score = doc.metadata.get('bm25_score', 'N/A')
            doc_source = doc.metadata.get('source', 'Unknown')
            score_str = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)
            rag_context += f"[{i+1}] {doc_source} (score:{score_str})\n{doc.page_content[:300]}...\n\n"
            citations.append(f"[{i+1}] {doc_source} (score:{score_str})")
        logger.info("[RAG] Final: %d chunks from hybrid retrieval", len(final_results))
    else:
        rag_context = "No relevant matches—use general principles.\n"
        logger.info("[RAG] No results from any search method")

    return rag_context, citations


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION & EDIT HISTORY
# ═══════════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════


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
    max_chunks: Optional[int] = None,
    search_method: str = "both"
) -> tuple[str, str, str, str, list[str]]:
    """
    Build all context types at once.
    Returns tuple of (runtime_error_note, rag_context, history_context, edit_history_text, citations).
    
    Args:
        rerank_method: Reranking method for RAG - "cross_encoder", "keyword", or "none"
        file_source: Full source code (used to assess clarity for RAG chunking)
        search_method: "bm25", "semantic", or "both" (default)
    """
    runtime_error_note = build_runtime_error_notes(path, ext, is_python, file_lines, file_source=file_source)
    
    # Assess instruction clarity and determine optimal RAG chunk limit (unless overridden)
    if max_chunks is None:
        max_chunks = assess_instruction_clarity(instruction, file_source) if file_source else 7
    rag_context, citations = build_rag_context(instruction, use_rag, rerank_method=rerank_method, max_chunks=max_chunks, file_path=path, search_method=search_method)
    
    history_context = build_history_context(session_chat_history)
    edit_history_text = build_edit_history(edit_log, path)
    
    return runtime_error_note, rag_context, history_context, edit_history_text, citations
