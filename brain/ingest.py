from rank_bm25 import BM25Okapi
import json, pickle
from typing import Any, List, Dict
from collections import Counter
import re
import os
import shutil

from datetime import datetime
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from .utils import pipe
from .config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, INDEX_META_PATH, LLM_MODEL
from .pdf_utils import load_pdfs


# ─── Topic Classification ─────────────────────────────────────────────────────
# Maps filename patterns → topic labels so retrieval can filter by domain.

_TOPIC_PATTERNS: list[tuple[list[str], str]] = [
    (["algorithm", "algo", "data_structure", "data-structure", "sorting",
      "dynamic_programming", "dp", "graph", "tree", "leetcode", "competitive",
      "binary_search", "backtracking", "greedy", "recursion", "heap", "trie",
      "hash", "linked_list", "stack", "queue", "bfs", "dfs", "dijkstra",
      "bellman", "kruskal", "prim", "union_find", "segment_tree", "fenwick",
      "topological", "sliding_window", "two_pointer", "bit_manipulation",
      "divide_and_conquer", "memoization", "tabulation", "knapsack",
      "shortest_path", "minimum_spanning", "network_flow"], "algorithms"),
    (["clean_code", "clean-code", "refactoring", "design_pattern", "design-pattern",
      "solid", "legacy", "code_complete", "pragmatic", "architecture",
      "software_design", "gang_of_four", "gof", "head_first_design",
      "dependency_injection", "microservice", "domain_driven"], "clean-code"),
    (["calculus", "linear_algebra", "math", "statistics", "probability",
      "discrete_math", "number_theory", "combinatorics", "geometry",
      "numerical_methods", "optimization"], "mathematics"),
    (["machine_learning", "ml", "deep_learning", "neural", "ai",
      "natural_language", "nlp", "transformer", "reinforcement_learning",
      "computer_vision", "tensorflow", "pytorch", "scikit"], "ml"),
    (["python", "cpython", "django", "flask", "fastapi", "pandas",
      "numpy", "asyncio"], "python"),
    (["cpp", "c++", "cplusplus", "stroustrup", "effective_modern",
      "concurrency_in_action", "stl", "boost", "cmake"], "cpp"),
    (["golang", "go_programming", "go_in_action"], "go"),
    (["rust", "rustacean", "cargo", "tokio", "async_rust"], "rust"),
    (["javascript", "typescript", "angular", "react", "node", "vue",
      "webpack", "nextjs", "express", "deno"], "web"),
    (["java", "spring", "jvm", "kotlin", "maven", "gradle",
      "effective_java", "concurrency_java"], "java"),
    (["blender", "3d_model", "game_engine", "unity", "unreal",
      "opengl", "vulkan", "shader", "graphics"], "creative"),
    (["database", "sql", "nosql", "postgres", "mysql", "mongodb",
      "redis", "elasticsearch", "indexing"], "database"),
    (["network", "tcp", "http", "socket", "protocol", "dns",
      "distributed_system", "rpc", "grpc"], "networking"),
    (["security", "cryptography", "encryption", "authentication",
      "oauth", "penetration", "owasp", "vulnerability"], "security"),
    (["operating_system", "os", "kernel", "linux", "systems_programming",
      "memory_management", "threading", "concurrency"], "systems"),
]


def _classify_document_topic(source_path: str) -> str:
    """Classify a document's topic from its filename/path.

    Returns a short label like 'algorithms', 'clean-code', 'python', etc.
    Falls back to 'general' if no pattern matches.
    """
    name = Path(source_path).stem.lower().replace("-", "_").replace(" ", "_")
    for keywords, topic in _TOPIC_PATTERNS:
        if any(kw in name for kw in keywords):
            return topic
    return "general"


def _enrich_metadata(doc) -> dict:
    """Build clean metadata: keep source filename + add topic classification
    and algorithm sub-category for improved retrieval."""
    meta = dict(doc.metadata or {})
    raw_source = meta.get("source") or meta.get("file_path") or ""
    # Keep just the filename (no full path for privacy)
    source_name = Path(raw_source).name if raw_source else "unknown"
    topic = _classify_document_topic(raw_source)

    # Detect algorithm sub-category from chunk content for finer-grained retrieval
    algo_category = ""
    if topic == "algorithms":
        content_lower = (doc.page_content or "").lower()
        _ALGO_CATEGORIES = [
            (["binary search", "bisect", "lower_bound", "upper_bound"], "binary-search"),
            (["dynamic programming", "memoization", "tabulation", "subproblem", "knapsack"], "dp"),
            (["two pointer", "two-pointer", "left pointer", "right pointer"], "two-pointer"),
            (["sliding window", "window size", "shrink window"], "sliding-window"),
            (["breadth first", "bfs", "queue"], "bfs"),
            (["depth first", "dfs", "backtrack"], "dfs-backtracking"),
            (["greedy", "locally optimal"], "greedy"),
            (["divide and conquer", "merge sort", "quick sort"], "divide-conquer"),
            (["union find", "disjoint set", "union-find"], "union-find"),
            (["topological sort", "topological order", "dag"], "topological-sort"),
            (["trie", "prefix tree"], "trie"),
            (["monotonic stack", "monotone stack", "next greater"], "monotonic-stack"),
            (["segment tree", "fenwick", "binary indexed"], "segment-tree"),
            (["graph", "adjacency", "shortest path", "dijkstra", "bellman"], "graph"),
        ]
        for keywords, cat in _ALGO_CATEGORIES:
            if any(kw in content_lower for kw in keywords):
                algo_category = cat
                break

    result = {
        k: v for k, v in meta.items()
        if k not in ("file_path",)  # drop file_path, keep everything else
    } | {"source": source_name, "topic": topic}
    if algo_category:
        result["algo_category"] = algo_category

    # Detect editorial / solution-explanation content
    name_lower = Path(raw_source).stem.lower().replace("-", "_").replace(" ", "_")
    if any(tag in name_lower for tag in ("editorial", "solution_explanation", "approach", "walkthrough")):
        result["doc_type"] = "editorial"
    return result


TOPIC_MAP_PATH = "cache/topic_map.json"

def _write_index_metadata(doc_count: int) -> None:
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "data_dir": DATA_DIR,
        "doc_count": doc_count,
        "created_at": datetime.now().isoformat() + "Z",
    }
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

_STOP_WORDS = {
    # ── Articles / determiners ──
    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'every', 'each',
    'some', 'any', 'all', 'both', 'either', 'neither', 'no', 'much',
    'many', 'few', 'little', 'several', 'enough', 'most', 'other',
    'another', 'such', 'own',
    # ── Pronouns ──
    'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'who', 'whom', 'whose', 'which', 'what', 'whatever', 'whoever',
    'one', 'ones', 'something', 'anything', 'nothing', 'everything',
    'someone', 'anyone', 'nobody', 'everybody', 'everyone', 'somewhere',
    'anywhere', 'nowhere', 'anybody',
    # ── Prepositions ──
    'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'over', 'about', 'against', 'without', 'within',
    'along', 'across', 'behind', 'beyond', 'among', 'around', 'toward',
    'towards', 'upon', 'onto', 'throughout', 'beside', 'besides',
    'despite', 'except', 'since', 'until', 'unless', 'per', 'via',
    # ── Conjunctions ──
    'and', 'or', 'but', 'nor', 'yet', 'so', 'because', 'although',
    'though', 'while', 'whereas', 'whether', 'however', 'therefore',
    'moreover', 'furthermore', 'nevertheless', 'meanwhile', 'otherwise',
    'hence', 'thus', 'still', 'also', 'too', 'then', 'else',
    # ── Be / have / do auxiliaries ──
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'done',
    # ── Modal verbs ──
    'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might',
    'must', 'need', 'dare', 'ought',
    # ── Common verbs / verb forms ──
    'get', 'got', 'gets', 'getting', 'make', 'makes', 'made', 'making',
    'go', 'goes', 'going', 'went', 'gone', 'come', 'came', 'comes',
    'take', 'takes', 'took', 'taken', 'give', 'gave', 'given',
    'say', 'said', 'says', 'see', 'saw', 'seen', 'know', 'knew', 'known',
    'think', 'thought', 'let', 'put', 'keep', 'kept', 'tell', 'told',
    'find', 'found', 'want', 'seem', 'show', 'shown', 'try', 'tried',
    'leave', 'left', 'call', 'called', 'ask', 'asked',
    'look', 'looked', 'looking', 'use', 'used', 'using', 'work', 'worked',
    # ── Adverbs / filler ──
    'not', 'just', 'only', 'very', 'really', 'quite', 'rather', 'already',
    'always', 'never', 'often', 'sometimes', 'usually', 'perhaps', 'maybe',
    'anyway', 'actually', 'basically', 'simply', 'certainly', 'clearly',
    'probably', 'possibly', 'especially', 'particularly', 'generally',
    'typically', 'essentially', 'definitely', 'obviously', 'apparently',
    'exactly', 'nearly', 'almost', 'well', 'even', 'ever', 'here', 'there',
    'now', 'then', 'when', 'where', 'how', 'why', 'again', 'further',
    'once', 'soon', 'later', 'early', 'far', 'long', 'way', 'back',
    'down', 'up', 'out', 'off', 'away', 'together', 'apart',
    # ── Common adjectives / filler adjectives ──
    'new', 'old', 'good', 'bad', 'great', 'small', 'large', 'big',
    'high', 'low', 'same', 'different', 'important', 'main', 'major',
    'first', 'last', 'next', 'previous', 'following', 'possible',
    'available', 'able', 'likely', 'certain', 'sure', 'true', 'right',
    'real', 'full', 'whole', 'common', 'simple', 'easy', 'hard',
    'less', 'least', 'more', 'most', 'than', 'better', 'best', 'worse',
    # ── Textbook / document filler ──
    'example', 'figure', 'chapter', 'section', 'page', 'note', 'notes',
    'table', 'see', 'shown', 'given', 'used', 'using', 'like',
    'consider', 'case', 'cases', 'called', 'known', 'means', 'way',
    'number', 'order', 'part', 'parts', 'form', 'point', 'end',
    'time', 'set', 'problem', 'result', 'results', 'step', 'following',
    'two', 'three', 'four', 'five', 'second', 'third',
    'may', 'must', 'often', 'etc', 'e.g', 'i.e',
}


def _extract_keywords_for_chunks(chunks: List[Any], max_keywords: int = 500) -> Dict[str, int]:
    """Extract top keywords from a list of chunks (used per-PDF or per-group).
    Returns up to max_keywords, skipping noise. Fewer if the PDF is small."""
    cleaned = []
    for doc in chunks:
        for raw in doc.page_content.lower().split():
            token = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', raw)
            if token and token not in _STOP_WORDS and len(token) > 2 and not token.isdigit():
                cleaned.append(token)

    word_freq = Counter(cleaned).most_common(max_keywords)
    return dict(word_freq)


def extract_keywords_per_pdf(docs: List[Any]) -> Dict[str, Dict[str, int]]:
    """Extract keywords PER PDF source file instead of globally.

    Returns a dict: { "filename.pdf": { "keyword": freq, ... }, ... }
    Each PDF gets up to 500 keywords from its own chunks only.
    """
    # Group chunks by source PDF
    by_source: Dict[str, List[Any]] = {}
    for doc in docs:
        md = getattr(doc, 'metadata', {})
        source = md.get('source') or md.get('file_path') or 'unknown'
        source_name = Path(source).name
        by_source.setdefault(source_name, []).append(doc)

    per_pdf_keywords: Dict[str, Dict[str, int]] = {}
    for source_name, source_chunks in by_source.items():
        keywords = _extract_keywords_for_chunks(source_chunks, max_keywords=500)
        per_pdf_keywords[source_name] = keywords

    return per_pdf_keywords


def extract_keywords_from_corpus(docs: List[Any]) -> Dict[str, int]:
    """Legacy: global keyword extraction. Preserved for backward compatibility.
    Merges per-PDF keywords, keeping the top 500 globally."""
    per_pdf = extract_keywords_per_pdf(docs)
    merged: Counter = Counter()
    for pdf_keywords in per_pdf.values():
        merged.update(pdf_keywords)
    return dict(merged.most_common(500))

CHROMA_BATCH_SIZE = 500  # embed in batches to avoid OOM
CHROMA_PROGRESS_PATH = Path("cache/chroma_progress.json")

# ── Shared log sink (read by api.py for frontend streaming) ──────────────────
_reingest_log: list[str] = []

def _log(msg: str):
    """Append msg to the shared log list AND print to stdout."""
    _reingest_log.append(msg)
    print(msg, flush=True)


def _save_chroma_progress(batch_idx: int, total_batches: int):
    with open(CHROMA_PROGRESS_PATH, 'w') as f:
        json.dump({"batch_done": batch_idx, "total": total_batches}, f)


def _load_chroma_progress() -> int:
    """Return the 0-based index of the NEXT batch to process (resume point)."""
    if CHROMA_PROGRESS_PATH.exists():
        try:
            with open(CHROMA_PROGRESS_PATH) as f:
                data = json.load(f)
            return data.get("batch_done", 0)
        except Exception:
            pass
    return 0


def _clear_chroma_progress():
    if CHROMA_PROGRESS_PATH.exists():
        CHROMA_PROGRESS_PATH.unlink()


def _add_to_chroma_batched(docs, embeddings, persist_directory, resume: bool = False):
    """Add documents to Chroma in batches. Resumable: skips already-done batches."""
    total = len(docs)
    if total == 0:
        return
    total_batches = (total - 1) // CHROMA_BATCH_SIZE + 1
    start_batch = _load_chroma_progress() if resume else 0
    if start_batch > 0:
        _log(f"Resuming Chroma embedding from batch {start_batch + 1}/{total_batches}")
    for batch_idx in range(start_batch, total_batches):
        i = batch_idx * CHROMA_BATCH_SIZE
        batch = docs[i:i + CHROMA_BATCH_SIZE]
        try:
            Chroma.from_documents(
                batch, embedding=embeddings, persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"},
            )
            _save_chroma_progress(batch_idx + 1, total_batches)
            _log(f"Chroma batch {batch_idx + 1}/{total_batches}: embedded {len(batch)} chunks")
        except Exception as e:
            _save_chroma_progress(batch_idx, total_batches)
            raise RuntimeError(
                f"Chroma batch {batch_idx + 1}/{total_batches} failed: {e}  "
                f"(resume will continue from batch {batch_idx + 1})"
            ) from e
    _clear_chroma_progress()


def ingest_docs(force: bool = False):
    """Ingest PDFs from DATA_DIR.

    If force is True, rebuilds from scratch (clears Chroma). Otherwise it will
    load any existing cached splits and only process new PDF files, appending
    their chunks to the existing index and adding new vectors to Chroma.

    Chroma embedding is resumable: if a previous force-rebuild crashed mid-batch,
    calling this again with force=True will skip already-embedded batches.
    """
    _reingest_log.clear()
    splits_path = Path("cache/splits.pkl")
    existing_splits = []

    # On force rebuild, check if we have a partially-completed Chroma run
    resuming = force and _load_chroma_progress() > 0 and os.path.exists(CHROMA_DIR)

    if not force and splits_path.exists():
        try:
            with splits_path.open('rb') as f:
                existing_splits = pickle.load(f)
            _log(f"Loaded {len(existing_splits)} existing chunks from cache")
        except Exception as e:
            _log(f"Failed to load existing splits.pkl: {e}")
            existing_splits = []

    # If resuming we already have splits.pkl and enriched docs — just redo Chroma
    if resuming and splits_path.exists():
        _log("Detected interrupted Chroma build -- resuming where we left off")
        with splits_path.open('rb') as f:
            combined_splits = pickle.load(f)
        _log(f"Loaded {len(combined_splits)} cached chunks")

        # rebuild enriched docs list (same order as original build)
        cleaned_all = []
        for doc in combined_splits:
            clean_doc = doc.copy()
            clean_doc.metadata = _enrich_metadata(doc)
            cleaned_all.append(clean_doc)

        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        _log(f"Resuming Chroma embedding ({len(cleaned_all)} chunks, batch size {CHROMA_BATCH_SIZE})...")
        _add_to_chroma_batched(cleaned_all, embeddings, CHROMA_DIR, resume=True)
        _log(f"Chroma DB complete: {len(cleaned_all)} docs at {CHROMA_DIR}")

        # Load topic map from cache
        top_keywords = {}
        if os.path.exists(TOPIC_MAP_PATH):
            with open(TOPIC_MAP_PATH) as f:
                top_keywords = json.load(f)

        _write_index_metadata(len(combined_splits))
        documents = load_pdfs(DATA_DIR)
        _log(f"Resume complete: {len(combined_splits)} chunks, {len(top_keywords)} keywords")
        return documents, top_keywords

    # ── Full run (non-resume path) ───────────────────────────────────────────

    # Load PDFs one-by-one so a bad file doesn't kill the whole batch
    pdf_dir = Path(DATA_DIR)
    pdf_files = sorted(pdf_dir.glob("**/*.pdf"))
    _log(f"Found {len(pdf_files)} PDF files in {DATA_DIR}")

    all_docs = []
    failed_files = []
    for idx, pdf in enumerate(pdf_files, 1):
        try:
            loader = PyPDFLoader(str(pdf))
            pages = loader.load()
            all_docs.extend(pages)
            _log(f"PDF {idx}/{len(pdf_files)}: {pdf.name} ({len(pages)} pages)")
        except Exception as e:
            failed_files.append((str(pdf.name), str(e)))
            _log(f"SKIP PDF {idx}/{len(pdf_files)}: {pdf.name} -- {e}")

    if failed_files:
        _log(f"WARNING: {len(failed_files)} PDFs failed to load, continuing with {len(all_docs)} pages")
    else:
        _log(f"Loaded {len(all_docs)} pages from {len(pdf_files)} PDFs")

    # determine which source files are already present
    existing_sources = set()
    for doc in existing_splits:
        md = getattr(doc, 'metadata', {})
        src = md.get('source') or md.get('file_path')
        if src:
            existing_sources.add(Path(src).resolve())

    # identify new docs (by source path) -- deduplicate path lists
    new_docs = []
    new_path_set: set[str] = set()
    skipped_path_set: set[str] = set()
    for doc in all_docs:
        md = getattr(doc, 'metadata', {})
        src = md.get('source') or md.get('file_path')
        resolved = Path(src).resolve() if src else None
        if resolved and resolved in existing_sources:
            skipped_path_set.add(str(resolved))
            continue
        new_docs.append(doc)
        if resolved:
            new_path_set.add(str(resolved))

    if new_path_set:
        _log(f"New files to ingest ({len(new_path_set)})")
        for p in sorted(new_path_set):
            _log(f"  - {Path(p).name}")
    if skipped_path_set:
        _log(f"Skipped {len(skipped_path_set)} already-ingested files")

    if not new_docs and existing_splits and not force:
        _log("No new documents to ingest; using existing index.")
        topic_map = {}
        if os.path.exists(TOPIC_MAP_PATH):
            with open(TOPIC_MAP_PATH, 'r') as f:
                topic_map = json.load(f)
        documents = load_pdfs(DATA_DIR)
        return documents, topic_map

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    new_splits = []
    if new_docs:
        new_splits = splitter.split_documents(new_docs)
    combined_splits = existing_splits + new_splits

    _log(f"Split into {len(new_splits)} new chunks (total {len(combined_splits)})")

    _log("Extracting keywords per PDF...")
    per_pdf_keywords = extract_keywords_per_pdf(combined_splits)
    total_kw = sum(len(v) for v in per_pdf_keywords.values())
    _log(f"Extracted keywords for {len(per_pdf_keywords)} PDFs ({total_kw} total across all)")

    # Save per-PDF keyword map (the main index now)
    per_pdf_map_path = Path("cache/per_pdf_keywords.json")
    with open(per_pdf_map_path, 'w') as f:
        json.dump(per_pdf_keywords, f, indent=2)
    _log(f"Saved per-PDF keyword map to {per_pdf_map_path}")

    # Also save legacy global map for backward compatibility
    top_keywords = extract_keywords_from_corpus(combined_splits)

    # rebuild BM25 from combined splits
    _log("Building BM25 index...")
    tokenized_docs = [doc.page_content.lower().split() for doc in combined_splits]
    bm25_idx = BM25Okapi(tokenized_docs)
    pickle.dump(bm25_idx, Path("cache/global_bm25.pkl").open('wb'))
    _log(f"Built BM25 index: {len(combined_splits)} chunks")

    with open(TOPIC_MAP_PATH, 'w') as f:
        json.dump(top_keywords, f, indent=2)

    # Cache splits BEFORE Chroma so resume can reload them
    with Path("cache/splits.pkl").open('wb') as f:
        pickle.dump(combined_splits, f)
    _log("Saved splits.pkl (BM25 + resume checkpoint)")

    _log("Starting Chroma vector DB embedding...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # prepare only the new splits for insertion -- enrich metadata with topic
    cleaned_new = []
    for doc in new_splits:
        clean_doc = doc.copy()
        clean_doc.metadata = _enrich_metadata(doc)
        cleaned_new.append(clean_doc)

    if os.path.exists(CHROMA_DIR) and not force:
        if cleaned_new:
            _add_to_chroma_batched(cleaned_new, embeddings, CHROMA_DIR)
            _log(f"Appended {len(cleaned_new)} chunks to Chroma")
        else:
            _log("No new documents to add to Chroma.")
    else:
        # fresh build (either no CHROMA_DIR or force requested)
        if os.path.exists(CHROMA_DIR):
            _log(f"Clearing old Chroma database...")
            shutil.rmtree(CHROMA_DIR)
        _clear_chroma_progress()
        cleaned_all = []
        for doc in combined_splits:
            clean_doc = doc.copy()
            clean_doc.metadata = _enrich_metadata(doc)
            cleaned_all.append(clean_doc)
        _log(f"Embedding {len(cleaned_all)} chunks into Chroma (batch size {CHROMA_BATCH_SIZE})...")
        _add_to_chroma_batched(cleaned_all, embeddings, CHROMA_DIR)
        _log(f"Chroma DB built: {len(cleaned_all)} docs")

    _write_index_metadata(len(combined_splits))

    documents = load_pdfs(DATA_DIR)

    _log(f"Ingest complete: {len(combined_splits)} chunks, {len(documents)} PDFs, {len(top_keywords)} keywords")
    return documents, top_keywords


def ingest_file(file_path: str):
    """Ingest a single PDF file (path relative to DATA_DIR or absolute).

    Returns (documents, topic_map) similar to ingest_docs, or raises on error.
    """
    p = Path(file_path)
    if not p.is_absolute():
        p = Path(DATA_DIR) / p

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    # load existing splits if available
    splits_path = Path("cache/splits.pkl")
    existing_splits = []
    if splits_path.exists():
        try:
            with splits_path.open('rb') as f:
                existing_splits = pickle.load(f)
        except Exception:
            existing_splits = []

    # check if file already ingested by comparing source
    existing_sources = set()
    for doc in existing_splits:
        md = getattr(doc, 'metadata', {})
        src = md.get('source') or md.get('file_path')
        if src:
            try:
                existing_sources.add(Path(src).resolve())
            except Exception:
                pass

    if p.resolve() in existing_sources:
        return None  # indicate already exists

    # split the single document
    try:
        loader = PyPDFLoader(str(p))
        pages = loader.load()
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    new_splits = splitter.split_documents(pages)

    combined_splits = existing_splits + new_splits

    # update BM25
    tokenized_docs = [doc.page_content.lower().split() for doc in combined_splits]
    bm25_idx = BM25Okapi(tokenized_docs)
    pickle.dump(bm25_idx, Path("cache/global_bm25.pkl").open('wb'))

    # update Chroma with new splits only
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    cleaned_new = []
    for doc in new_splits:
        clean_doc = doc.copy()
        clean_doc.metadata = _enrich_metadata(doc)
        cleaned_new.append(clean_doc)

    if os.path.exists(CHROMA_DIR):
        if cleaned_new:
            Chroma.from_documents(
                cleaned_new, embedding=embeddings, persist_directory=CHROMA_DIR,
                collection_metadata={"hnsw:space": "cosine"},
            )
    else:
        cleaned_all = []
        for doc in combined_splits:
            clean_doc = doc.copy()
            clean_doc.metadata = _enrich_metadata(doc)
            cleaned_all.append(clean_doc)
        Chroma.from_documents(
            cleaned_all, embedding=embeddings, persist_directory=CHROMA_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )

    # cache updated splits
    with Path("cache/splits.pkl").open('wb') as f:
        pickle.dump(combined_splits, f)

    _write_index_metadata(len(combined_splits))

    documents = load_pdfs(DATA_DIR)

    top_keywords = extract_keywords_from_corpus(combined_splits)
    with open(TOPIC_MAP_PATH, 'w') as f:
        json.dump(top_keywords, f, indent=2)

    # Update per-PDF keyword map
    per_pdf_keywords = extract_keywords_per_pdf(combined_splits)
    per_pdf_map_path = Path("cache/per_pdf_keywords.json")
    with open(per_pdf_map_path, 'w') as f:
        json.dump(per_pdf_keywords, f, indent=2)

    return documents, top_keywords


async def batch_ingest_all(verbose: bool = True):
    """
    Batch ingest all PDFs from DATA_DIR with detailed progress reporting.
    Only processes NEW files (skips already ingested ones).
    
    Args:
        verbose: If True, prints detailed progress information
        
    Returns:
        Tuple of (documents, topic_map)
    """
    if verbose:
        print("="*70)
        print("BATCH PDF INGESTION")
        print("="*70)
        print()
        print(f"Scanning folder: {DATA_DIR}")
    
    # Count PDFs
    pdf_files = list(Path(DATA_DIR).glob("**/*.pdf"))
    
    if verbose:
        print(f"Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"  • {pdf.name}")
        print()
    
    if not pdf_files:
        if verbose:
            print("  No PDF files found in data folder.")
            print("   Drop some PDFs into the data/ folder first.")
        return [], {}
    
    if verbose:
        print("Starting ingestion...")
        print("(This will only process NEW files, skipping already-ingested ones)")
        print()
    
    # Run ingestion
    docs, topic_map = await ingest_docs(force=False)
    
    if verbose:
        print()
        print("="*70)
        print("INGESTION COMPLETE")
        print("="*70)
        
        if topic_map:
            topic_count = len(topic_map)
            print(f"Extracted {topic_count:,} keywords/phrases")
            
            # Show top topics
            sorted_topics = sorted(topic_map.items(), key=lambda x: x[1], reverse=True)[:15]
            print("\nTop 15 topics:")
            for term, score in sorted_topics:
                marker = "[PHRASE]" if ' ' in term else "        "
                print(f"  {marker} {term}")
        
        print()
        print("You can now query your documents!")
        print()
    
    return docs, topic_map


# ── Problem → Editorial pair ingestion ────────────────────────────────────────

def ingest_editorials(pairs: list[dict]) -> int:
    """Ingest problem→editorial pairs into Chroma and BM25 for hard-problem reasoning.

    Each dict in *pairs* should have:
        - "problem":   str   the problem statement / title
        - "editorial": str   the explanation of *why* the solution works
        - "tags":      list[str] (optional) e.g. ["dp", "segment-tree", "lazy-propagation"]

    The editorial text is chunked normally and stored with metadata:
        topic = "algorithms", doc_type = "editorial",
        problem_title = <problem>, algo_category = first tag (if any).

    Returns the number of new chunks added.
    """
    from langchain_core.documents import Document

    if not pairs:
        return 0

    splits_path = Path("cache/splits.pkl")
    existing_splits = []
    if splits_path.exists():
        with splits_path.open("rb") as f:
            existing_splits = pickle.load(f)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    new_chunks = []
    for entry in pairs:
        problem = entry.get("problem", "").strip()
        editorial = entry.get("editorial", "").strip()
        tags = entry.get("tags", [])
        if not editorial:
            continue

        # Combine problem statement + editorial so the chunk is self-contained
        combined = f"Problem: {problem}\n\nEditorial / Solution Explanation:\n{editorial}"
        raw_doc = Document(
            page_content=combined,
            metadata={
                "source": f"editorial_{problem[:60].replace(' ', '_')}",
                "topic": "algorithms",
                "doc_type": "editorial",
                "problem_title": problem,
                "algo_category": tags[0] if tags else "",
            },
        )
        new_chunks.extend(splitter.split_documents([raw_doc]))

    if not new_chunks:
        return 0

    combined_splits = existing_splits + new_chunks

    # Update BM25
    tokenized = [doc.page_content.lower().split() for doc in combined_splits]
    bm25_idx = BM25Okapi(tokenized)
    pickle.dump(bm25_idx, Path("cache/global_bm25.pkl").open("wb"))

    # Update Chroma
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    cleaned = []
    for doc in new_chunks:
        c = doc.copy()
        c.metadata = _enrich_metadata(doc)
        # Preserve editorial-specific fields that _enrich_metadata may not set
        c.metadata.setdefault("doc_type", "editorial")
        c.metadata.setdefault("problem_title", doc.metadata.get("problem_title", ""))
        cleaned.append(c)

    if os.path.exists(CHROMA_DIR):
        Chroma.from_documents(
            cleaned, embedding=embeddings, persist_directory=CHROMA_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )
    else:
        Chroma.from_documents(
            cleaned, embedding=embeddings, persist_directory=CHROMA_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )

    # Save updated splits
    with splits_path.open("wb") as f:
        pickle.dump(combined_splits, f)

    _write_index_metadata(len(combined_splits))
    _log(f"Ingested {len(new_chunks)} editorial chunks from {len(pairs)} problem/editorial pairs")
    return len(new_chunks)


# Entry point for running as script: python -m brain.ingest
if __name__ == "__main__":
    import asyncio
    asyncio.run(batch_ingest_all(verbose=True))