from pathlib import Path
import pickle
from typing import Any, List, Dict

BM25_IDX = None
BM25_CHUNKS = None

def load_bm25_index(index_path: str = "cache/global_bm25.pkl") -> Any:
    try:
        bm25_idx = pickle.load(Path(index_path).open('rb'))
        print(f"[FAST_SEARCH] BM25 Index Loaded: {bm25_idx}")
        return bm25_idx
    except FileNotFoundError:
        print(f"[FAST_SEARCH] Error: BM25 index not found at {index_path}")
        return None
    except Exception as e:
        print(f"[FAST_SEARCH] Error loading BM25 index: {e}")
        return None

def get_bm25_scores(query: str, bm25_idx: Any) -> List[float]:
    if bm25_idx is None: return []
    try:
        query_tokenized = query.lower().split()
        scores = bm25_idx.get_scores(query_tokenized)
        return scores
    except Exception as e:
        print(f"[FAST_SEARCH] Error: {e}")
        return []
    
def sort_and_limit_results(scores: List[float]) -> List[int]:
    try:
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        print(f"[FAST_SEARCH] Sorted Indices: {sorted_indices}")
        return sorted_indices
    except Exception as e:
        print(f"[FAST_SEARCH] Error sorting and limiting results: {e}")
        return []


def _expand_query_variants(query: str) -> List[str]:
    variants = [query]
    tokens = query.lower().split()

    # Add a shorter keyword-only variant (remove filler)
    stop_words = {"a", "an", "the", "to", "and", "or", "so", "is", "in", "of",
                  "for", "with", "how", "that", "this", "it", "be", "can", "i"}
    keywords = [t for t in tokens if t not in stop_words and len(t) > 2]
    if keywords and keywords != tokens:
        variants.append(" ".join(keywords))

    # Add a bigram-focused variant (pairs of adjacent keywords)
    if len(keywords) >= 2:
        bigrams = [f"{keywords[i]} {keywords[i+1]}" for i in range(len(keywords) - 1)]
        variants.append(" ".join(bigrams[:4]))  # cap to first 4 bigrams

    return list(dict.fromkeys(variants))  # deduplicate while preserving order


def _bm25_search_single(query: str, bm25_idx: Any, all_chunks: list, top_k: int) -> List[tuple]:
    scores = bm25_idx.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(float(scores[i]), i) for i in ranked]


def fast_topic_search(query: str, return_scores: bool = False, top_k: int = 20, candidate_k: int = 100, rerank_method: str = "keyword", topic_filter: list[str] | None = None):
    """
    BM25 search with multi-query expansion and keyword reranking.
    - Generates query variants (no LLM needed, instant)
    - Retrieves top-candidate_k candidates across all variants (wider recall)
    - Deduplicates by chunk index, merging scores via max
    - Reranks the merged pool using specified method (keyword by default; cross_encoder optional)
    - Returns top_k results
    """
    bm25_idx = load_bm25_index()
    if bm25_idx is None:
        return []

    splits_path = Path("cache/splits.pkl")
    if not splits_path.exists():
        return []

    with splits_path.open('rb') as f:
        all_chunks = pickle.load(f)

    # --- Multi-query expansion ---
    variants = _expand_query_variants(query)

    # --- Retrieve wider candidate pool across all variants ---
    merged: Dict[int, float] = {}  # chunk_idx -> best BM25 score across variants
    for variant in variants:
        for score, idx in _bm25_search_single(variant, bm25_idx, all_chunks, top_k=candidate_k):
            merged[idx] = max(merged.get(idx, 0.0), score)

    # Sort merged pool by best score
    ranked_pool = sorted(merged.items(), key=lambda x: x[1], reverse=True)

    # --- Build candidate doc list ---
    candidates = []
    for idx, score in ranked_pool:
        if idx < len(all_chunks):
            doc = all_chunks[idx]
            doc.metadata["bm25_score"] = score
            if "source" not in doc.metadata:
                doc.metadata["source"] = "Unknown"
            candidates.append(doc)

    # --- Rerank the candidate pool using specified method ---
    from .query_pipeline import rerank_documents
    from .config import CROSS_ENCODER_MODEL, RERANK_BATCH_SIZE

    # Apply topic filter if provided (e.g. ["algorithms", "clean-code"])
    if topic_filter:
        allowed = set(topic_filter)
        candidates = [c for c in candidates if c.metadata.get("topic", "general") in allowed]
        if not candidates:
            # Fall back to unfiltered if filter eliminated everything
            candidates = [all_chunks[idx] for idx, _ in ranked_pool if idx < len(all_chunks)]

    reranked = rerank_documents(
        docs=candidates,
        query=query,
        method=rerank_method,
        cross_encoder_model=CROSS_ENCODER_MODEL,
        batch_size=RERANK_BATCH_SIZE,
        verbose=False
    )

    print(f"[FAST_SEARCH] {len(variants)} query variant(s), {len(candidates)} candidates -> top {top_k} after {rerank_method} rerank")
    return reranked[:top_k]

def initialize_bm25(raw_docs=None):
    global BM25_IDX, BM25_CHUNKS

    bm25_idx = load_bm25_index()
    if bm25_idx is None and raw_docs is not None:
        from rank_bm25 import BM25Okapi
        tokenized_docs = [doc.page_content.lower().split() for doc in raw_docs]
        bm25_idx = BM25Okapi(tokenized_docs)
        pickle.dump(bm25_idx, Path("cache/global_bm25.pkl").open('wb'))

    BM25_IDX = bm25_idx

    # load chunks
    splits_path = Path("cache/splits.pkl")
    if splits_path.exists():
        with splits_path.open('rb') as f:
            BM25_CHUNKS = pickle.load(f)
    else:
        BM25_CHUNKS = raw_docs