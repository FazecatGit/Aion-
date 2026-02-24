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
        
def fast_topic_search(query: str, return_scores: bool = False):
    bm25_idx = load_bm25_index()
    if bm25_idx is None:
        return []

    query_tokenized = query.lower().split()
    scores = bm25_idx.get_scores(query_tokenized)

    sorted_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:10]

    splits_path = Path("cache/splits.pkl")
    if not splits_path.exists():
        return []

    with splits_path.open('rb') as f:
        all_chunks = pickle.load(f)

    results = []
    for i in sorted_indices:
        if i < len(all_chunks):
            doc = all_chunks[i]
            doc.metadata["score"] = float(scores[i]) 
            results.append(doc)
    
    return results

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