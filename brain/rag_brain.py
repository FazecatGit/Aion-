import json

from pathlib import Path
from operator import itemgetter

from brain.fast_search import fast_topic_search
from brain.keyword_search import search_documents
#from .ingest import fast_topic_search
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document
from .metrics import evaluate_retrieval
from .prompts import STRICT_RAG_PROMPT
from .config import (
    DATA_DIR, INDEX_META_PATH, EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE, RERANK_BATCH_SIZE,
    RETRIEVAL_K, CHUNK_SIZE, CHUNK_OVERLAP, FUSION_MODE, FUSION_ALPHA, FUSION_K_PARAM,
    ENABLE_QUERY_SPELL_CORRECTION, ENABLE_QUERY_REWRITE, ENABLE_QUERY_EXPANSION,
    RETRIEVAL_CANDIDATE_MULTIPLIER, RERANK_METHOD, CROSS_ENCODER_MODEL, CHROMA_DIR, USE_SMART_K
)
from .pdf_utils import load_pdfs
from .query_pipeline import enhance_query_for_retrieval, rerank_documents

def get_dynamic_k(question: str) -> int:
    word_count = len(question.split())
    concepts = len(set(w for w in question.split() if len(w) > 4))
    return 8 if concepts >= 3 else 5 if word_count > 10 else 3

async def get_smart_k(question: str, llm_model) -> int:
    prompt = f"""Classify query complexity 1-3 ONLY with this number:

1 = if the topic of the word is basic that is considered easy in the world of progamming
2 = if the topic of the word is intermediate that is considered common in the world of progamming 
3 = if the topic of the word is advanced that is considered difficult in the world of progamming

Query: "{question}"

Just the number (1, 2, or 3):"""
  
    complexity = await llm_model.ainvoke(prompt)
    return {1:3, 2:5, 3:8}.get(int(complexity), 5)

def _load_index_metadata() -> dict:
    with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _validate_index_metadata() -> None:
    if not Path(INDEX_META_PATH).exists():
        raise ValueError("Index metadata not found. Run ingest_docs() to create the index.")
    meta = _load_index_metadata()
    expected = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "data_dir": DATA_DIR,
    }
    mismatches = []
    for key, value in expected.items():
        if meta.get(key) != value:
            mismatches.append(f"{key}: expected '{value}', got '{meta.get(key)}'")
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"Index metadata mismatch. Re-ingest required. Details: {details}")


def _doc_key(doc: Document) -> tuple:
    meta = doc.metadata or {}
    source = meta.get("source")
    page = meta.get("page")
    if source is not None and page is not None:
        return (source, page)
    return (source, page, doc.page_content[:200])


def _match_filters(filters: dict | None) -> callable:
    if not filters:
        return lambda doc: True

    def _extract_list(val):
        if isinstance(val, dict):
            return set(val.get("$in", []))
        if isinstance(val, list):
            return set(val)
        return {val}

    allowed_sources = _extract_list(filters["source"]) if filters.get("source") else None
    allowed_pages   = _extract_list(filters["page"])   if filters.get("page")   else None

    def predicate(doc: Document) -> bool:
        meta = doc.metadata or {}
        if allowed_sources is not None and meta.get("source") not in allowed_sources:
            return False
        if allowed_pages is not None and meta.get("page") not in allowed_pages:
            return False
        return True

    return predicate

def _filter_docs(docs: list[Document], filters: dict | None) -> list[Document]:
    return list(filter(_match_filters(filters), docs))


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    range_val = max_score - min_score
    return [(s - min_score) / range_val for s in scores]


def _assign_rank_scores(docs: list[Document], k_param: int = 60) -> dict:
    return {
        _doc_key(doc): 1.0 / (k_param + rank)
        for rank, doc in enumerate(docs, start=1)
    }


def _merge_score_dicts(*score_dicts) -> dict:
    result = {}
    for scores in score_dicts:
        for key, score in scores.items():
            result[key] = result.get(key, 0) + score
    return result


def _rrf_fusion(semantic_docs: list[Document], keyword_docs: list[Document], k_param: int = 60) -> list[Document]:
    semantic_scores = _assign_rank_scores(semantic_docs, k_param)
    keyword_scores = _assign_rank_scores(keyword_docs, k_param)
    rrf_scores = _merge_score_dicts(semantic_scores, keyword_scores)
    
    doc_map = {_doc_key(doc): doc for doc in semantic_docs + keyword_docs}
    sorted_keys = sorted(rrf_scores.items(), key=itemgetter(1), reverse=True)
    
    return [doc_map[key] for key, _ in sorted_keys]


def _weighted_fusion(semantic_docs: list[Document], keyword_docs: list[Document], alpha: float = 0.5) -> list[Document]:
    semantic_scores = _normalize_scores(list(range(len(semantic_docs), 0, -1)))
    keyword_scores = _normalize_scores(list(range(len(keyword_docs), 0, -1)))
    
    semantic_map = {
        _doc_key(doc): (doc, semantic_scores[i])
        for i, doc in enumerate(semantic_docs)
    }
    keyword_map = {
        _doc_key(doc): (doc, keyword_scores[i])
        for i, doc in enumerate(keyword_docs)
    }
    
    all_keys = set(semantic_map.keys()) | set(keyword_map.keys())
    blended = {
        key: (
            semantic_map[key][0] if key in semantic_map else keyword_map[key][0],
            alpha * semantic_map.get(key, (None, 0.0))[1] + (1 - alpha) * keyword_map.get(key, (None, 0.0))[1]
        )
        for key in all_keys
    }
    
    sorted_items = sorted(blended.items(), key=lambda x: x[1][1], reverse=True)
    return [doc for _, (doc, _) in sorted_items]


def _build_keyword_docs(keyword_results: list[dict]) -> list[Document]:
    return [
        Document(page_content=doc['content'], metadata=doc['metadata'])
        for doc in keyword_results
    ]


def _select_fusion(fusion_mode: str):
    return {
        "rrf": _rrf_fusion,
        "weighted": _weighted_fusion,
    }.get(fusion_mode)


def _fuse_results(semantic_docs: list[Document], keyword_docs: list[Document], mode: str, **kwargs) -> list[Document]:
    fusion_fn = _select_fusion(mode)
    if fusion_fn:

        if mode == "rrf":
            return fusion_fn(semantic_docs, keyword_docs, k_param=kwargs.get("k_param", 60))
        else:
            return fusion_fn(semantic_docs, keyword_docs, alpha=kwargs.get("alpha", 0.5))
    
    seen = set()
    result = []
    for doc in semantic_docs + keyword_docs:
        key = _doc_key(doc)
        if key not in seen:
            result.append(doc)
            seen.add(key)
    return result

async def hybrid_retrieval(
    question: str,
    k: int = RETRIEVAL_K,
    filters: dict | None = None,
    fusion_mode: str = "rrf",
    alpha: float = 0.5,
    k_param: int = 60,
    rerank_method: str = RERANK_METHOD,
    cross_encoder_model: str = CROSS_ENCODER_MODEL,
    batch_size: int = RERANK_BATCH_SIZE,
    verbose: bool = False,
    raw_docs: list[dict] | None = None,
) -> list[Document]:
    if raw_docs is None:
        raw_docs = load_pdfs(DATA_DIR) 
    
    _validate_index_metadata()

    print(f"[DEBUG] Running fast topic pre-filter")
    bm25_results = fast_topic_search(question)
    
    if len(bm25_results) >= 5:
        print("[DEBUG] BM25 sufficient - skipping Chroma!")
        reranked_docs = rerank_documents(
            docs=bm25_results[:k*2],
            query=question,
            method=rerank_method,
            cross_encoder_model=cross_encoder_model,
            batch_size=batch_size,
            verbose=verbose,
        )
        return reranked_docs[:k]
    
    print("[DEBUG] BM25 insufficient, running full hybrid")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    effective_question = enhance_query_for_retrieval(
        query=question,
        llm_model=LLM_MODEL,
        enable_spell_correction=ENABLE_QUERY_SPELL_CORRECTION,
        enable_rewrite=ENABLE_QUERY_REWRITE,
        enable_expansion=ENABLE_QUERY_EXPANSION,
        verbose=verbose,
    )
    
    if not isinstance(effective_question, str):
        if isinstance(effective_question, (list, tuple)):
            eq_str = " ".join(map(str, effective_question))
        else:
            eq_str = str(effective_question)
    else:
        eq_str = effective_question

    candidate_multiplier = max(1, RETRIEVAL_CANDIDATE_MULTIPLIER)     

    if USE_SMART_K:
        llm = OllamaLLM(model=LLM_MODEL)  
        base_k = await get_smart_k(question, llm)
    else:
        base_k = get_dynamic_k(question)

    candidate_k = max(base_k * candidate_multiplier, base_k) 

    if verbose:
        print(f"[DEBUG] candidate_k: {candidate_k}, effective_question: '{eq_str}'")

    search_kwargs = {"k": candidate_k}
    if filters:
        search_kwargs["filter"] = filters
    
    try:
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        # For langchain 0.1+, invoke returns a list of Documents
        result = retriever.invoke(eq_str)
        semantic_docs = result if isinstance(result, list) else []
    except Exception as e:
        if verbose:
            print(f"[DEBUG] ChromaDB retrieval error: {e}")
        semantic_docs = []
    
    if verbose:
        print(f"[DEBUG] semantic_docs count: {len(semantic_docs)}")

    dict_docs = [{"content": d['content'], "metadata": d['metadata']} for d in raw_docs]
    
    if verbose:
        print(f"[DEBUG] raw_docs count: {len(raw_docs)}")
    
    bm25_results = search_documents(
        query=eq_str, 
        documents=dict_docs, 
        n_results=candidate_k, 
        use_bm25=True
    )
    
    if verbose:
        print(f"[DEBUG] bm25_results count: {len(bm25_results)}")
    
    keyword_docs = [
        Document(page_content=res["content"], metadata=res["metadata"]) 
        for res in bm25_results
    ]

    fused_docs = _fuse_results(
        semantic_docs, 
        keyword_docs, 
        fusion_mode,
        alpha=alpha,
        k_param=k_param
    )

    fused_docs = _filter_docs(fused_docs, filters) 

    if verbose:
        print(f"[DEBUG] Fused docs count after filtering: {len(fused_docs)}")

    reranked_docs = rerank_documents(
        docs=fused_docs,
        query=eq_str,
        method=rerank_method,
        cross_encoder_model=cross_encoder_model,
        batch_size=batch_size,
        verbose=verbose,
    )
    
    return reranked_docs[:k]

async def query_brain(
    question: str,
    verbose: bool = False,
    fusion_mode: str = None,
    alpha: float = None,
    k_param: int = None,
    k: int = RETRIEVAL_K,
    filters: dict | None = None,
    raw_docs: list[dict] | None = None,
    debug_relevant: list[str] | None = None,
) -> list[Document]:
    fusion_mode = fusion_mode or FUSION_MODE
    alpha       = alpha    if alpha    is not None else FUSION_ALPHA
    k_param     = k_param  if k_param  is not None else FUSION_K_PARAM

    docs = await hybrid_retrieval(
        question,
        k=k,
        filters=filters,
        raw_docs=raw_docs,
        fusion_mode=fusion_mode,
        alpha=alpha,
        k_param=k_param,
        rerank_method=RERANK_METHOD,
        cross_encoder_model=CROSS_ENCODER_MODEL,
        batch_size=RERANK_BATCH_SIZE,
        verbose=verbose,
    )

    if verbose:
        print("RETRIEVED:", len(docs))
        for d in docs:
            print(d.metadata)
            print(d.page_content[:300])
            print("----")

        if debug_relevant is not None:
            retrieved_ids = [d.metadata.get("source", "") for d in docs]
            scores = evaluate_retrieval(retrieved_ids, debug_relevant)
            print(f"[DEBUG] precision={scores['precision']:.2f}  recall={scores['recall']:.2f}  f1={scores['f1']:.2f}")

    return docs
