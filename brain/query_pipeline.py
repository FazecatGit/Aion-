

from typing import Optional
import json
import numpy as np

from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

from .keyword_search import tokenize_text
from .prompts import (
    build_query_expansion_prompt,
    build_query_rewrite_prompt,
    build_spell_correction_prompt,
)


_cross_encoder_cache: dict = {}

def _safe_llm_invoke(llm: OllamaLLM, prompt: str, verbose: bool = False) -> Optional[str]:
    try:
        output = llm.invoke(prompt)
    except Exception:
        if verbose:
            print(f"[DEBUG] LLM invoke failed for prompt: {prompt[:100]}...")
        return None

    if output is None:
        return None
    return str(output).strip()


def _build_query_llm(llm_model: str) -> OllamaLLM:
    return OllamaLLM(model=llm_model, temperature=0)


def _spell_correct_query(query: str, llm: OllamaLLM) -> str:
    corrected = _safe_llm_invoke(llm, build_spell_correction_prompt(query))
    return corrected if corrected else query


def _rewrite_query(query: str, llm: OllamaLLM) -> str:
    rewritten = _safe_llm_invoke(llm, build_query_rewrite_prompt(query))
    return rewritten if rewritten else query


def _expand_query(query: str, llm: OllamaLLM) -> str:
    expanded = _safe_llm_invoke(llm, build_query_expansion_prompt(query))
    return expanded if expanded else query


def enhance_query_for_retrieval(
    query: str,
    llm_model: str,
    enable_spell_correction: bool,
    enable_rewrite: bool,
    enable_expansion: bool,
    verbose: bool = False,
) -> str:
    if not any([enable_spell_correction, enable_rewrite, enable_expansion]):
        return query

    if len(query.split()) <= 2:
        return query
    
    llm = _build_query_llm(llm_model)
    enhanced_query = query

    if enable_spell_correction:
        enhanced_query = _spell_correct_query(enhanced_query, llm)
    if enable_rewrite:
        enhanced_query = _rewrite_query(enhanced_query, llm)
    if enable_expansion:
        enhanced_query = _expand_query(enhanced_query, llm)

    if verbose and enhanced_query != query:
        print(f"Enhanced query: '{query}' -> '{enhanced_query}'")
    return enhanced_query


def _keyword_rerank_documents(docs: list[Document], query: str) -> list[Document]:
    query_tokens = set(tokenize_text(query))
    if not query_tokens:
        return docs

    scored_docs = []
    query_lower = query.lower()
    for doc in docs:
        content = doc.page_content or ""
        content_tokens = set(tokenize_text(content))
        overlap = len(query_tokens & content_tokens)
        score = overlap / max(1, len(query_tokens))
        if query_lower in content.lower():
            score += 0.25
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs]


def _cross_encoder_rerank_documents(
    docs: list[Document],
    query: str,
    cross_encoder_model: str,
    batch_size: int = 32,
    score_threshold: float = 0.0,
    verbose: bool = False,
) -> list[Document]:
    try:
        from sentence_transformers import CrossEncoder
        import numpy as np
    except Exception:
        if verbose:
            print("Cross-encoder unavailable; falling back to keyword rerank.")
        return _keyword_rerank_documents(docs, query)

    try:
        if cross_encoder_model not in _cross_encoder_cache:
            _cross_encoder_cache[cross_encoder_model] = CrossEncoder(cross_encoder_model)
        model = _cross_encoder_cache[cross_encoder_model]

        pairs  = [(query, (doc.page_content or "")[:2000]) for doc in docs]
        raw    = model.predict(pairs, batch_size=batch_size)
        scores = 1 / (1 + np.exp(-np.array(raw)))  # sigmoid → 0-1

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        if verbose:
            for score, doc in ranked:
                print(f"[RERANK] {score:.3f} → {doc.metadata.get('source', '')}")

        return [doc for score, doc in ranked if score >= score_threshold]

    except Exception:
        if verbose:
            print("Cross-encoder scoring failed; falling back to keyword rerank.")
        return _keyword_rerank_documents(docs, query)

def rerank_documents(
    docs: list[Document],
    query: str,
    method: str,
    cross_encoder_model: str,
    batch_size: int = 32,
    verbose: bool = False,
) -> list[Document]:
    method_normalized = (method or "none").strip().lower()
    if method_normalized == "none":
        return docs
    if method_normalized == "keyword":
        return _keyword_rerank_documents(docs, query)
    if method_normalized == "cross_encoder":
        return _cross_encoder_rerank_documents(docs, query, cross_encoder_model, batch_size=batch_size, verbose=verbose)

    if verbose:
        print(f"Unknown rerank method '{method}'. Skipping rerank.")
    return docs


def evaluate_documents_with_llm(
    query: str,
    docs: list[Document],
    llm_model: str,
    verbose: bool = False,
) -> list[int]:
    if not docs:
        return []

    llm = OllamaLLM(model=llm_model, temperature=0)

    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content[:300]
        formatted_docs.append(f"{i}. {content}")

    evaluation_prompt = f"""Rate how relevant each document/snippet is to this query on a 0-3 scale:

Query: "{query}"

Documents:
{chr(10).join(formatted_docs)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Return ONLY the scores as a JSON list, nothing else. For example: [3, 1, 2, 0]"""

    try:
        response = llm.invoke(evaluation_prompt)
        if response:
            response_text = response.strip()
            if verbose:
                print(f"LLM evaluation response: {response_text}")

            try:
                if "[" in response_text and "]" in response_text:
                    json_str = response_text[response_text.find("["):response_text.rfind("]")+1]
                    scores = json.loads(json_str)

                    if not isinstance(scores, list):
                        raise ValueError("Scores is not a list")

                    if len(scores) >= len(docs):          
                        scores = scores[:len(docs)]
                    else:                               
                        scores = scores + [1] * (len(docs) - len(scores))

                    return [max(0, min(3, int(s))) for s in scores]

            except (json.JSONDecodeError, ValueError):
                if verbose:
                    print("Failed to parse LLM scores as JSON")

        if verbose:
            print("Falling back to neutral scores (all 1s)")
        return [1] * len(docs)

    except Exception as e:
        if verbose:
            print(f"LLM evaluation error: {e}")
        return [1] * len(docs)
