import json
from datetime import datetime
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from .prompts import RAG_PROMPT, STRICT_RAG_PROMPT
from .config import (
    DATA_DIR, FAISS_DIR, INDEX_META_PATH, EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE,
    RETRIEVAL_K, CHUNK_SIZE, CHUNK_OVERLAP, FUSION_MODE, FUSION_ALPHA, FUSION_K_PARAM
)
from .keyword_search import search_documents
from .pdf_utils import load_pdfs

def _write_index_metadata(doc_count: int) -> None:
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "data_dir": DATA_DIR,
        "doc_count": doc_count,
        "created_at": datetime.isoformat() + "Z",
    }
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


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


def _filter_docs(docs: list[Document], filters: dict | None) -> list[Document]:
    if not filters:
        return docs
    sources = set(filters.get("source", [])) if filters.get("source") else None
    pages = set(filters.get("page", [])) if filters.get("page") else None
    filtered = []
    for doc in docs:
        meta = doc.metadata or {}
        if sources is not None and meta.get("source") not in sources:
            continue
        if pages is not None and meta.get("page") not in pages:
            continue
        filtered.append(doc)
    return filtered


def _rrf_fusion(semantic_docs: list[Document], keyword_docs: list[Document], k_param: int = 60) -> list[Document]:
    rrf_scores = {} 
    doc_map = {} 
    
    for rank, doc in enumerate(semantic_docs, start=1):
        key = _doc_key(doc)
        score = 1.0 / (k_param + rank)
        rrf_scores[key] = rrf_scores.get(key, 0) + score
        if key not in doc_map:
            doc_map[key] = doc
    
    for rank, doc in enumerate(keyword_docs, start=1):
        key = _doc_key(doc)
        score = 1.0 / (k_param + rank)
        rrf_scores[key] = rrf_scores.get(key, 0) + score
        if key not in doc_map:
            doc_map[key] = doc
    
    sorted_keys = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    fused_docs = [doc_map[key] for key, score in sorted_keys]
    
    return fused_docs


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]


def _weighted_fusion(semantic_docs: list[Document], keyword_docs: list[Document], alpha: float = 0.5) -> list[Document]:
    semantic_scores = list(range(len(semantic_docs), 0, -1)) 
    keyword_scores = list(range(len(keyword_docs), 0, -1))
    
    norm_semantic = _normalize_scores(semantic_scores)
    norm_keyword = _normalize_scores(keyword_scores)
    
    doc_map = {}
    
    for i, doc in enumerate(semantic_docs):
        key = _doc_key(doc)
        if key not in doc_map:
            doc_map[key] = (doc, 0.0)
        _, prev_score = doc_map[key]
        blended = alpha * norm_semantic[i] + (1 - alpha) * 0.0 
        doc_map[key] = (doc, blended)
    
    for i, doc in enumerate(keyword_docs):
        key = _doc_key(doc)
        if key not in doc_map:
            doc_map[key] = (doc, 0.0)
        current_doc, prev_semantic_score = doc_map[key]
        blended = prev_semantic_score + (1 - alpha) * norm_keyword[i]
        doc_map[key] = (current_doc, blended)
    
    sorted_items = sorted(doc_map.items(), key=lambda x: x[1][1], reverse=True)
    fused_docs = [doc for key, (doc, score) in sorted_items]
    
    return fused_docs


def ingest_docs():
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(FAISS_DIR)
    _write_index_metadata(len(splits))
    print(f"Ingested {len(splits)} chunks.")

def hybrid_retrieval(question: str, k: int = RETRIEVAL_K, filters: dict | None = None, fusion_mode: str = "rrf", alpha: float = 0.5, k_param: int = 60):

    _validate_index_metadata()
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": max(k * 3, k)})
    semantic_docs = semantic_retriever.invoke(question)
    semantic_docs = _filter_docs(semantic_docs, filters)
    

    pdf_docs = load_pdfs(DATA_DIR)
    keyword_docs = search_documents(question, pdf_docs, n_results=max(k * 3, k))
    
    keyword_docs_obj = [
        Document(page_content=doc['content'], metadata=doc['metadata'])
        for doc in keyword_docs
    ]
    

    if fusion_mode == "rrf":
        combined = _rrf_fusion(semantic_docs, keyword_docs_obj, k_param=k_param)
    elif fusion_mode == "weighted":
        combined = _weighted_fusion(semantic_docs, keyword_docs_obj, alpha=alpha)
    else:
        seen = set()
        combined = []
        for doc in semantic_docs:
            key = _doc_key(doc)
            if key not in seen:
                combined.append(doc)
                seen.add(key)
        for doc in keyword_docs_obj:
            key = _doc_key(doc)
            if key not in seen:
                combined.append(doc)
                seen.add(key)
    
    return combined[:k]

def query_brain(question: str, verbose: bool = False, fusion_mode: str = None, alpha: float = None, k_param: int = None):
    fusion_mode = fusion_mode or FUSION_MODE
    alpha = alpha if alpha is not None else FUSION_ALPHA
    k_param = k_param if k_param is not None else FUSION_K_PARAM
    
    docs = hybrid_retrieval(question, fusion_mode=fusion_mode, alpha=alpha, k_param=k_param)

    if verbose:
        print("RETRIEVED:", len(docs))
        for d in docs:
            print(d.metadata)
            print(d.page_content[:300])
            print("----")

    llm = OllamaLLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    result = STRICT_RAG_PROMPT.invoke({"context": context, "input": question})
    llm_output = llm.invoke(result.to_string())
    
    return llm_output

if __name__ == "__main__":
    print("1: Ingest | 2: Query")
    choice = input("Choose: ")
    if choice == "1":
        ingest_docs()
    else:
        print("Query mode. Type 'quit' to exit.\n")
        while True:
            q = input("Ask: ").strip()
            if q.lower() == "quit":
                print("Goodbye!")
                break
            if q:
                result = query_brain(q)
                print(f"\nAnswer: {result}\n")
