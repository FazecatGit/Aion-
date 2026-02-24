from rank_bm25 import BM25Okapi
import json, pickle
from typing import Any, List, Dict
from collections import Counter
import re
import os
import shutil

from datetime import datetime
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from .utils import pipe
from .config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, INDEX_META_PATH, LLM_MODEL
from .pdf_utils import load_pdfs


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

def extract_keywords_from_corpus(docs: List[Any]) -> Dict[str, int]:
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'of', 'in', 'to', 'for', 'with', 'by', 'from', 'as', 'at', 'this', 'that',
        'it', 'its', 'if', 'can', 'we', 'you', 'your', 'them', 'their', 'more', 'than',
        'also', 'anyway', 'anything', 'anything else', 'anyone', 'anything at all',
        'anything in the world', 'anything like that', 'anything more', 'anything other than',
        'anything similar', 'anything to', 'anything under any circumstances', 'anything whatever',
        'anything within reason', 'anything you can think of', 'anything you could imagine',
        'anything whatsoever', 'anybody', 'anybody else', 'anybody in the world', 'anybody like that',
        'anybody more', 'anybody other than', 'anybody similar', 'anybody to', 'anybody under any circumstances',
        'anybody whatever', 'anybody within reason', 'anybody you can think of', 'anybody you could imagine',
        'anybody whatsoever', 
    }

    tokenized_docs = [doc.page_content.lower().split() for doc in docs]
    all_tokens = [token for doc in tokenized_docs 
                  for token in doc if token not in stop_words and len(token) > 2]
    
    word_freq = Counter(all_tokens).most_common(250)
    
    scores = [freq for _, freq in word_freq]
    
    return dict(zip([w for w, f in word_freq], scores))

async def ingest_docs():
    if os.path.exists(CHROMA_DIR):
        print(f"Clearing old Chroma database at {CHROMA_DIR}...")
        shutil.rmtree(CHROMA_DIR)

    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = pipe(
        loader.load(),
        lambda docs: splitter.split_documents(docs)
    )

    print(f"Loaded {len(splits)} chunks")

    print("Extracting top keywords from corpus...")
    top_keywords = extract_keywords_from_corpus(splits)
    print(f"Found {len(top_keywords)} key topics...")

    tokenized_docs = [doc.page_content.lower().split() for doc in splits] 
    bm25_idx = BM25Okapi(tokenized_docs)
    pickle.dump(bm25_idx, Path("cache/global_bm25.pkl").open('wb'))
    print(f"Built BM25 index: {len(splits)} chunks")

    with open(TOPIC_MAP_PATH, 'w') as f:
        json.dump(top_keywords, f, indent=2)

    print("Building Chroma vector DB...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    cleaned_splits = []
    for doc in splits:
        clean_doc = doc.copy()
        clean_doc.metadata = {k: v for k, v in doc.metadata.items() 
                            if k not in ['source', 'file_path']} 
        cleaned_splits.append(clean_doc)

    Chroma.from_documents(cleaned_splits, embedding=embeddings, persist_directory=CHROMA_DIR)
    with Path("cache/splits.pkl").open('wb') as f:
        pickle.dump(splits, f)  # Use original splits (not cleaned)
    print("✓ Cached splits.pkl for BM25")

    _write_index_metadata(len(splits))

    documents = load_pdfs(DATA_DIR)

    print(f"Ingest complete: {len(splits)} chunks → {len(documents)} PDFs, {len(top_keywords)} keywords, BM25 ready")
    return documents, top_keywords