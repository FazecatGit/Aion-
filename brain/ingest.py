import json
from datetime import datetime
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from .utils import pipe
from .config import DATA_DIR, FAISS_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, INDEX_META_PATH, LLM_MODEL
from .pdf_utils import load_pdfs
from .keyword_search import search_documents
from langchain_ollama import OllamaLLM

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
    

def generate_tech_synonyms(topic: str) -> list[str]:
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)
    prompt = f"List exactly 5 programming-related synonyms or related concepts for the word '{topic}'. Output ONLY a comma-separated list, nothing else."
    
    response = llm.invoke(prompt)
    return [word.strip().lower() for word in response.split(",")]

def ingest_docs():
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    splits = pipe(
        loader.load(),
        lambda docs: splitter.split_documents(docs)
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(FAISS_DIR)
    _write_index_metadata(len(splits))

    documents = load_pdfs(DATA_DIR)
    topic_synonyms = {}
    if Path(TOPIC_MAP_PATH).exists():
        with open(TOPIC_MAP_PATH, "r", encoding="utf-8") as f:
            topic_synonyms = json.load(f)

    for doc in documents:
        topic = doc['metadata'].get('topic', '').lower()
        if topic and topic not in topic_synonyms:
            print(f"Asking Ollama to map synonyms for new topic: {topic}...")
            topic_synonyms[topic] = generate_tech_synonyms(topic)

    with open(TOPIC_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(topic_synonyms, f, indent=2)

    print(f"Ingested {len(splits)} vector chunks and mapped {len(topic_synonyms)} topics.")
    return documents, topic_synonyms

def fast_topic_search(query: str, documents: list):
    try:
        with open(TOPIC_MAP_PATH, "r", encoding="utf-8") as f:
            topic_synonyms = json.load(f)
    except FileNotFoundError:
        print("Topic map not found. Please run ingest_docs() first.")
        return []

    expanded_topics = set()
    for topic, synonyms in topic_synonyms.items():
        if topic in query.lower():
            expanded_topics.add(topic)
            expanded_topics.update(synonyms)
            
    relevant_chunks = []
    if expanded_topics:
        relevant_chunks = [doc for doc in documents if doc['metadata'].get('topic') in expanded_topics]
    
    if not relevant_chunks:
        relevant_chunks = documents

    results = search_documents(query, relevant_chunks, n_results=5)
        
    return results
