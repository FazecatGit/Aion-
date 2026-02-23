import os
from pathlib import Path


def _env_bool(name: str, default: str = "0") -> bool:
	return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = os.getenv("RAG_DATA_DIR", str(BASE_DIR / "data"))
#FAISS_DIR = os.getenv("RAG_FAISS_DIR", str(BASE_DIR / "faiss_index"))
CHROMA_DIR = os.getenv("RAG_CHROMA_DIR", str(BASE_DIR / "cache" / "chroma_db"))
INDEX_META_PATH = os.getenv("RAG_INDEX_META_PATH", str(Path(CHROMA_DIR) / "index_meta.json"))

# Embedding model settings
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")

# LLM settings
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen2.5-coder:7b")
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0"))

# Retrieval settings
RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "5"))

# Text splitting settings
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))

# Hybrid fusion settings
FUSION_MODE = os.getenv("RAG_FUSION_MODE", "rrf")  # "rrf" or "weighted"
FUSION_ALPHA = float(os.getenv("RAG_FUSION_ALPHA", "0.5"))  # For weighted mode: 1.0 = semantic, 0.0 = keyword
FUSION_K_PARAM = int(os.getenv("RAG_FUSION_K_PARAM", "60"))  # For RRF mode: higher = more conservative

# Query enhancement settings
ENABLE_QUERY_SPELL_CORRECTION = _env_bool("RAG_ENABLE_QUERY_SPELL_CORRECTION", "1")
ENABLE_QUERY_REWRITE = _env_bool("RAG_ENABLE_QUERY_REWRITE", "0")
ENABLE_QUERY_EXPANSION = _env_bool("RAG_ENABLE_QUERY_EXPANSION", "0")

# Retrieval depth settings
RETRIEVAL_CANDIDATE_MULTIPLIER = int(os.getenv("RAG_RETRIEVAL_CANDIDATE_MULTIPLIER", "3"))

# Reranking settings
RERANK_METHOD = os.getenv("RAG_RERANK_METHOD", "keyword")  # "none", "keyword", "cross_encoder"
CROSS_ENCODER_MODEL = os.getenv("RAG_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_BATCH_SIZE = 32 # For cross-encoder reranking

# ChromaDB config
CHROMA_PERSIST_DIR = CHROMA_DIR		      # where vectors live
CHROMA_COLLECTION_NAME = "aion-code"     # collection for code docs
CHROMA_CHUNK_SIZE = 512                 # optional, for new ingest
CHROMA_CHUNK_OVERLAP = 64               # optional for new ingest

# fitler settings for query pipeline
USE_SMART_K = True