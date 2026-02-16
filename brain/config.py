import os
from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = os.getenv("RAG_DATA_DIR", str(BASE_DIR / "data"))
FAISS_DIR = os.getenv("RAG_FAISS_DIR", str(BASE_DIR / "faiss_index"))
INDEX_META_PATH = os.getenv("RAG_INDEX_META_PATH", str(Path(FAISS_DIR) / "index_meta.json"))

# Embedding model settings
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")

# LLM settings
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "dolphin-llama3:8b")
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0"))

# Retrieval settings
RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "5"))

# Text splitting settings
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# Hybrid fusion settings
FUSION_MODE = os.getenv("RAG_FUSION_MODE", "rrf")  # "rrf" or "weighted"
FUSION_ALPHA = float(os.getenv("RAG_FUSION_ALPHA", "0.5"))  # For weighted mode: 1.0 = semantic, 0.0 = keyword
FUSION_K_PARAM = int(os.getenv("RAG_FUSION_K_PARAM", "60"))  # For RRF mode: higher = more conservative
