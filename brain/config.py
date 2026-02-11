"""Configuration settings for RAG brain system."""
import os
from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = os.getenv("RAG_DATA_DIR", str(BASE_DIR / "data"))
FAISS_DIR = os.getenv("RAG_FAISS_DIR", str(BASE_DIR / "faiss_index"))

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
