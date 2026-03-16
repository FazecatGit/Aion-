"""
Session Memory — Chroma-backed conversation store for the code agent.

Stores conversation turns (user instructions + agent actions + outcomes) so the
agent can recall what it tried before and refine strategies across requests.

Usage:
    memory = SessionMemory()                    # one per server lifetime
    memory.add_turn(session_id, role, content, metadata)
    relevant = memory.recall(session_id, query, k=5)
    memory.clear_session(session_id)
"""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from brain.config import EMBEDDING_MODEL
from print_logger import get_logger

logger = get_logger("code_agent")

# Separate directory so it doesn't collide with the RAG knowledge store
_MEMORY_PERSIST_DIR = "cache/session_memory_db"
_MEMORY_COLLECTION  = "agent-session-memory"


class SessionMemory:
    """Chroma-backed session memory for the code agent."""

    def __init__(self, persist_dir: str = _MEMORY_PERSIST_DIR):
        self._embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self._store = Chroma(
            collection_name=_MEMORY_COLLECTION,
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
        )
        logger.info("[SESSION_MEMORY] Initialized (persist=%s)", persist_dir)

    # ═══════════════════════════════════════════════════════════════════════
    # WRITE
    # ═══════════════════════════════════════════════════════════════════════

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Store a conversation turn. Returns the document ID."""
        doc_id = str(uuid.uuid4())
        meta = {
            "session_id": session_id,
            "role": role,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }
        # Truncate very long content to keep embeddings meaningful
        text = content[:2000] if len(content) > 2000 else content
        self._store.add_documents(
            [Document(page_content=text, metadata=meta)],
            ids=[doc_id],
        )
        logger.debug("[SESSION_MEMORY] Stored turn (%s) id=%s len=%d", role, doc_id, len(text))
        return doc_id

    def add_agent_action(
        self,
        session_id: str,
        instruction: str,
        diff: str,
        file_path: str,
        passed_tests: Optional[bool] = None,
    ) -> str:
        """Convenience: store an agent edit action with structured metadata."""
        summary = f"INSTRUCTION: {instruction}\nFILE: {file_path}\nDIFF:\n{diff[:800]}"
        meta = {
            "type": "agent_action",
            "file_path": file_path,
        }
        if passed_tests is not None:
            meta["passed_tests"] = str(passed_tests)
        return self.add_turn(session_id, "agent", summary, metadata=meta)

    # ═══════════════════════════════════════════════════════════════════════
    # READ
    # ═══════════════════════════════════════════════════════════════════════

    def recall(
        self,
        session_id: str,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        """Retrieve the most relevant past turns for a given query within a session."""
        # Truncate query to prevent embedding model overflow (nomic-embed-text has 8192 token limit)
        truncated_query = query[:2000] if len(query) > 2000 else query
        # Fetch more than k so we can filter by session_id
        results = self._store.similarity_search(
            truncated_query,
            k=k * 3,
            filter={"session_id": session_id},
        )
        # Already filtered by session_id via Chroma filter
        return results[:k]

    def get_recent(self, session_id: str, n: int = 6) -> List[Document]:
        """Get the N most recent turns in a session (by timestamp order)."""
        # Chroma doesn't support ORDER BY, so fetch a batch and sort client-side
        results = self._store.similarity_search(
            "",  # empty query returns varied results
            k=n * 3,
            filter={"session_id": session_id},
        )
        # Sort by timestamp descending, take last n
        results.sort(key=lambda d: d.metadata.get("timestamp", ""), reverse=True)
        return results[:n]

    # ═══════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════════════

    def build_context_block(self, session_id: str, query: str, k: int = 4) -> str:
        """Build a formatted context string from relevant past turns."""
        # Truncate query for recall (embedding overflow protection)
        docs = self.recall(session_id, query[:2000], k=k)
        if not docs:
            return ""

        lines = ["SESSION MEMORY (relevant past interactions):"]
        for i, doc in enumerate(docs, 1):
            role = doc.metadata.get("role", "unknown").upper()
            ts = doc.metadata.get("timestamp", "")
            content = doc.page_content[:400]
            lines.append(f"  [{i}] {role} ({ts}): {content}")
        lines.append("")
        return "\n".join(lines)

    def clear_session(self, session_id: str) -> int:
        """Delete all turns for a session. Returns count of deleted docs."""
        # Chroma delete with filter
        try:
            results = self._store.get(where={"session_id": session_id})
            if results and results.get("ids"):
                ids = results["ids"]
                self._store.delete(ids=ids)
                logger.info("[SESSION_MEMORY] Cleared %d turns for session %s", len(ids), session_id)
                return len(ids)
        except Exception as e:
            logger.warning("[SESSION_MEMORY] Clear failed: %s", e)
        return 0

    def clear_all(self) -> None:
        """Delete all session memory."""
        try:
            collection = self._store._collection
            collection.delete(where={})
            logger.info("[SESSION_MEMORY] All memory cleared")
        except Exception as e:
            logger.warning("[SESSION_MEMORY] Clear all failed: %s", e)
