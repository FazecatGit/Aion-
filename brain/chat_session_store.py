"""
Persistent Chat Session Store — Chroma-backed with smart summarization.

Each "session" is a named conversation (like a chat tab).
Sessions survive server restarts via Chroma persistence.

When a session gets long, older turns are summarized into a compact
digest so the LLM context window doesn't overflow, but the full
history remains searchable in Chroma for semantic recall.
"""

import logging
import uuid
import json
from datetime import datetime
from typing import List, Dict, Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document

from brain.config import EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger("chat_session_store")

_PERSIST_DIR = "cache/chat_sessions_db"
_COLLECTION  = "aion-chat-sessions"

# How many recent turns to keep verbatim (the rest get summarized)
_RECENT_WINDOW = 8
# Maximum turns before triggering a summarization pass
_SUMMARIZE_THRESHOLD = 16


class ChatSessionStore:
    """Chroma-backed persistent chat session store."""

    def __init__(self, persist_dir: str = _PERSIST_DIR):
        self._embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self._store = Chroma(
            collection_name=_COLLECTION,
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
        )
        logger.info("[CHAT_STORE] Initialized (persist=%s)", persist_dir)

    # ── Session management ───────────────────────────────────────────────

    def list_sessions(self) -> List[Dict]:
        """Return all sessions with their metadata."""
        try:
            results = self._store.get(where={"type": "session_meta"})
            sessions = []
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"]):
                    meta = results["metadatas"][i] if results.get("metadatas") else {}
                    sessions.append({
                        "session_id": meta.get("session_id", ""),
                        "title": meta.get("title", "Untitled"),
                        "created_at": meta.get("created_at", ""),
                        "last_active": meta.get("last_active", ""),
                        "turn_count": int(meta.get("turn_count", 0)),
                    })
            # Sort by last_active descending
            sessions.sort(key=lambda s: s.get("last_active", ""), reverse=True)
            return sessions
        except Exception as e:
            logger.warning("[CHAT_STORE] list_sessions failed: %s", e)
            return []

    def create_session(self, title: str = "New Chat") -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        doc = Document(
            page_content=f"Session: {title}",
            metadata={
                "type": "session_meta",
                "session_id": session_id,
                "title": title,
                "created_at": now,
                "last_active": now,
                "turn_count": "0",
            },
        )
        self._store.add_documents([doc], ids=[f"meta_{session_id}"])
        logger.info("[CHAT_STORE] Created session %s: %s", session_id, title)
        return session_id

    def rename_session(self, session_id: str, new_title: str) -> bool:
        """Rename a session."""
        try:
            self._store.update_document(
                f"meta_{session_id}",
                Document(
                    page_content=f"Session: {new_title}",
                    metadata={
                        "type": "session_meta",
                        "session_id": session_id,
                        "title": new_title,
                        "last_active": datetime.now().isoformat(),
                    },
                ),
            )
            return True
        except Exception as e:
            logger.warning("[CHAT_STORE] rename failed: %s", e)
            return False

    def delete_session(self, session_id: str) -> int:
        """Delete a session and all its turns. Returns count of deleted docs."""
        try:
            results = self._store.get(where={"session_id": session_id})
            if results and results.get("ids"):
                ids = results["ids"]
                # Also include the meta doc
                meta_id = f"meta_{session_id}"
                if meta_id not in ids:
                    ids.append(meta_id)
                self._store.delete(ids=ids)
                logger.info("[CHAT_STORE] Deleted %d docs for session %s", len(ids), session_id)
                return len(ids)
        except Exception as e:
            logger.warning("[CHAT_STORE] delete failed: %s", e)
        return 0

    # ── Turn storage ─────────────────────────────────────────────────────

    def add_turn(self, session_id: str, role: str, content: str) -> str:
        """Store a single conversation turn. Returns the doc ID."""
        doc_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        text = content[:4000] if len(content) > 4000 else content
        doc = Document(
            page_content=text,
            metadata={
                "type": "turn",
                "session_id": session_id,
                "role": role,
                "timestamp": now,
                "turn_id": doc_id,
            },
        )
        self._store.add_documents([doc], ids=[doc_id])

        # Update session metadata (last_active, turn_count)
        try:
            meta_results = self._store.get(ids=[f"meta_{session_id}"])
            if meta_results and meta_results.get("metadatas"):
                old_meta = meta_results["metadatas"][0]
                count = int(old_meta.get("turn_count", 0)) + 1
                self._store.update_document(
                    f"meta_{session_id}",
                    Document(
                        page_content=f"Session: {old_meta.get('title', 'Untitled')}",
                        metadata={
                            **old_meta,
                            "last_active": now,
                            "turn_count": str(count),
                        },
                    ),
                )
        except Exception:
            pass

        return doc_id

    def get_turns(self, session_id: str) -> List[Dict]:
        """Get all turns for a session, ordered by timestamp."""
        try:
            results = self._store.get(
                where={"$and": [{"session_id": session_id}, {"type": "turn"}]}
            )
            if not results or not results.get("documents"):
                return []

            turns = []
            for i, doc_text in enumerate(results["documents"]):
                meta = results["metadatas"][i] if results.get("metadatas") else {}
                turns.append({
                    "role": meta.get("role", "unknown"),
                    "content": doc_text,
                    "timestamp": meta.get("timestamp", ""),
                    "turn_id": meta.get("turn_id", ""),
                })
            # Sort by timestamp
            turns.sort(key=lambda t: t.get("timestamp", ""))
            return turns
        except Exception as e:
            logger.warning("[CHAT_STORE] get_turns failed: %s", e)
            return []

    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get turns formatted as session_chat_history [{role, content}]."""
        turns = self.get_turns(session_id)
        return [{"role": t["role"], "content": t["content"]} for t in turns]

    # ── Semantic recall ──────────────────────────────────────────────────

    def recall(self, session_id: str, query: str, k: int = 4) -> List[Dict]:
        """Find the most relevant past turns for the given query within a session."""
        try:
            results = self._store.similarity_search(
                query,
                k=k * 3,
                filter={"$and": [{"session_id": session_id}, {"type": "turn"}]},
            )
            # Already filtered by session_id
            relevant = results[:k]
            return [
                {"role": doc.metadata.get("role", "unknown"), "content": doc.page_content}
                for doc in relevant
            ]
        except Exception as e:
            logger.warning("[CHAT_STORE] recall failed: %s", e)
            return []

    # ── Smart summarization ──────────────────────────────────────────────

    def get_context_window(self, session_id: str, query: str = "") -> List[Dict]:
        """
        Build a smart context window for the LLM:
        - If total turns <= _RECENT_WINDOW: return all turns verbatim
        - If turns > _RECENT_WINDOW: summarize older turns + keep recent verbatim
        - If query provided: also include semantically relevant older turns

        Returns a list suitable for session_chat_history.
        """
        all_turns = self.get_turns(session_id)

        if len(all_turns) <= _RECENT_WINDOW:
            return [{"role": t["role"], "content": t["content"]} for t in all_turns]

        # Split: older turns to summarize + recent turns to keep verbatim
        older = all_turns[:-_RECENT_WINDOW]
        recent = all_turns[-_RECENT_WINDOW:]

        context: List[Dict] = []

        # Check if we already have a summary stored
        summary = self._get_stored_summary(session_id)
        if not summary:
            summary = self._summarize_turns(older)
            if summary:
                self._store_summary(session_id, summary)

        if summary:
            context.append({"role": "System", "content": f"[CONVERSATION SUMMARY]\n{summary}"})

        # If query provided, also retrieve semantically relevant older turns
        if query:
            relevant = self.recall(session_id, query, k=3)
            for r in relevant:
                # Skip if it's already in the recent window
                if not any(
                    t["content"] == r["content"] for t in recent
                ):
                    context.append(r)

        # Add recent turns verbatim
        for t in recent:
            context.append({"role": t["role"], "content": t["content"]})

        return context

    def _summarize_turns(self, turns: List[Dict]) -> str:
        """Use LLM to create a compact summary of older conversation turns."""
        if not turns:
            return ""

        conversation = "\n".join(
            f"{t['role'].upper()}: {t['content'][:300]}"
            for t in turns
        )

        try:
            llm = OllamaLLM(model=LLM_MODEL, temperature=0)
            prompt = f"""Summarize this conversation history into a compact paragraph.
Preserve:
- Key technical topics discussed (languages, frameworks, algorithms)
- Important decisions or conclusions reached
- File paths and specific code constructs mentioned
- Any preferences or requirements the user expressed

Do NOT include greetings or small talk. Focus on technical substance.
Maximum 200 words.

Conversation:
{conversation[:3000]}

Summary:"""
            summary = llm.invoke(prompt).strip()
            logger.info("[CHAT_STORE] Summarized %d older turns (%d chars)", len(turns), len(summary))
            return summary
        except Exception as e:
            logger.warning("[CHAT_STORE] Summarization failed: %s", e)
            return ""

    def _get_stored_summary(self, session_id: str) -> str:
        """Retrieve a previously stored summary for this session."""
        try:
            results = self._store.get(
                where={"$and": [{"session_id": session_id}, {"type": "summary"}]}
            )
            if results and results.get("documents"):
                return results["documents"][0]
        except Exception:
            pass
        return ""

    def _store_summary(self, session_id: str, summary: str):
        """Store (or update) a conversation summary for this session."""
        doc_id = f"summary_{session_id}"
        try:
            # Try to update existing
            existing = self._store.get(ids=[doc_id])
            if existing and existing.get("documents"):
                self._store.update_document(
                    doc_id,
                    Document(
                        page_content=summary,
                        metadata={
                            "type": "summary",
                            "session_id": session_id,
                            "timestamp": datetime.now().isoformat(),
                        },
                    ),
                )
            else:
                self._store.add_documents(
                    [Document(
                        page_content=summary,
                        metadata={
                            "type": "summary",
                            "session_id": session_id,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )],
                    ids=[doc_id],
                )
        except Exception as e:
            logger.debug("[CHAT_STORE] Store summary failed: %s", e)

    def should_summarize(self, session_id: str) -> bool:
        """Check if a session has grown past the summarization threshold."""
        turns = self.get_turns(session_id)
        return len(turns) > _SUMMARIZE_THRESHOLD

    def trigger_summarization(self, session_id: str):
        """Force re-summarize older turns in this session."""
        all_turns = self.get_turns(session_id)
        if len(all_turns) <= _RECENT_WINDOW:
            return
        older = all_turns[:-_RECENT_WINDOW]
        summary = self._summarize_turns(older)
        if summary:
            self._store_summary(session_id, summary)
