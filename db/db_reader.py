import chromadb
from chromadb.config import Settings
from brain.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


def get_all_code_snippets(limit: int | None = None) -> list[dict]:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

    results = collection.get(
        include=["documents", "metadatas", "ids"],
        limit=limit,
    )

    docs = []
    for idx, doc_id in enumerate(results["ids"]):
        docs.append({
            "id": doc_id,
            "text": results["documents"][idx],
            "metadata": results["metadatas"][idx],
        })
    return docs

# Alias for compatibility with code_agent.py
get_code_documents = get_all_code_snippets
