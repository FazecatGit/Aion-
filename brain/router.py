import os
import json
from pathlib import Path
from langchain_ollama import OllamaLLM
from .config import LLM_MODEL, DATA_DIR

_KEYWORD_CACHE = None

def build_keyword_map(topic_map: dict) -> dict:
    global _KEYWORD_CACHE
    if _KEYWORD_CACHE is not None:
        return _KEYWORD_CACHE
    
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)
    filenames = list(topic_map.values())
    
    prompt = f"""You are a keyword mapping assistant.
For each document filename below, output a JSON dictionary where:
- The key is the document filename (exact, unchanged)
- The value is a list of 3-5 short keywords a user might type to refer to that document

Filenames:
{json.dumps(filenames, indent=2)}

Example output format:
{{
  "python_tutorial.pdf": ["python", "py", "print", "loop"],
  "Effective_Go.pdf": ["go", "golang", "goroutine", "concurrency"]
}}

OUTPUT ONLY THE JSON DICTIONARY:"""

    try:
        response = llm.invoke(prompt).strip()
        if response.startswith("```json"):
            response = response.strip("```json").strip("```").strip()
        _KEYWORD_CACHE = json.loads(response)
        print(f"[DEBUG] Keyword map built: {list(_KEYWORD_CACHE.keys())}")
        return _KEYWORD_CACHE
    except Exception as e:
        print(f"[DEBUG] Keyword map failed: {e}")
        return {}

def keyword_pre_filter(query: str, topic_map: dict) -> list[str]:
    keyword_map = build_keyword_map(topic_map)
    query_lower = query.lower()
    matched_ids = []
    
    for doc_id, filename in topic_map.items():
        keywords = keyword_map.get(filename, [])
        if any(kw.lower() in query_lower for kw in keywords):
            matched_ids.append(doc_id)
    
    return matched_ids

def get_dynamic_topic_list() -> list[str]:
    topics = set()
    if os.path.exists(DATA_DIR):
        for file in os.listdir(DATA_DIR):
            if file.endswith(".pdf"):
                topics.add(file)
    return list(topics)

def extract_dynamic_filters(query: str, verbose: bool = False) -> dict:
    available_topics = get_dynamic_topic_list()
    topic_map = {str(i): topic for i, topic in enumerate(available_topics)}
    
    matched_ids = keyword_pre_filter(query, topic_map)
    
    if verbose:
        print(f"[DEBUG] Keyword pre-filter matched IDs: {matched_ids}")
    
    if not matched_ids:
        llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)
        prompt = f"""You are a strict data routing assistant. 
Analyze the user query: "{query}"

Available documents:
{json.dumps(topic_map, indent=2)}

Return a JSON list of ID numbers. Pick ALL documents that match the topics mentioned.
If the user asks about "python AND go", pick BOTH the python document AND the go document.
If unsure, output: []

OUTPUT ONLY THE JSON LIST:"""

        try:
            response = llm.invoke(prompt).strip()
            if response.startswith("```json"):
                response = response.strip("```json").strip("```").strip()
            matched_ids = json.loads(response)
            if isinstance(matched_ids, str):
                matched_ids = [matched_ids]
        except Exception as e:
            print(f"[DEBUG] Router LLM fallback failed: {e}")
            return {}
    
    matched_docs = [topic_map[doc_id] for doc_id in matched_ids if doc_id in topic_map]
    
    if not matched_docs:
        return {}
    
    full_paths = [os.path.join(DATA_DIR, doc) for doc in matched_docs]
    
    if verbose:
        print(f"[DEBUG] Filter paths: {full_paths}")
    
    return {"source": {"$in": full_paths}}
