import json
from langchain_ollama import OllamaLLM

from .config import LLM_MODEL, LLM_TEMPERATURE, DATA_DIR
from .pdf_utils import load_pdfs
from .ingest import fast_topic_search
from .rag_brain import hybrid_retrieval
from langchain_core.documents import Document

session_chat_history = []

def _run_fast_search(query: str, verbose: bool = False) -> list:
    documents = load_pdfs(DATA_DIR)
        
    if verbose: print(f"\n[DEBUG] Running fast search for: '{query}'")
    
    return fast_topic_search(query, documents)

def _format_search_results_for_prompt(results: list) -> str:
    if not results:
        return ""
        
    formatted = []
    for i, doc in enumerate(results, 1):
        if isinstance(doc, dict):
            content = doc.get("content", doc.get("text", ""))[:500]
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "?")
            
        else:
            content = getattr(doc, "page_content", "")[:500]
            metadata = getattr(doc, "metadata", {})
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "?")
            
        formatted.append(f"--- Document [{i}] ---\nSource File: {source} (Page {page})\nContent: {content}")
    
    return "\n\n".join(formatted)


def answer_question(query: str, formatted_docs: str, llm_model: str) -> str:
    llm = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)
    prompt = f"""Answer the following question based on the provided documents.
Question: {query}
Documents:
{formatted_docs}

Instructions:
- Provide a direct, concise answer
- Use only information from the documents
- If the answer isn't in the documents, say "I don't have enough information"
Answer:"""
    return llm.invoke(prompt)

def summarize_documents(query: str, formatted_docs: str, llm_model: str) -> str:
    llm = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)
    prompt = f"""Provide a comprehensive summary of the following documents that addresses this query.
Query: {query}
Documents:
{formatted_docs}

Instructions:
- Create a cohesive summary combining information from all documents
- Highlight key points and insights
Summary:"""
    return llm.invoke(prompt)

def cite_documents(query: str, formatted_docs: str, llm_model: str) -> str:
    """Uses the LLM to extract the exact quotes and sources used to answer the query."""
    llm = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)
    prompt = f"""You are a citation extractor. Your job is to show the user exactly which documents were used for context.

Query: {query}

Documents:
{formatted_docs}

Instructions:
- Look at the Documents provided above. 
- For EVERY document source provided, extract one relevant sentence as an EXACT, VERBATIM quote.
- Do NOT evaluate if the quote perfectly answers the query. If the document is in the context, you MUST extract a quote from it.
- Format each citation strictly like this: "[Source File Name, Page X] 'exact quote goes here'"

Citations:"""
    
    return llm.invoke(prompt)

def detailed_answer(query: str, formatted_docs: str, llm_model: str) -> str:
    global session_chat_history
    llm = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)

    recent_history = session_chat_history[-5:]

    history_str = []
    for msg in recent_history:
        content = msg['content']
        if msg['role'] == 'Assistant' and len(content) > 300:
            content = content[:300] + "..."
        history_str.append(f"{msg['role'].upper()}: {content}")

    history_text = "\n".join(history_str)

    prompt = f"""Provide a detailed, comprehensive answer to this question.

PREVIOUS CONVERSATION HISTORY:
{history_text}

NEW QUESTION: {query}
DOCUMENTS:
{formatted_docs}

Instructions:
- Start with a brief introduction
- Provide a thorough, multi-paragraph answer
- Cover different angles and perspectives
- If this is a follow-up question, use the PREVIOUS CONVERSATION HISTORY to understand the context!
- End with key takeaways or conclusions
Answer:"""
    
    response = llm.invoke(prompt)
    session_chat_history.append({"role": "User", "content": query})
    session_chat_history.append({"role": "Assistant", "content": response})
    return response

def query_brain_comprehensive(query: str, llm_model: str = None, verbose: bool = False) -> dict:
    llm_model = llm_model or LLM_MODEL

    if verbose: print(f"[DEBUG] Running hybrid retrieval for: '{query}'")
    results = hybrid_retrieval(query, k=5, verbose=verbose)
    
    if not results:
        error_msg = "I don't have enough information in the provided documents."
        return {
            'answer': error_msg,
            'summary': error_msg,
            'citations': "None",
            'detailed': error_msg
        }

    formatted_docs = _format_search_results_for_prompt(results)
    
    if verbose: print(f"[DEBUG] Found {len(results)} chunks. Generating responses...")

    return {
        'answer': answer_question(query, formatted_docs, llm_model),
        'summary': summarize_documents(query, formatted_docs, llm_model),
        'citations': cite_documents(query, formatted_docs, llm_model), 
        'detailed': detailed_answer(query, formatted_docs, llm_model)
    }
