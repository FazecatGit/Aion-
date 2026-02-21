import json
from langchain_ollama import OllamaLLM

from .config import LLM_MODEL, LLM_TEMPERATURE, DATA_DIR
from .pdf_utils import load_pdfs
from .ingest import fast_topic_search

def _run_fast_search(query: str, verbose: bool = False) -> list:
    documents = load_pdfs(DATA_DIR)
        
    if verbose: print(f"\n[DEBUG] Running fast search for: '{query}'")
    
    return fast_topic_search(query, documents)

def _format_search_results_for_prompt(results: list) -> str:
    if not results:
        return ""
        
    formatted = []
    for i, doc in enumerate(results, 1):
        content = doc['content'][:500]
        source = doc['metadata'].get("source", "Unknown")
        page = doc['metadata'].get("page", "?")
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
    prompt = f"""Find and extract the most relevant citations from the provided documents that answer this query.

Query: {query}

Documents:
{formatted_docs}

Instructions:
- Extract EXACT, VERBATIM quotes from the documents. Do not change the wording.
- Format each citation like this: "[Source File Name, Page X] 'exact quote goes here'"
- Include at least 2-3 key citations if they exist.
- Do not make up quotes. If there are no relevant quotes, say "No relevant quotes found."

Citations:"""
    return llm.invoke(prompt)

def detailed_answer(query: str, formatted_docs: str, llm_model: str) -> str:
    llm = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)
    prompt = f"""Provide a detailed, comprehensive answer to this question.
Question: {query}
Documents:
{formatted_docs}

Instructions:
- Start with a brief introduction
- Provide a thorough, multi-paragraph answer
- Cover different angles and perspectives
- End with key takeaways or conclusions
Answer:"""
    return llm.invoke(prompt)

def query_brain_comprehensive(query: str, llm_model: str = None, verbose: bool = False) -> dict:
    llm_model = llm_model or LLM_MODEL

    results = _run_fast_search(query, verbose=verbose)
    
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
