import asyncio

from langchain_ollama import OllamaLLM

from brain.pdf_utils import load_pdfs
from brain.route_execution import route_execution_mode
from brain.router import extract_dynamic_filters

from .config import LLM_MODEL, LLM_TEMPERATURE, DATA_DIR
from .fast_search import fast_topic_search
from .rag_brain import query_brain
from .query_pipeline import evaluate_documents_with_llm
from pathlib import Path

session_chat_history = []
_last_filters        = {}


def _format_search_results_for_prompt(results: list) -> str:
    if not results:
        return ""

    formatted = []
    for i, doc in enumerate(results, 1):
        if isinstance(doc, dict):
            content  = doc.get("content", doc.get("text", ""))[:500]
            metadata = doc.get("metadata", {})
            source   = metadata.get("source", "Unknown")
            page     = metadata.get("page", "?")
        else:
            content  = getattr(doc, "page_content", "")[:500]
            metadata = getattr(doc, "metadata", {})
            source   = metadata.get("source", "Unknown")
            page     = metadata.get("page", "?")

        source = Path(source).stem
        formatted.append(f"--- Document [{i}] ---\nSource File: {source} (Page {page})\nContent: {content}")

    return "\n\n".join(formatted)


async def answer_question(query: str, formatted_docs: str, llm_model: str, session_chat_history: list[dict] | None = None) -> str:
    llm    = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)

    recent_history = "\n".join([
        f"{msg['role'].upper()}: {msg['content'][:200]}..."
        for msg in session_chat_history[-3:] if len(msg['content']) > 50
    ]) if session_chat_history else ""

    prompt = f"""You are an expert programming assistant with deep knowledge of software engineering.

Using the provided context chunks, answer the question directly and concisely.
- If the context contains the answer, use it and cite the source
- If the context is partially relevant, supplement with your own knowledge
- If the documents don't contain enough information, say so honestly.
- Keep it to 2-4 sentences maximum

previous conversation (if relevant):
{recent_history}

Documents:
{formatted_docs}

Question: {query}

Answer:"""
    return await llm.ainvoke(prompt)


async def summarize_documents(query: str, formatted_docs: str, llm_model: str, session_chat_history: list[dict] | None = None) -> str:
    llm    = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)

    recent_history = "\n".join([
        f"{msg['role'].upper()}: {msg['content'][:200]}..."
        for msg in session_chat_history[-3:] if len(msg['content']) > 50
    ]) if session_chat_history else ""


    prompt = f"""You are an expert programming assistant.

Summarize the key points relevant to the question using the documents and your own knowledge.
- Only include points directly relevant to the question
- Do not repeat what was already said in the direct answer
- NEVER mention what the documents do or do not contain — just answer
- If the documents are thin on this topic, use your own expertise to fill the gaps
- Maximum 3 bullet points

previous conversation (if relevant):
{recent_history}

Documents:
{formatted_docs}

Question: {query}

Summary:"""
    return await llm.ainvoke(prompt)

async def cite_documents(query: str, formatted_docs: str, llm_model: str, session_chat_history: list[dict] | None = None) -> str:
    llm    = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)

    recent_history = "\n".join([
        f"{msg['role'].upper()}: {msg['content'][:200]}..."
        for msg in session_chat_history[-3:] if len(msg['content']) > 50
    ]) if session_chat_history else ""

    prompt = f"""Extract only the citations that are directly relevant to answering this question.
- Ignore chunks that are only loosely related
- Format: [Book Title, Page X] "exact quote"
- Maximum 3 citations
- If a chunk is not directly about the question topic, skip it entirely

previous conversation (if relevant):
{recent_history}

Documents:
{formatted_docs}

Question: {query}

Citations:"""
    return await llm.ainvoke(prompt)


async def detailed_answer(query: str, formatted_docs: str, llm_model: str, session_chat_history: list[dict] | None = None) -> str:
    llm            = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)
    recent_history = session_chat_history[-5:] if session_chat_history else []

    history_str = []
    for msg in recent_history:
        content = msg['content']
        if msg['role'] == 'Assistant' and len(content) > 300:
            content = content[:300] + "..."
        history_str.append(f"{msg['role'].upper()}: {content}")

    history_text = "\n".join(history_str)

    prompt = f"""You are an expert programming assistant.

Provide a detailed explanation that ADDS NEW INFORMATION beyond the direct answer.
- Do NOT repeat what was already covered in the direct answer
- Include code examples if helpful
- Cover edge cases, gotchas, or practical usage tips
- If this is a follow-up question, use the PREVIOUS CONVERSATION HISTORY for context
- If the direct answer already covered everything, just add a practical example

PREVIOUS CONVERSATION HISTORY:
{history_text}

Documents:
{formatted_docs}

Question: {query}

Detailed Explanation:"""

    response = await llm.ainvoke(prompt)
    if session_chat_history is not None:
        session_chat_history.append({"role": "User",      "content": query})
        session_chat_history.append({"role": "Assistant", "content": response})
    return response


async def fast_pipeline(query: str, llm_model: str):
    results = fast_topic_search(query)
    top_docs = results[:3]
    formatted_docs = _format_search_results_for_prompt(top_docs)

    citations_list = [f"[{i+1}] {doc.metadata.get('source', 'Unknown')}" 
                     for i, doc in enumerate(top_docs)]
    citations = " | ".join(citations_list)

    llm = OllamaLLM(model=llm_model, temperature=0)
    
    prompt = f"""Answer concisely (max 4 sentences). Include code example if relevant.

Documents:
{formatted_docs}

Question: {query}
Cite using [1], [2], [3] after relevant sentences."""
    
    answer = await llm.ainvoke(prompt)
    
    summary = (answer.split('.')[0] + "..." if '.' in answer else answer[:100] + "...")
    detailed = f"BM25 top: {results[0].metadata.get('score', 'N/A'):.1f}, {len(results)} chunks"

    return {
        "answer": answer,
        "summary": summary,
        "citations": citations, 
        "detailed": detailed
    }

async def deep_pipeline(
    query,
    llm_model,
    verbose=False,
    raw_docs=None,
    session_chat_history=None
):
    global _last_filters

    dynamic_filters = extract_dynamic_filters(query, verbose=verbose)

    if not dynamic_filters and _last_filters:
        dynamic_filters = _last_filters
        if verbose:
            print(f"[DEBUG] No filter found, reusing last: {dynamic_filters}")
    else:
        _last_filters = dynamic_filters

    if verbose:
        print(f"[DEBUG] Running hybrid retrieval for: '{query}'")

    raw_docs = raw_docs or load_pdfs(DATA_DIR)

    results = await query_brain(
        question=query,
        k=5,
        filters=dynamic_filters,
        raw_docs=raw_docs,
        verbose=verbose
    )

    if not results:
        error_msg = "I don't have enough information in the provided documents."
        return {
            "answer": error_msg,
            "summary": error_msg,
            "citations": "None",
            "detailed": error_msg
        }

    formatted_docs = _format_search_results_for_prompt(results)

    coros = [
        answer_question(query, formatted_docs, llm_model, session_chat_history),
        summarize_documents(query, formatted_docs, llm_model, session_chat_history),
        cite_documents(query, formatted_docs, llm_model, session_chat_history),
        detailed_answer(query, formatted_docs, llm_model, session_chat_history)
    ]

    answer, summary, citations, detailed = await asyncio.gather(*coros)

    return {
        "answer": answer,
        "summary": summary,
        "citations": citations,
        "detailed": detailed
    }

async def query_brain_comprehensive(
    query: str,
    llm_model: str = None,
    verbose: bool = False,
    raw_docs: list[dict] | None = None,
    session_chat_history: list[dict] | None = None
) -> dict:

    llm_model = llm_model or LLM_MODEL

    mode = route_execution_mode(query)

    if verbose:
        print(f"[ROUTER] Execution mode: {mode}")

    if mode == "fast":
        return await fast_pipeline(query, llm_model)

    return await deep_pipeline(
        query=query,
        llm_model=llm_model,
        verbose=verbose,
        raw_docs=raw_docs,
        session_chat_history=session_chat_history
    )