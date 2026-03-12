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


# Known topics the system can retrieve docs for
_KNOWN_TOPICS = [
    # Programming languages
    'c++', 'cpp', 'go', 'golang', 'python', 'rust', 'typescript', 'javascript',
    'java', 'blender', 'angular',
    # Software engineering
    'refactoring', 'concurrency', 'threading', 'multithreading',
    'security', 'cryptography', 'database', 'sql', 'networking',
    'design patterns', 'clean code', 'testing',
    # Mathematics
    'calculus', 'derivative', 'integration', 'differentiation', 'limit',
    'linear algebra', 'matrix', 'vector', 'eigenvalue', 'determinant',
    'statistics', 'probability', 'discrete math', 'geometry', 'trigonometry',
    'optimization', 'number theory', 'combinatorics', 'set theory',
    'differential equation', 'partial derivative', 'gradient',
    'taylor series', 'fourier', 'laplace', 'complex analysis',
    'logarithm', 'exponential', 'polynomial', 'quadratic', 'factoring',
    'arithmetic', 'modular arithmetic', 'prime', 'gcd', 'lcm',
    'permutation', 'combination', 'binomial', 'recurrence relation',
    'graph theory', 'topology', 'abstract algebra', 'group theory',
    # Algorithms & data structures
    'algorithm', 'data structure', 'dynamic programming', 'graph',
    'sorting', 'searching', 'binary search', 'two pointers', 'sliding window',
    'greedy', 'backtracking', 'recursion', 'divide and conquer',
    'breadth first search', 'depth first search', 'bfs', 'dfs',
    'dijkstra', 'bellman ford', 'floyd warshall', 'topological sort',
    'minimum spanning tree', 'union find', 'disjoint set',
    'hash table', 'hash map', 'linked list', 'stack', 'queue', 'deque',
    'binary tree', 'binary search tree', 'avl tree', 'red black tree',
    'heap', 'priority queue', 'trie', 'segment tree', 'fenwick tree',
    'bit manipulation', 'string matching', 'kmp', 'rabin karp',
    'memoization', 'tabulation', 'knapsack', 'longest common subsequence',
    'shortest path', 'maximum flow', 'minimum cut',
    'amortized analysis', 'time complexity', 'space complexity',
]

# Topics that are mathematics/science rather than programming
_MATH_TOPICS = {
    # Calculus
    'calculus', 'derivative', 'integration', 'integral', 'differentiation',
    'product rule', 'chain rule', 'quotient rule', 'power rule',
    'limit', 'continuity', 'l\'hopital', 'squeeze theorem',
    'taylor series', 'maclaurin series', 'fourier', 'laplace',
    'differential equation', 'ordinary differential equation', 'ode',
    'partial derivative', 'gradient', 'divergence', 'curl',
    'double integral', 'triple integral', 'line integral', 'surface integral',
    'green\'s theorem', 'stokes theorem', 'divergence theorem',
    'implicit differentiation', 'related rates', 'optimization',
    'riemann sum', 'fundamental theorem of calculus', 'improper integral',
    'arc length', 'parametric', 'polar coordinates',
    # Linear algebra
    'linear algebra', 'matrix', 'matrices', 'vector', 'eigenvalue',
    'eigenvector', 'determinant', 'inverse matrix', 'transpose',
    'gaussian elimination', 'row reduction', 'echelon form',
    'linear transformation', 'vector space', 'basis', 'dimension',
    'dot product', 'cross product', 'norm', 'orthogonal',
    'svd', 'singular value decomposition', 'lu decomposition',
    'rank', 'nullity', 'kernel', 'image', 'span',
    'cramers rule', 'systems of equations',
    # Statistics & probability
    'statistics', 'probability', 'mean', 'median', 'mode', 'variance',
    'standard deviation', 'normal distribution', 'binomial distribution',
    'poisson distribution', 'hypothesis testing', 'confidence interval',
    'regression', 'correlation', 'bayes theorem', 'conditional probability',
    'expected value', 'random variable', 'central limit theorem',
    # Discrete math & number theory
    'discrete math', 'combinatorics', 'permutation', 'combination',
    'number theory', 'modular arithmetic', 'prime', 'gcd', 'lcm',
    'euclidean algorithm', 'fermat', 'euler totient',
    'graph theory', 'pigeonhole principle', 'inclusion exclusion',
    'recurrence relation', 'generating function',
    'set theory', 'boolean algebra', 'propositional logic',
    # Geometry & trig
    'geometry', 'trigonometry', 'sin', 'cos', 'tan',
    'pythagorean', 'unit circle', 'radians',
    'area', 'volume', 'perimeter', 'circumference',
    'congruence', 'similarity', 'transformation',
    # Algebra
    'polynomial', 'quadratic', 'factoring', 'logarithm', 'exponential',
    'complex number', 'imaginary', 'rational expression',
    'arithmetic', 'sequence', 'series', 'summation', 'binomial theorem',
    'complex analysis', 'abstract algebra', 'group theory', 'topology',
}

# Algorithm/data-structure topics for fallback matching when the problem
# doesn't match a dedicated known algorithm
_ALGO_TOPICS = {
    'dynamic programming', 'dp', 'greedy', 'backtracking', 'recursion',
    'divide and conquer', 'binary search', 'two pointers', 'sliding window',
    'breadth first search', 'bfs', 'depth first search', 'dfs',
    'dijkstra', 'bellman ford', 'floyd warshall', 'topological sort',
    'minimum spanning tree', 'kruskal', 'prim', 'union find',
    'hash table', 'hash map', 'linked list', 'stack', 'queue', 'deque',
    'binary tree', 'binary search tree', 'avl tree', 'heap', 'priority queue',
    'trie', 'segment tree', 'fenwick tree', 'bit manipulation',
    'string matching', 'kmp', 'rabin karp', 'suffix array',
    'memoization', 'tabulation', 'knapsack', 'longest common subsequence',
    'shortest path', 'maximum flow', 'minimum cut', 'bipartite matching',
    'sorting', 'merge sort', 'quick sort', 'heap sort', 'counting sort',
    'radix sort', 'bucket sort', 'insertion sort', 'selection sort',
    'graph', 'tree', 'array', 'matrix', 'string',
    'monotonic stack', 'monotonic queue', 'prefix sum', 'difference array',
    'binary indexed tree', 'sparse table', 'disjoint set',
    'a star', 'minimax', 'alpha beta pruning',
    'convex hull', 'sweep line', 'interval scheduling',
}


def _is_math_query(query: str) -> bool:
    """Detect if the query is about mathematics rather than programming."""
    q = query.lower()
    return any(topic in q for topic in _MATH_TOPICS)


def _is_algo_query(query: str) -> bool:
    """Detect if the query is about algorithms/data structures."""
    q = query.lower()
    return any(topic in q for topic in _ALGO_TOPICS)


def _extract_query_topics(query: str) -> list[str]:
    """Return distinct known topics mentioned in the query.

    Falls back to _ALGO_TOPICS and _MATH_TOPICS if no known topic is found
    in the primary list, so we still get useful context for uncommon problems.
    """
    q = query.lower()
    found = []
    for topic in _KNOWN_TOPICS:
        if topic in q and topic not in found:
            found.append(topic)
    # deduplicate aliases
    if 'cpp' in found and 'c++' in found:
        found.remove('cpp')
    if 'golang' in found and 'go' in found:
        found.remove('golang')
    # Fallback: if no known topic matched, check algo and math sets
    if not found:
        for topic in _ALGO_TOPICS | _MATH_TOPICS:
            if topic in q and topic not in found:
                found.append(topic)
    return found


def _is_follow_up_query(query: str) -> bool:
    """Detect if a query is vague / referential and needs conversation context to resolve."""
    q = query.lower().strip()
    words = q.split()

    # Phrases that are almost always follow-ups regardless of length
    strong_follow_up_phrases = [
        'tell me more', 'can you expand', 'explain more', 'explain step',
        'elaborate on', 'more about', 'what do you mean', 'go into more detail',
        'can you explain', 'explain to me more', 'does that exist for',
        'expand on it', 'expand on that', 'these three', 'those three',
        'all three', 'for these', 'for those',
    ]
    for phrase in strong_follow_up_phrases:
        if phrase in q:
            return True

    # Short queries (<= 8 words) with referential words
    if len(words) <= 8:
        referential_words = [
            'step', 'it', 'that', 'this', 'those', 'these', 'above',
            'previous', 'last', 'more', 'detail', 'explain', 'elaborate',
            'again', 'expand', 'clarify',
        ]
        for word in referential_words:
            if word in q:
                return True
    return False


def _can_answer_from_history(query: str, chat_history: list[dict]) -> bool:
    """Check if the last assistant message is a detailed explanation the user is asking about."""
    if not chat_history:
        return False
    q = query.lower()
    # Only trigger for queries referencing specific parts of a prior explanation
    reference_words = ['step', 'iteration', 'example', 'part', 'section', 'point', 'bullet']
    has_reference = any(w in q for w in reference_words)
    if not has_reference:
        return False
    # Find the last assistant message
    for msg in reversed(chat_history):
        if msg['role'] == 'Assistant' and len(msg['content']) > 200:
            return True
    return False


async def _answer_from_history(query: str, chat_history: list[dict], llm_model: str) -> dict:
    """Answer a follow-up question directly from conversation history (no RAG search)."""
    # Get the last substantial assistant message
    last_explanation = ""
    for msg in reversed(chat_history):
        if msg['role'] == 'Assistant' and len(msg['content']) > 200:
            last_explanation = msg['content']
            break

    # Also get the original user question that triggered that explanation
    last_user_query = ""
    for msg in reversed(chat_history):
        if msg['role'] == 'User':
            last_user_query = msg['content']
            break

    llm = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)

    prompt = f"""You are an expert programming assistant.

The user previously asked a question and received a detailed explanation.
Now they are asking a follow-up about a specific part of that explanation.
Answer ONLY from the explanation below — do NOT make up new content or reference unrelated topics.

Original question: {last_user_query}

Previous explanation:
{last_explanation}

User's follow-up question: {query}

Answer:"""

    answer = await llm.ainvoke(prompt)
    print(f"[HISTORY ANSWER] Answered '{query}' from conversation history (no RAG search)")
    return {
        "answer": answer,
        "summary": "(Answered from conversation context)",
        "citations": "(Based on previous explanation)",
        "detailed": answer,
    }


def _rewrite_follow_up_query(query: str, chat_history: list[dict], llm_model: str) -> str:
    """Use the LLM to rewrite a vague follow-up into a self-contained search query."""
    if not chat_history:
        return query

    # Build compact history (last 4 turns)
    recent = chat_history[-4:]
    history_lines = []
    for msg in recent:
        role = msg['role'].upper()
        content = msg['content']
        # Truncate long assistant responses to keep prompt small
        if role == 'ASSISTANT' and len(content) > 400:
            content = content[:400] + '...'
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines)

    prompt = f"""Given this conversation history, rewrite the user's latest question
into a SELF-CONTAINED search query that includes all necessary context.
The rewritten query should be something a search engine can answer without
any prior conversation.

CRITICAL RULES:
- Preserve ALL specific programming languages, technologies, or topics from the conversation.
- If the user was discussing TypeScript, Python, and Rust, the rewritten query MUST mention TypeScript, Python, and Rust — NOT different languages.
- Include the specific technical detail the user is asking about.
- Return ONLY the rewritten query, nothing else.

Conversation:
{history_text}

Latest question: {query}

Rewritten search query:"""

    try:
        llm = OllamaLLM(model=llm_model, temperature=0)
        rewritten = llm.invoke(prompt)
        rewritten = rewritten.strip().strip('"').strip("'")
        if rewritten and len(rewritten) > 5:
            print(f"[FOLLOW-UP REWRITE] '{query}' -> '{rewritten}'")
            return rewritten
    except Exception as e:
        print(f"[FOLLOW-UP REWRITE] Failed: {e}")

    return query


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
    llm = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)

    history_entries = []
    if session_chat_history:
        for msg in session_chat_history[-4:]:
            if len(msg['content']) > 30:
                history_entries.append(f"{msg['role'].upper()}: {msg['content']}")
    recent_history = "\n".join(history_entries)

    # Scale answer length to number of topics in the query
    topics = _extract_query_topics(query)
    if len(topics) >= 2:
        length_instruction = (
            f"- This question covers {len(topics)} topics ({', '.join(topics)}). "
            f"Address EACH one — use a short paragraph or bullet section per topic."
        )
    else:
        length_instruction = "- Keep it to 2-4 sentences maximum"

    # Adapt persona based on query domain
    is_math = _is_math_query(query)
    if is_math:
        persona = """You are an expert mathematics tutor and educator.
Your goal is to help the student UNDERSTAND the concept, not just state a formula.
- Explain the rule/concept in plain language first
- Show the formal definition/formula
- Walk through a concrete example step by step
- Mention when and why this concept is used"""
    else:
        persona = """You are an expert assistant with deep knowledge of software engineering and computer science.
Using the provided context chunks, answer the question directly and concisely.
- If the context contains the answer, use it and cite the source
- If the context is partially relevant, supplement with your own knowledge
- If the documents don't contain enough information, say so honestly."""

    prompt = f"""{persona}
{length_instruction}

previous conversation (if relevant):
{recent_history}

Documents:
{formatted_docs}

Question: {query}

Answer:"""
    return await llm.ainvoke(prompt)


async def summarize_documents(query: str, formatted_docs: str, llm_model: str, session_chat_history: list[dict] | None = None) -> str:
    llm  = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)

    recent_history = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in session_chat_history[-3:] if len(msg['content']) > 50
    ]) if session_chat_history else ""

    is_math = _is_math_query(query)
    if is_math:
        domain_note = "- If this is a math concept, summarize the key formula, when to use it, and one quick example"
    else:
        domain_note = "- Only include points directly relevant to the question"

    prompt = f"""You are an expert assistant.

Summarize the key points relevant to the question using the documents and your own knowledge.
{domain_note}
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
    llm  = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)

    recent_history = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
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
    llm  = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)
    recent_history = session_chat_history[-5:] if session_chat_history else []

    history_str = []
    for msg in recent_history:
        history_str.append(f"{msg['role'].upper()}: {msg['content']}")

    history_text = "\n".join(history_str)

    is_math = _is_math_query(query)
    if is_math:
        persona_instruction = """You are an expert mathematics tutor.

Provide a DETAILED step-by-step explanation that helps the student truly understand the concept.
- Start with the intuition: WHY does this rule/concept exist?
- Show the formal definition with proper notation
- Walk through a COMPLETE worked example with every step shown
- Show a second example that is slightly harder
- Mention common mistakes students make
- If relevant, connect to related concepts (e.g., how product rule relates to chain rule)"""
    else:
        persona_instruction = """You are an expert programming assistant.

Provide a detailed explanation that ADDS NEW INFORMATION beyond the direct answer.
- Do NOT repeat what was already covered in the direct answer
- Include code examples if helpful
- Cover edge cases, gotchas, or practical usage tips"""

    prompt = f"""{persona_instruction}
- If this is a follow-up question, use the PREVIOUS CONVERSATION HISTORY for context
- If the direct answer already covered everything, just add a practical example

PREVIOUS CONVERSATION HISTORY:
{history_text}

Documents:
{formatted_docs}

Question: {query}

Detailed Explanation:"""

    response = await llm.ainvoke(prompt)
    return response


async def fast_pipeline(query: str, llm_model: str, session_chat_history: list[dict] | None = None):
    topics = _extract_query_topics(query)
    is_math = _is_math_query(query)

    # Determine topic filter to boost relevant PDFs
    topic_filter = None
    if is_math:
        topic_filter = ["mathematics"]

    if len(topics) >= 2:
        # Multi-topic query: retrieve docs per-topic to guarantee each topic has coverage
        seen_keys: set = set()
        merged_docs = []
        for topic in topics:
            topic_results = fast_topic_search(topic, topic_filter=topic_filter)
            for doc in topic_results[:4]:  # up to 4 docs per topic
                key = (doc.metadata.get('source', ''), doc.metadata.get('page', ''))
                if key not in seen_keys:
                    seen_keys.add(key)
                    merged_docs.append(doc)
        top_docs = merged_docs[:10]  # cap total
    else:
        results = fast_topic_search(query, topic_filter=topic_filter)
        top_docs = results[:5]

    formatted_docs = _format_search_results_for_prompt(top_docs)

    coros = [
        answer_question(query, formatted_docs, llm_model, session_chat_history),
        summarize_documents(query, formatted_docs, llm_model, session_chat_history),
        cite_documents(query, formatted_docs, llm_model, session_chat_history),
        detailed_answer(query, formatted_docs, llm_model, session_chat_history),
    ]

    answer, summary, citations, detailed = await asyncio.gather(*coros)

    return {
        "answer": answer,
        "summary": summary,
        "citations": citations,
        "detailed": detailed,
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

async def deep_semantic_pipeline(
    query,
    llm_model,
    verbose=False,
    raw_docs=None,
    session_chat_history=None
):
    """
    Deep retrieval using semantic (cross-encoder) reranking.
    Best for complex questions requiring high-quality result ordering.
    """
    global _last_filters

    dynamic_filters = extract_dynamic_filters(query, verbose=verbose)

    if not dynamic_filters and _last_filters:
        dynamic_filters = _last_filters
        if verbose:
            print(f"[DEBUG] No filter found, reusing last: {dynamic_filters}")
    else:
        _last_filters = dynamic_filters

    if verbose:
        print(f"[DEBUG] Running semantic retrieval for: '{query}'")

    raw_docs = raw_docs or load_pdfs(DATA_DIR)

    results = await query_brain(
        question=query,
        k=5,
        filters=dynamic_filters,
        raw_docs=raw_docs,
        verbose=verbose,
        rerank_method="cross_encoder",
        force_semantic_chroma=True
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
    session_chat_history: list[dict] | None = None,
    mode_override: str | None = None  # 'auto','fast','deep','deep_semantic','both'
) -> dict:

    llm_model = llm_model or LLM_MODEL

    # ── Follow-up handling ─────────────────────────────────────────────
    effective_query = query
    is_follow_up = session_chat_history and _is_follow_up_query(query)

    if is_follow_up:
        # If the user is referencing a specific part of a prior explanation
        # ("step 3", "second iteration"), answer directly from history —
        # no RAG search needed because the context lives in the conversation.
        if _can_answer_from_history(query, session_chat_history):
            print(f"[FOLLOW-UP] Answering from conversation history")
            result = await _answer_from_history(query, session_chat_history, llm_model)
            # Store in history
            if session_chat_history is not None:
                session_chat_history.append({"role": "User", "content": query})
                session_chat_history.append({"role": "Assistant", "content": result.get("answer", "")})
            return result

        # Otherwise rewrite the vague follow-up into a self-contained query
        # so BM25 / Chroma can retrieve the right documents.
        effective_query = _rewrite_follow_up_query(query, session_chat_history, llm_model)

    # allow explicit override; otherwise let router choose
    mode = mode_override or route_execution_mode(effective_query)

    if verbose:
        print(f"[ROUTER] Execution mode: {mode}")

    if mode == "fast":
        result = await fast_pipeline(effective_query, llm_model, session_chat_history)
    elif mode == "deep_semantic":
        result = await deep_semantic_pipeline(
            query=effective_query,
            llm_model=llm_model,
            verbose=verbose,
            raw_docs=raw_docs,
            session_chat_history=session_chat_history
        )
    elif mode == "both":
        fast_res = await fast_pipeline(effective_query, llm_model, session_chat_history)
        deep_res = await deep_pipeline(
            query=effective_query,
            llm_model=llm_model,
            verbose=verbose,
            raw_docs=raw_docs,
            session_chat_history=session_chat_history,
        )
        result = {"fast": fast_res, "deep": deep_res}
    else:
        # deep (default)
        result = await deep_pipeline(
            query=effective_query,
            llm_model=llm_model,
            verbose=verbose,
            raw_docs=raw_docs,
            session_chat_history=session_chat_history
        )

    # ── Store the ORIGINAL query and the answer in chat history ──────
    # so follow-up questions can see what the user actually asked.
    if session_chat_history is not None:
        session_chat_history.append({"role": "User", "content": query})
        # For "both" mode, store the deep answer; otherwise the main answer
        answer_text = ""
        if isinstance(result, dict):
            if "deep" in result:
                answer_text = result["deep"].get("answer", "")
            else:
                answer_text = result.get("answer", "")
        if answer_text:
            session_chat_history.append({"role": "Assistant", "content": answer_text})

    return result