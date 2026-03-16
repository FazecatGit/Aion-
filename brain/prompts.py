from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# System prompt for RAG
RAG_SYSTEM_PROMPT = """You are an AI assistant that answers questions based on provided documents.
Use ONLY the context provided to answer questions accurately.
If the answer is not in the context, say "I don't have that information in the documents."
Be concise and helpful."""

# RAG Query Prompt
RAG_PROMPT = ChatPromptTemplate.from_template(
    """Context from documents:
{context}

Question: {input}

Answer based only on the context above:"""
)

# Alternative: More detailed RAG prompt
DETAILED_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", """Context:
{context}

Question: {input}

Provide a detailed answer with sources when possible.""")
])

SIMPLE_PROMPT = PromptTemplate(
    input_variables=["input"],
    template="Answer this question: {input}"
)

SUMMARIZE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text concisely:\n\n{text}"
)

GENERATE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="Generate 3 important questions about this text:\n\n{text}"
)

STRICT_RAG_PROMPT = ChatPromptTemplate.from_template(
    """You MUST ONLY answer using the provided context. Do NOT make up information.

If the context does NOT contain relevant information to answer the question, you MUST respond EXACTLY with:
"I don't have that information in the documents."

Context:
{context}

Question: {input}

Answer:"""
)


QUERY_SPELL_CORRECTION_PROMPT = """Correct spelling mistakes in this user question for document retrieval.
Return only the corrected question text, no explanations.

Question: {query}"""


QUERY_REWRITE_PROMPT = """Rewrite this question to improve retrieval quality against internal PDFs.
Preserve intent and keep it concise. Return only the rewritten question.

Question: {query}"""


QUERY_EXPANSION_PROMPT = """Expand this question with a few short related terms and synonyms for retrieval.
Keep it focused and under 20 extra words. Return only expanded query text.

Question: {query}"""


CODE_QUERY_EXPANSION_PROMPT = """You are helping a code agent retrieve relevant algorithm documentation.

Given a coding problem description, output 3-8 precise technical keywords that would appear in an algorithm textbook covering the solution technique. Focus on:
- The algorithmic technique (e.g. "dynamic programming", "segment tree", "lazy propagation")
- The mathematical concept (e.g. "modular inverse", "prefix sum", "recurrence relation")
- The data structure (e.g. "monotonic stack", "trie", "union-find")

Return ONLY the keywords separated by spaces, no explanations.

Problem: {query}"""


def build_spell_correction_prompt(query: str) -> str:
    return QUERY_SPELL_CORRECTION_PROMPT.format(query=query)


def build_query_rewrite_prompt(query: str) -> str:
    return QUERY_REWRITE_PROMPT.format(query=query)


def build_query_expansion_prompt(query: str) -> str:
    return QUERY_EXPANSION_PROMPT.format(query=query)


def build_code_query_expansion_prompt(query: str) -> str:
    return CODE_QUERY_EXPANSION_PROMPT.format(query=query)