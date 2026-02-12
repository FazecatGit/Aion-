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

# Simple query prompt (no context)
SIMPLE_PROMPT = PromptTemplate(
    input_variables=["input"],
    template="Answer this question: {input}"
)

# Summarization prompt
SUMMARIZE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text concisely:\n\n{text}"
)

# Question generation (for testing)
GENERATE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="Generate 3 important questions about this text:\n\n{text}"
)

# Strict RAG prompt that enforces context-only answers
STRICT_RAG_PROMPT = ChatPromptTemplate.from_template(
    """You MUST ONLY answer using the provided context. Do NOT make up information.

Context:
{context}

Question: {input}

If the answer is not found in the context, respond EXACTLY with: "I don't have that information in the documents."
Answer:"""
)