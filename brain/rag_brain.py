from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .prompts import RAG_PROMPT
from .config import (
    DATA_DIR, FAISS_DIR, EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE,
    RETRIEVAL_K, CHUNK_SIZE, CHUNK_OVERLAP
)

def ingest_docs():
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(FAISS_DIR)
    print(f"Ingested {len(splits)} chunks.")

def query_brain(question: str):
    prompt = RAG_PROMPT
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    docs = retriever.invoke(question)

    print("RETRIEVED:", len(docs))
    for d in docs:
        print(d.metadata)
        print(d.page_content[:300])
        print("----")

    llm = OllamaLLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    
    prompt = PromptTemplate.from_template(
        """You MUST ONLY answer using the provided context. Do NOT make up information.

Context:
{context}

Question: {input}

If the answer is not found in the context, respond EXACTLY with: "I don't have that information in the documents."
Answer:"""
    )
    
    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(question)

if __name__ == "__main__":
    print("1: Ingest | 2: Query")
    choice = input("Choose: ")
    if choice == "1":
        ingest_docs()
    else:
        print("Query mode. Type 'quit' to exit.\n")
        while True:
            q = input("Ask: ").strip()
            if q.lower() == "quit":
                print("Goodbye!")
                break
            if q:
                result = query_brain(q)
                print(f"\nAnswer: {result}\n")
