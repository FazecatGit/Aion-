from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from .prompts import RAG_PROMPT, STRICT_RAG_PROMPT
from .config import (
    DATA_DIR, FAISS_DIR, EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE,
    RETRIEVAL_K, CHUNK_SIZE, CHUNK_OVERLAP
)
from .keyword_search import search_documents
from .pdf_utils import load_pdfs

def ingest_docs():
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(FAISS_DIR)
    print(f"Ingested {len(splits)} chunks.")

def hybrid_retrieval(question: str, k: int = RETRIEVAL_K):

    # Semantic search via FAISS
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    semantic_docs = semantic_retriever.invoke(question)
    
    # Keyword search
    pdf_docs = load_pdfs(DATA_DIR)
    keyword_docs = search_documents(question, pdf_docs, n_results=k)
    
    # Convert keyword results to Document objects
    keyword_docs_obj = [
        Document(page_content=doc['content'], metadata=doc['metadata'])
        for doc in keyword_docs
    ]
    
    # Combine and deduplicate (keep order of semantic results and then add unique keywords into results)
    seen_content = set()
    combined = []
    
    for doc in semantic_docs:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in seen_content:
            combined.append(doc)
            seen_content.add(content_hash)
    
    for doc in keyword_docs_obj:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in seen_content:
            combined.append(doc)
            seen_content.add(content_hash)
    
    return combined[:k]

def query_brain(question: str, verbose: bool = False):
    docs = hybrid_retrieval(question)

    if verbose:
        print("RETRIEVED:", len(docs))
        for d in docs:
            print(d.metadata)
            print(d.page_content[:300])
            print("----")

    llm = OllamaLLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    
    # Format context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    result = STRICT_RAG_PROMPT.invoke({"context": context, "input": question})
    llm_output = llm.invoke(result.to_string())
    
    return llm_output

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
