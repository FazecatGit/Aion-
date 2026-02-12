"""Keyword search functionality for PDFs using text processing and stemming."""
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as nltk_stopwords

# Auto-download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_text(text: str) -> list[str]:
    text = clean_text(text)
    stopwords = set(nltk_stopwords.words('english'))
    res = []

    def _filter(tok):
        tok = tok.strip()
        if tok and tok not in stopwords:
            return True
        return False

    for tok in text.split():
        if _filter(tok):
            tok = stemmer.stem(tok)
            res.append(tok)     
    return res

def has_matching_token(query_tokens: list[str], doc_tokens: list[str]) -> bool:
    for q in query_tokens:
        for d in doc_tokens:
            if q in d:
                return True
    return False

def search_documents(query: str, documents: list[dict], n_results: int = 5) -> list[dict]:
    results = []
    query_tokens = tokenize_text(query)
    
    for doc in documents:
        doc_content = doc.get('content', '')
        doc_tokens = tokenize_text(doc_content)
        
        if has_matching_token(query_tokens, doc_tokens):
            results.append(doc)
            if len(results) == n_results:
                break
    
    return results
