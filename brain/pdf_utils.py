from pathlib import Path
import pickle
from typing import List
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(data_dir: str = None, use_cache: bool = True) -> List[dict]:
    """Load documents from `data_dir`.

    If `use_cache` is True and `cache/splits.pkl` exists, load the cached splits
    (precomputed chunks) and return lightweight document dicts constructed
    from those splits. This avoids parsing all PDF files on startup.
    """
    if data_dir is None:
        data_dir = str(Path(__file__).parent.parent / "data")

    documents = []
    data_path = Path(data_dir)

    cache_path = Path("cache/splits.pkl")
    if use_cache and cache_path.exists():
        try:
            with cache_path.open('rb') as f:
                splits = pickle.load(f)
            for page in splits:
                # support both langchain Document objects and dict-like
                content = getattr(page, 'page_content', None) or page.get('page_content', '')
                metadata = getattr(page, 'metadata', None) or page.get('metadata', {})
                documents.append({'content': content, 'metadata': metadata})
            return documents
        except Exception as e:
            print(f"Warning: failed to load cached splits: {e}")

    if not data_path.exists():
        print(f"Warning: Data directory not found at {data_path}")
        return documents

    for pdf_file in data_path.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()

            topic = pdf_file.stem.lower()

            for i, page in enumerate(pages):
                documents.append({
                    'content': page.page_content,
                    'metadata': {
                        'source': pdf_file.name,
                        'page': i + 1,
                        'topic': topic,
                        **page.metadata
                    }
                })
        except Exception as e:
            print(f"Error loading {pdf_file.name}: {e}")

    return documents
