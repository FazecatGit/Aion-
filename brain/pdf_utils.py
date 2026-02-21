from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs(data_dir: str = None) -> list[dict]:
    if data_dir is None:
        data_dir = str(Path(__file__).parent.parent / "data")
    
    documents = []
    data_path = Path(data_dir)
    
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
