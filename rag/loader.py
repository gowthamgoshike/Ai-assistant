import fitz # PyMuPDF
from typing import Dict, Any, List

class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Dict[str, Any]]:
        """Extracts text page by page with basic metadata."""
        documents = []
        try:
            doc = fitz.open(self.file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():  # Skip empty pages
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": self.file_path,
                            "page": page_num + 1
                        }
                    })
            return documents
        except Exception as e:
            print(f"Error loading PDF {self.file_path}: {e}")
            return []