import fitz  # PyMuPDF
from pathlib import Path

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts raw text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text() for page in doc])
    doc.close()
    return text