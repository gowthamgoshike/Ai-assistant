import fitz  # PyMuPDF

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Safely extracts digital text from raw PDF bytes."""
    # Tell PyMuPDF to read the raw bytes as a PDF document
    doc = fitz.open("pdf", file_bytes)
    
    text = ""
    for page in doc:
        text += page.get_text()
        
    return text