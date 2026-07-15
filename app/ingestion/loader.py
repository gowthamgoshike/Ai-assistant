import fitz  # PyMuPDF

def extract_data_from_pdf(file_bytes: bytes):
    """Extracts text page-by-page and retrieves document metadata."""
    doc = fitz.open("pdf", file_bytes)
    
    # Get internal PDF metadata (Author, CreationDate, Title, etc.)
    doc_metadata = doc.metadata or {}
    
    pages_data = []
    for page_num, page in enumerate(doc):
        pages_data.append({
            "page_number": page_num + 1,  # Acts as our section tracker
            "text": page.get_text()
        })
        
    return pages_data, doc_metadata