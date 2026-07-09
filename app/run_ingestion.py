# run_ingestion.py
from pathlib import Path
from app.ingestion.loader import extract_text_from_pdf
from app.ingestion.splitter import get_chunker
from app.ingestion.vector_ops import create_vector_store

# Set your PDF path here
pdf_path = Path("/Users/gowthamgoshike/projects/Ai-assistant/data/Gowtham_Goshike_Data_Scientist_Resume.pdf")
text = extract_text_from_pdf(pdf_path)
chunks = get_chunker().split_text(text)
create_vector_store(chunks)
print("Ingestion complete. ChromaDB created.")