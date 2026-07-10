import os
import glob
from app.ingestion.loader import extract_text_from_pdf
from app.ingestion.splitter import get_chunker
from app.ingestion.vector_ops import get_embedding_model
from langchain_chroma import Chroma

DATA_DIR = "./data"
PERSIST_DIR = "./data/chroma_db"

def ingest_local_directory():
    print("Checking for new documents to vectorize...")
    
    # Connect directly to your existing persistent store
    vector_db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embedding_model()
    )
    
    # Extract structural metadata if records exist
    existing_metadata = vector_db.get()
    processed_sources = set()
    if existing_metadata and 'metadatas' in existing_metadata:
        for meta in existing_metadata['metadatas']:
            if meta and 'source' in meta:
                processed_sources.add(meta['source'])

    # Discover target local PDFs
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    
    new_chunks = []
    new_metadatas = []
    
    for file_path in pdf_files:
        file_name = os.path.basename(file_path)
        
        # Incremental guard: skip if file structural footprint exists
        if file_name in processed_sources:
            print(f"-> Skipping: '{file_name}' (Already indexed).")
            continue
            
        print(f"-> Processing new file: '{file_name}'")
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
                
            text = extract_text_from_pdf(file_bytes)
            chunks = get_chunker(text)
            
            for chunk in chunks:
                new_chunks.append(chunk)
                new_metadatas.append({"source": file_name})
        except Exception as e:
            print(f"Failed to process {file_name}: {str(e)}")

    # Incremental update: append only new records
    if new_chunks:
        vector_db.add_texts(texts=new_chunks, metadatas=new_metadatas)
        print(f"Successfully appended {len(new_chunks)} new chunks to ChromaDB.")
    else:
        print("No new unique documents found. Vector store is up to date.")

if __name__ == "__main__":
    ingest_local_directory()