import os
import glob
from datetime import datetime
from app.ingestion.loader import extract_data_from_pdf
from app.ingestion.splitter import get_chunker
from app.ingestion.vector_ops import get_embedding_model
from langchain_chroma import Chroma
from langchain_core.documents import Document

DATA_DIR = "./data"
PERSIST_DIR = "./data/chroma_db"

def ingest_local_directory():
    print("Checking for new documents to vectorize...")
    
    vector_db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embedding_model()
    )
    
    # Existing file tracker to prevent duplicates
    existing_metadata = vector_db.get()
    processed_sources = set()
    if existing_metadata and 'metadatas' in existing_metadata:
        for meta in existing_metadata['metadatas']:
            if meta and 'source' in meta:
                processed_sources.add(meta['source'])

    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    new_docs = []
    splitter = get_chunker()
    
    for file_path in pdf_files:
        file_name = os.path.basename(file_path)
        
        if file_name in processed_sources:
            print(f"-> Skipping: '{file_name}' (Already indexed).")
            continue
            
        print(f"-> Processing new file: '{file_name}'")
        try:
            # 1. Grab OS-level Timestamps
            file_mod_time = os.path.getmtime(file_path)
            timestamp_str = datetime.fromtimestamp(file_mod_time).strftime('%Y-%m-%d %H:%M:%S')
            ingestion_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            with open(file_path, "rb") as f:
                file_bytes = f.read()
                
            # 2. Extract page data and PDF internal metadata
            pages_data, pdf_meta = extract_data_from_pdf(file_bytes)
            
            # Extract PDF title if available, otherwise fallback to filename
            doc_title = pdf_meta.get("title", "") if pdf_meta.get("title") else file_name
            
            # 3. Chunk the text PAGE by PAGE to preserve section accuracy
            for page in pages_data:
                page_text = page["text"]
                page_num = page["page_number"]
                
                if not page_text.strip():
                    continue  # Skip empty pages
                    
                chunks = splitter.split_text(page_text)
                
                # 4. Bind the extended metadata dictionary to the chunk
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file_name,
                            "title": doc_title,
                            "page_number": page_num,           # The specific section/page
                            "chunk_index": i,
                            "total_page_chunks": len(chunks),
                            "content_length": len(chunk),
                            "file_last_modified": timestamp_str, # When the file was last edited
                            "ingested_at": ingestion_time        # When the DB processed it
                        }
                    )
                    new_docs.append(doc)
                
        except Exception as e:
            print(f"Failed to process {file_name}: {str(e)}")

    # Incremental update: append using add_documents
    if new_docs:
        vector_db.add_documents(documents=new_docs)
        print(f"Successfully appended {len(new_docs)} extended metadata chunks to ChromaDB.")
    else:
        print("No new unique documents found. Vector store is up to date.")

if __name__ == "__main__":
    ingest_local_directory()