from rag.loader import PDFLoader
from rag.splitter import ChunkingSystem
from rag.store import VectorStoreManager

def run_ingestion_pipeline(pdf_path: str):
    print("🚀 Initializing Ingestion Pipeline...")
    
    # 1. Load
    loader = PDFLoader(file_path=pdf_path)
    raw_docs = loader.load()
    
    # 2. Chunk
    splitter = ChunkingSystem(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)
    
    # 3. Store
    store = VectorStoreManager(persist_directory="./docker/chroma_data")
    store.store_chunks(chunks)
    
    print("✅ Pipeline execution complete.")

if __name__ == "__main__":
    # Example PDF target
    run_ingestion_pipeline("/Users/gowthamgoshike/projects/Ai-assistant/files/Gowtham_Goshike_Data_Scientist_Resume.pdf")