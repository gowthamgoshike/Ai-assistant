import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store(chunks, collection_name="rag_collection"):
    embedding_model = get_embedding_model()
    
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", 8000))
    chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    
    # 🚨 NEW: Wipe the existing collection to prevent cross-contamination
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass # Triggers if the collection doesn't exist yet on a fresh boot
        
    return Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        client=chroma_client,
        collection_name=collection_name
    )