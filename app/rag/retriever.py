# app/rag/retriever.py
from langchain_chroma import Chroma
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever 
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker 
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from app.ingestion.vector_ops import get_embedding_model

PERSIST_DIR = "./data/chroma_db"

def get_relevant_context(query: str):
    """Retrieves and Re-ranks documents based on semantic relevance."""
    
    # 1. Connect to your database
    vector_db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embedding_model()
    )
    
    # 2. Stage 1: Fast Base Retrieval
    base_retriever = vector_db.as_retriever(search_kwargs={"k": 15})
    
    # 3. Stage 2: Initialize the Cross-Encoder Reranker
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=5)
    
    # 4. Combine them into a Compression Pipeline
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    # Execute the two-stage search
    compressed_docs = compression_retriever.invoke(query)
    
    # Return the clean text strings to your tool
    return [doc.page_content for doc in compressed_docs]