# app/rag/retriever.py
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever 
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker 
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from app.ingestion.vector_ops import get_embedding_model

PERSIST_DIR = "./data/chroma_db"

def get_relevant_context(query: str):
    """Executes Hybrid Search (Semantic + BM25) and Re-ranks the fused results."""
    
    # 1. Connect to your dense vector database
    vector_db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embedding_model()
    )
    
    # Semantic Search: Pull top 10 conceptual matches
    dense_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    
    # 2. Keyword Search: Initialize BM25 Sparse Retriever
    # Extract the raw text chunks directly from the persistent Chroma collection
    existing_data = vector_db.get()
    all_chunks = existing_data['documents'] if existing_data else []
    
    bm25_retriever = BM25Retriever.from_texts(all_chunks)
    bm25_retriever.k = 10
    
    # 3. Hybrid Search: Fuse Semantic and Keyword results
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5] # 50/50 balance between exact keywords and semantics
    )
    
    # 4. Stage 2: Initialize the Cross-Encoder Reranker
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=5)
    
    # 5. Combine the Hybrid Retrieval + Reranking into one master pipeline
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )
    
    # Execute the master search
    print(f"\n🔍 Executing Hybrid Search & Reranking for: '{query}'")
    compressed_docs = compression_retriever.invoke(query)
    
    print(f"\n{'-'*50}")
    print("📊 CROSS-ENCODER RELEVANCE SCORES")
    print(f"{'-'*50}")
    
    final_strings = []
    
    for doc in compressed_docs:
        # The CrossEncoderReranker automatically injects the 'relevance_score' into metadata
        score = doc.metadata.get('relevance_score', 0.0)
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page_number', 'N/A')
        
        # 🛡️ STRICT CUTOFF RULE: Only keep chunks with a positive score
        if score > 0:
            print(f"✅ ACCEPTED | Score: {score:.4f} | Page: {page} | Source: {source}")
            final_strings.append(doc.page_content)
        else:
            print(f"❌ REJECTED | Score: {score:.4f} | Page: {page} | Source: {source} (Irrelevant)")
            
    print(f"{'-'*50}")
    print(f"Kept {len(final_strings)} highly relevant chunks for the LLM.\n")
    
    # Return only the vetted text strings to your LangGraph Agent
    return final_strings