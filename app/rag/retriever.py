import os
import chromadb
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever 
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker 
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from app.ingestion.vector_ops import get_embedding_model

def get_relevant_context(query: str):
    """Executes Hybrid Search (Semantic + BM25) and Re-ranks the fused results."""
    
    # 1. Connect to the standalone ChromaDB container over the network
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", 8000))
    chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    
    # Target the identical collection name used in vector_ops.py
    vector_db = Chroma(
        client=chroma_client,
        collection_name="rag_collection",
        embedding_function=get_embedding_model()
    )
    
    # Semantic Search: Pull top 10 conceptual matches
    dense_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    
    # 2. Keyword Search: Initialize BM25 Sparse Retriever
    # Extract the raw text chunks directly from the remote Chroma collection
    existing_data = vector_db.get()
    all_chunks = existing_data.get('documents', []) if existing_data else []
    
    # Safeguard: Prevent BM25 from crashing if the database is empty
    if not all_chunks:
        print("⚠️ Warning: ChromaDB is empty. Returning no results.")
        return []
        
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
    
    adjusted_docs = []
    
    # STEP 1: Apply Custom Rules to the Scores
    for doc in compressed_docs:
        # Extract base metadata
        base_score = doc.metadata.get('relevance_score', 0.0)
        source = str(doc.metadata.get('source', 'Unknown')).lower()
        page = doc.metadata.get('page_number', 'N/A')
        
        final_score = base_score
        
        # 🧠 RULE 1: Boost priority for specific high-value files
        if 'resume' in source or 'cv' in source:
            final_score += 3.0
            print(f"📈 BOOSTED (+3.0) | Source: {source} | New Score: {final_score:.4f}")
            
        # 🧠 RULE 2: Penalize outdated or secondary files
        elif 'archive' in source or 'old' in source:
            final_score -= 2.0
            print(f"📉 PENALIZED (-2.0) | Source: {source} | New Score: {final_score:.4f}")
            
        # Store the adjusted data as a dictionary for easy sorting
        adjusted_docs.append({
            'content': doc.page_content,
            'score': final_score,
            'source': source,
            'page': page
        })
        
    # STEP 2: Re-sort the list so the highest scores are at the top (Index 0)
    adjusted_docs.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n{'-'*50}")
    print("🛡️ FINAL RANKING & SELECTION")
    print(f"{'-'*50}")
    
    final_strings = []
    
    # STEP 3: Allow the top reranked chunks directly into the context window
    for item in adjusted_docs:
        print(f"✅ ACCEPTED | Score: {item['score']:.4f} | Page: {item['page']} | Source: {item['source']}")
        final_strings.append(item['content'])
            
    print(f"{'-'*50}")
    print(f"Kept {len(final_strings)} highly relevant chunks for the LLM.\n")
    
    return final_strings