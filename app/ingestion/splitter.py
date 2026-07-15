from langchain_experimental.text_splitter import SemanticChunker
from app.ingestion.vector_ops import get_embedding_model

def get_chunker():
    """
    Returns a Semantic Chunker that groups sentences by meaning
    rather than strict character counts.
    """
    print("🧠 Initializing Semantic Chunker...")
    
    # We must use your existing embedding model to calculate sentence similarity
    embed_model = get_embedding_model()
    
    # Create the semantic chunker
    # breakpoint_threshold_type="percentile" means it will split the chunk 
    # whenever the semantic difference between two sentences is in the top % of differences.
    semantic_splitter = SemanticChunker(
        embed_model, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90  # Adjust this (0-100) to make chunks larger or smaller
    )
    
    return semantic_splitter