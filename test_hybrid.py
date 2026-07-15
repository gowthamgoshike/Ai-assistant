# test_hybrid.py
import os
from dotenv import load_dotenv
from app.rag.retriever import get_relevant_context

# Load environment variables just in case
load_dotenv()

def run_test():
    # A query designed to check if both keywords and semantic meanings are being fetched
    query = "What specific technologies are used in the Real-Time Sales Analytics Platform?"
    
    print(f"Testing Query: '{query}'\n")
    print("Executing Hybrid Search (BM25 + Semantic) -> Cross-Encoder Reranking...")
    
    # This triggers your newly updated two-stage retrieval pipeline
    results = get_relevant_context(query)
    
    print(f"\n✅ Successfully retrieved and ranked the top {len(results)} chunks:")
    for i, chunk in enumerate(results):
        print(f"\n--- Rank {i+1} ---")
        # Prints the first 300 characters of each chunk along with the metadata header
        print(chunk[:300].strip() + "...\n")

if __name__ == "__main__":
    run_test()