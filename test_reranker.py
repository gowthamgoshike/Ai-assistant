from app.rag.retriever import get_relevant_context

sample_query = "What technologies did Gowtham use to engineer the Real-Time Sales Analytics Platform?"

print(f"Testing Query: '{sample_query}'\n")
print("Executing two-stage retrieval (ChromaDB -> Cross-Encoder)...")

# This triggers both Stage 1 (ChromaDB) and Stage 2 (HuggingFace Reranker)
results = get_relevant_context(sample_query)

print(f"\n✅ Successfully retrieved and ranked the top {len(results)} chunks:")
for i, chunk in enumerate(results):
    print(f"\n--- Rank {i+1} ---")
    # Print the first 300 characters of each chunk to verify semantic relevance
    print(chunk[:300].strip() + "...\n")