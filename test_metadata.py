from app.ingestion.vector_ops import get_embedding_model
from langchain_chroma import Chroma

PERSIST_DIR = "./data/chroma_db"

def test_metadata_retrieval():
    print("🔍 Connecting to ChromaDB...")
    
    # Connect to your existing database
    vector_db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embedding_model()
    )
    
    # Run a test query (this matches content we know is in your books.pdf)
    query = "What are the important Python libraries used for Machine Learning?"
    print(f"\n🗣️ Querying: '{query}'")
    
    # Retrieve the top 2 most relevant chunks
    results = vector_db.similarity_search(query, k=2)
    
    if not results:
        print("❌ No results found. Did you run the ingestion script?")
        return

    print("\n✅ Results Retrieved! Check out your new metadata:\n")
    
    for i, doc in enumerate(results):
        print(f"{'='*40}")
        print(f"🏆 RANK {i+1}")
        print(f"{'='*40}")
        
        # Format and print the metadata nicely
        meta = doc.metadata
        print(f"📄 Source File  : {meta.get('source')}")
        print(f"🔖 Title        : {meta.get('title')}")
        print(f"📖 Page Number  : {meta.get('page_number')}")
        print(f"🧩 Chunk Index  : {meta.get('chunk_index')} of {meta.get('total_page_chunks')}")
        print(f"🕒 Last Edited  : {meta.get('file_last_modified')}")
        print(f"📥 Ingested At  : {meta.get('ingested_at')}")
        print(f"{'-'*40}")
        # Print a snippet of the actual text
        print(f"📝 Content Snippet:\n{doc.page_content[:250]}...\n")

if __name__ == "__main__":
    test_metadata_retrieval()