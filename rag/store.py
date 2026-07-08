import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./docker/chroma_data"):
        # Disable telemetry right at client initialization
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="rag_collection", 
            embedding_function=self.embedding_fn
        )

    def store_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return

        ids = [f"id_{i}_{chunk['metadata']['page']}" for i, chunk in enumerate(chunks)]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # Use upsert to handle updates gracefully on subsequent runs
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Successfully integrated {len(chunks)} chunks in the vector database.")