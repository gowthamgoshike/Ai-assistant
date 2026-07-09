from app.ingestion.vector_ops import Chroma, get_embedding_model

def get_relevant_context(query: str, k=3, persist_directory="./data/chroma_db"):
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=get_embedding_model())
    results = vector_db.similarity_search(query, k=k)
    return [doc.page_content for doc in results]