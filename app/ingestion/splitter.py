from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_chunker(chunk_size=1000, chunk_overlap=200):
    """Splits text while respecting paragraph boundaries."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )