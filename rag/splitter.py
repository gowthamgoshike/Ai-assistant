from typing import List, Dict, Any

class ChunkingSystem:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Splits documents into smaller overlapping chunks."""
        chunks = []
        
        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]
            
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                
                # Create unique chunk metadata
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_range"] = f"{start}-{end}"
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
                
                # Move window forward by chunk_size minus overlap
                start += (self.chunk_size - self.chunk_overlap)
                
        return chunks
    