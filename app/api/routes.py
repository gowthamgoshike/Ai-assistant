from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field
import logging

from app.ingestion.loader import extract_data_from_pdf
from app.ingestion.splitter import get_chunker
from app.ingestion.vector_ops import create_vector_store
from app.rag.agent import agent_executor

# Setup logging for production observability
logger = logging.getLogger(__name__)

router = APIRouter()

# -------------------------------------------------------------------
# 1. REQUEST VALIDATION
# -------------------------------------------------------------------
class QueryRequest(BaseModel):
    # Enforce that the query is at least 3 characters long
    query: str = Field(..., min_length=3, description="The user's question for the AI")
    session_id: str = "test_user_1"
# -------------------------------------------------------------------
# 2. HEALTH CHECK ENDPOINT
# -------------------------------------------------------------------
@router.get("/health", tags=["System"])
async def health_check():
    """Simple endpoint to verify the API is running and responsive."""
    return {"status": "healthy", "service": "RAG API Router is operational"}

# -------------------------------------------------------------------
# 3. DOCUMENT UPLOAD ENDPOINT
# -------------------------------------------------------------------
@router.post("/upload", tags=["Data Management"])
async def upload_document(file: UploadFile = File(...)):
    """Extracts, chunks, and stores a PDF document in the vector store."""
    
    # Validation: Reject non-PDF files immediately
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Invalid file format. Only .pdf files are accepted."
        )
        
    try:
        # Read the file directly into memory
        content = await file.read()
        
        # Pass 'content' instead of the undefined 'file_bytes'
        pages_data, pdf_meta = extract_data_from_pdf(content)
        
        # 1. Combine the text from all pages into a single string
        # (Using safe .get() defaults in case the key is 'text' or 'page_content')
        full_text = "\n".join([
            page.get("text", page.get("page_content", "")) 
            for page in pages_data
        ])
        
        # 2. Get the chunker instance
        chunker = get_chunker()
        
        # 3. ✅ FIXED: Use split_text to output a list of plain strings
        chunks = chunker.split_text(full_text)
        
        # Ingest the string chunks into ChromaDB
        create_vector_store(chunks)
        
        return {
            "status": "success", 
            "message": f"Document '{file.filename}' ingested successfully."
        }
        
    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {str(e)}")
        # Return a clean 500 error instead of crashing the server
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the document: {str(e)}"
        )

## -------------------------------------------------------------------
# 4. CHAT GENERATION ENDPOINT
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# 4. CHAT GENERATION ENDPOINT
# -------------------------------------------------------------------
@router.post("/chat", tags=["Generation"])
def chat(request: QueryRequest):
    """Passes the validated query to the LangGraph agent for reasoning and retrieval."""
    
    try:
        # Merge both configs: thread_id tells MemorySaver which history to load,
        # and recursion_limit prevents infinite tool-calling loops.
        config = {
            "configurable": {"thread_id": request.session_id},
            "recursion_limit": 10
        }
        
        response = agent_executor.invoke(
            {"messages": [("user", request.query)]},
            config=config
        )
        
        # Extract the final answer from the agent's last message
        final_answer = response["messages"][-1].content
        
        return {
            "status": "success",
            "answer": final_answer
        }
        
    except Exception as e:
        logger.error(f"Agent execution failed for query '{request.query}': {str(e)}")
        # If it hits the 10-loop limit, it raises a GraphRecursionError, which is caught here.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"The RAG engine encountered an unexpected error: {str(e)}"
        )