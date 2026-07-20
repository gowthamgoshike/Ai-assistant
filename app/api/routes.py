import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.ingestion.loader import extract_data_from_pdf
from app.ingestion.splitter import get_chunker
from app.ingestion.vector_ops import create_vector_store
from app.rag.agent import workflow_agent

logger = logging.getLogger(__name__)

router = APIRouter()

# -------------------------------------------------------------------
# 1. REQUEST SCHEMAS
# -------------------------------------------------------------------
class QueryRequest(BaseModel):
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
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Invalid file format. Only .pdf files are accepted."
        )
        
    try:
        content = await file.read()
        pages_data, pdf_meta = extract_data_from_pdf(content)
        
        full_text = "\n".join([
            page.get("text", page.get("page_content", "")) 
            for page in pages_data
        ])
        
        chunker = get_chunker()
        chunks = chunker.split_text(full_text)
        
        create_vector_store(chunks)
        
        return {
            "status": "success", 
            "message": f"Document '{file.filename}' ingested successfully."
        }
        
    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the document: {str(e)}"
        )

# -------------------------------------------------------------------
# 4. CHAT GENERATION ENDPOINT (RAW TEXT STREAMING)
# -------------------------------------------------------------------
@router.post("/chat", tags=["Generation"])
async def chat(request: QueryRequest):
    """Streams response tokens from the LangGraph workflow as raw text in real-time."""
    
    async def token_generator():
        try:
            config = {
                "configurable": {"thread_id": request.session_id},
                "recursion_limit": 10
            }
            inputs = {"messages": [("user", request.query)]}
            
            # Stream execution events using LangGraph's astream_events v2 API
            async for event in workflow_agent.astream_events(inputs, config=config, version="v2"):
                
                # 1. Capture token stream events emitted by the LLM
                if event["event"] == "on_chat_model_stream":
                    
                    # 2. Identify which node generated the event
                    node_name = event.get("metadata", {}).get("langgraph_node")
                    
                    # 3. Stream ONLY from synthesis/direct nodes (prevents classifier leakage)
                    if node_name in ["generate", "direct_node"]:
                        chunk_text = event["data"]["chunk"].content
                        if chunk_text:
                            yield chunk_text

        except Exception as e:
            logger.error(f"Streaming error for session {request.session_id}: {str(e)}")
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(
        token_generator(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )