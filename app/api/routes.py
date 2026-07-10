from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from app.ingestion.loader import extract_text_from_pdf
from app.ingestion.splitter import get_chunker
from app.ingestion.vector_ops import create_vector_store
from app.rag.agent import agent_executor

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    text = extract_text_from_pdf(content)
    chunks = get_chunker(text)
    create_vector_store(chunks)
    return {"message": "Document ingested successfully."}

@router.post("/chat")
async def chat(request: QueryRequest):
    # The agent manages the internal thought process
    response = agent_executor.invoke({"messages": [("user", request.query)]})
    # Extract the final answer from the agent's last message
    final_answer = response["messages"][-1].content
    return {"answer": final_answer}