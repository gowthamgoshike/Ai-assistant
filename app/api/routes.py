from fastapi import APIRouter
from pydantic import BaseModel
from app.rag.retriever import get_relevant_context
from app.rag.generator import generate_answer

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/chat")
async def chat(request: QueryRequest):
    context = get_relevant_context(request.query)
    answer = generate_answer(request.query, context)
    return {"answer": answer.content}