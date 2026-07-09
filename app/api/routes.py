from fastapi import APIRouter, UploadFile, File
from app.ingestion.loader import extract_text_from_pdf
from app.ingestion.splitter import get_chunker
from app.ingestion.vector_ops import create_vector_store
from app.rag.retriever import get_relevant_context
from app.rag.generator import generate_answer

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. Extract and process
    content = await file.read()
    text = extract_text_from_pdf(content)
    
    # 2. Chunk and Ingest
    chunks = get_chunker(text)
    create_vector_store(chunks)
    
    return {"message": "Document ingested and stored in ChromaDB."}

@router.post("/query")
async def query_assistant(query: str):
    # 1. Retrieve
    context = get_relevant_context(query)
    
    # 2. Generate
    response = generate_answer(query, context)
    
    return {"answer": response}