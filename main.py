# main.py
from fastapi import FastAPI
from app.api.routes import router

from dotenv import load_dotenv
load_dotenv() # This loads the variables from .env into your environment
app = FastAPI(title="RAG AI Assistant")

# Include your routes
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)