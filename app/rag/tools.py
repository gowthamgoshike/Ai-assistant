# app/rag/tools.py
from langchain.tools import tool
from app.rag.retriever import get_relevant_context

@tool
def rag_retriever_tool(query: str):
    """Use this tool to search internal documents for specific information."""
    context = get_relevant_context(query)
    return "\n".join(context)