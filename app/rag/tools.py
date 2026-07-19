from langchain_core.tools import tool
from app.rag.retriever import get_relevant_context

@tool
def rag_retriever_tool(query: str) -> str:
    """Use this tool to search internal documents for specific information."""
    # Call the search function from retriever.py
    context = get_relevant_context(query)
    
    # Join the chunks into a single string for the LLM
    return "\n\n".join(context)