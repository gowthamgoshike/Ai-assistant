from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from app.rag.tools import rag_retriever_tool

llm = ChatOllama(model="llama3.1", temperature=0)

# The agent now has access to your RAG tool AND the live Web Search tool
tools = [rag_retriever_tool, TavilySearchResults()]

agent_executor = create_react_agent(llm, tools)