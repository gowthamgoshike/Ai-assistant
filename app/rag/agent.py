import os
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from app.rag.tools import rag_retriever_tool

load_dotenv()

# -------------------------------------------------------------------
# 1. STATE DEFINITION
# -------------------------------------------------------------------
class WorkflowState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    route: str

# -------------------------------------------------------------------
# 2. LLM & TOOLS
# -------------------------------------------------------------------
llm = ChatOllama(
    model="llama3.1", 
    temperature=0, 
    base_url="http://ollama:11434"
)
tavily_search = TavilySearch(max_results=3)

# -------------------------------------------------------------------
# 3. PROCESSING NODES
# -------------------------------------------------------------------
def classifier_node(state: WorkflowState):
    """Classifies user intent to determine the next graph branch."""
    last_user_msg = state["messages"][-1].content
    
    router_prompt = (
        "Classify the user's input into one of three categories:\n"
        "- 'rag': Questions strictly about uploaded PDF/document contents.\n"
        "- 'web': Questions requiring real-time info, news, weather, or outside knowledge.\n"
        "- 'direct': Casual greetings, personal context ('what is my name'), or small talk.\n"
        "Respond with ONLY one word: 'rag', 'web', or 'direct'."
    )
    
    response = llm.invoke([
        SystemMessage(content=router_prompt),
        HumanMessage(content=last_user_msg)
    ])
    
    decision = response.content.strip().lower()
    if decision not in ["rag", "web", "direct"]:
        decision = "direct"
        
    return {"route": decision}

def rag_retrieval_node(state: WorkflowState):
    """Retrieves document chunks from ChromaDB."""
    last_user_msg = state["messages"][-1].content
    docs = rag_retriever_tool.invoke({"query": last_user_msg})
    return {"context": str(docs)}

def web_search_node(state: WorkflowState):
    """Performs web search using Tavily."""
    last_user_msg = state["messages"][-1].content
    results = tavily_search.invoke({"query": last_user_msg})
    return {"context": str(results)}

def generate_node(state: WorkflowState):
    """Synthesizes the final answer using retrieved context."""
    context = state.get("context", "")
    messages = list(state["messages"])
    
    sys_prompt = SystemMessage(
        content=f"Answer the user's question directly using the provided context below.\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"Do not mention internal context or search tools in your final answer."
    )
    
    response = llm.invoke([sys_prompt] + messages)
    return {"messages": [response]}

def direct_chat_node(state: WorkflowState):
    """Handles greetings and memory lookup without context retrieval."""
    messages = list(state["messages"])
    sys_prompt = SystemMessage(
        content="You are a helpful assistant. Respond naturally to the user using past conversation history."
    )
    response = llm.invoke([sys_prompt] + messages)
    return {"messages": [response]}

# -------------------------------------------------------------------
# 4. CONDITIONAL ROUTER EDGE
# -------------------------------------------------------------------
def route_decision(state: WorkflowState) -> Literal["rag_node", "web_node", "direct_node"]:
    route = state.get("route", "direct")
    if route == "rag":
        return "rag_node"
    elif route == "web":
        return "web_node"
    return "direct_node"

# -------------------------------------------------------------------
# 5. GRAPH ASSEMBLY & COMPILATION
# -------------------------------------------------------------------
workflow = StateGraph(WorkflowState)

workflow.add_node("classifier", classifier_node)
workflow.add_node("rag_node", rag_retrieval_node)
workflow.add_node("web_node", web_search_node)
workflow.add_node("generate", generate_node)
workflow.add_node("direct_node", direct_chat_node)

workflow.add_edge(START, "classifier")

workflow.add_conditional_edges(
    "classifier",
    route_decision,
    {
        "rag_node": "rag_node",
        "web_node": "web_node",
        "direct_node": "direct_node"
    }
)

workflow.add_edge("rag_node", "generate")
workflow.add_edge("web_node", "generate")

workflow.add_edge("generate", END)
workflow.add_edge("direct_node", END)

memory = MemorySaver()
workflow_agent = workflow.compile(checkpointer=memory)