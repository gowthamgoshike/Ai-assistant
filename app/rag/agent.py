import os
from dotenv import load_dotenv

# 1. Load the hidden .env file into your environment immediately
load_dotenv()

from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from app.rag.tools import rag_retriever_tool
from langchain_core.messages import SystemMessage
# 1. Define the shared state dictionary structure
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 2. Bind the tools to the local reasoning LLM engine
llm = ChatOllama(
    model="llama3.1", 
    temperature=0, 
    base_url="http://ollama:11434"
)
tools = [
    rag_retriever_tool, 
    TavilySearch(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))
]

llm_with_tools = llm.bind_tools(tools)

# 2. Update the System Prompt to act as a Router
def call_model(state: AgentState):
    """Executes the LLM node logic to generate an output or pick a tool."""
    messages = state['messages']
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        system_prompt = SystemMessage(
            content="You are a helpful and intelligent assistant. "
                    "You have access to two tools:\n"
                    "1. 'rag_retriever_tool': Use this FIRST to answer specific questions about the user's uploaded PDF document.\n"
                    "2. 'tavily_search_results_json': Use this for general knowledge, up-to-date web searches, or if the PDF lacks the requested information.\n"
                    "Always rely on the tools to answer the question. "
                    "CRITICAL: Never mention the tool names or your internal search processes in your final answer. Respond directly and naturally."
        )
        messages = [system_prompt] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 4. Define Route Condition Edge Logic
def should_continue(state: AgentState) -> str:
    """Evaluates if the model wants to call a tool or finish."""
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 5. Build and Compile the State Graph
workflow = StateGraph(AgentState)

# Add processing workers
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# Establish layout connectivity
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent")

# Compile the execution runtime engine
agent_executor = workflow.compile()