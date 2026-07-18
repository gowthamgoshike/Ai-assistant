# test_agent.py
import os
from dotenv import load_dotenv
from app.rag.agent import agent_executor

load_dotenv()

def test_graph():
    print("Testing Local RAG Route...")
    # This invokes the graph state directly
    res_local = agent_executor.invoke({
        "messages": [("user", "What is the biological family of the domestic dog")]
    })
    print("RAG Response:", res_local["messages"][-1].content)
    print("-" * 40)

    print("Testing Web Search Route...")
    res_web = agent_executor.invoke({
        "messages": [("user", "Who won the latest Formula 1 race?")]
    })
    print("Web Search Response:", res_web["messages"][-1].content)

if __name__ == "__main__":
    # Ensure you have your key loaded
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: Please set TAVILY_API_KEY in your environment or .env file.")
    else:
        test_graph()