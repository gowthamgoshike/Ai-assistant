import asyncio
import json
import os
import sys
import pandas as pd
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.rag.agent import workflow_agent


async def run_pipeline(query: str, session_id: str) -> Dict[str, Any]:
    """Invokes the LangGraph workflow and extracts state outputs."""
    config = {"configurable": {"thread_id": session_id}}
    inputs = {"messages": [("user", query)]}

    # Execute full graph state
    final_state = await workflow_agent.ainvoke(inputs, config=config)

    # Extract final assistant message
    messages = final_state.get("messages", [])
    answer = messages[-1].content if messages else ""

    # Extract retrieved context documents if present in graph state
    raw_contexts = final_state.get("context", [])
    contexts = [
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in raw_contexts
    ]

    # Extract routing decisions from metadata or state
    executed_route = final_state.get("route", "unknown")

    return {
        "answer": answer,
        "contexts": contexts,
        "executed_route": executed_route,
    }


async def evaluate_dataset(dataset_path: str):
    """Loads benchmark dataset, runs agent, and calculates evaluation scores."""
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        test_cases = json.load(f)

    results = []
    print(f"\n--- Running Evaluation on {len(test_cases)} Test Cases ---\n")

    for idx, item in enumerate(test_cases, 1):
        query = item["query"]
        ground_truth = item["ground_truth"]
        expected_route = item.get("expected_route", "rag")

        print(f"[{idx}/{len(test_cases)}] Evaluating: '{query}'")

        # Run pipeline
        out = await run_pipeline(query, session_id=f"eval_session_{idx}")

        # Basic Route Check
        route_correct = out["executed_route"] == expected_route

        results.append(
            {
                "query": query,
                "ground_truth": ground_truth,
                "generated_answer": out["answer"],
                "contexts": out["contexts"],
                "expected_route": expected_route,
                "executed_route": out["executed_route"],
                "route_accuracy": 1.0 if route_correct else 0.0,
            }
        )

    df = pd.DataFrame(results)

    # Summary Report
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Test Cases : {len(df)}")
    print(f"Route Accuracy   : {df['route_accuracy'].mean() * 100:.1f}%")
    print("====================================================\n")

    # Export results to CSV
    output_csv = "eval/eval_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"Detailed evaluation results saved to: {output_csv}")


if __name__ == "__main__":
    asyncio.run(evaluate_dataset("eval/test_dataset.json"))