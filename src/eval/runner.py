import json
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.agent.setup import setup_agent

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.dataset_schema import SingleTurnSample
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper

def check_facts(answer: str, expected_facts: list[str]) -> tuple[list[str], list[str]]:
    """Check which expected facts appear in the answer (case-insensitive)."""
    facts_found = []
    facts_missing = []
    for fact in expected_facts:
        if fact.lower() in answer.lower():
            facts_found.append(fact)
        else:
            facts_missing.append(fact)
    return (facts_found, facts_missing)

def check_tool_selection(tools_used: list[str], expected_tool: str) -> bool:
    """ Check if the expected tool was used """
    if expected_tool == "none":
        return len(tools_used) == 0
    elif expected_tool == "both":
        return "rag_tool" in tools_used and "file_status_tool" in tools_used
    else:
        return expected_tool in tools_used

    
def check_sources(tool_results: str, expected_sources: list[str]) -> bool:
    """ Check if the expected sources were used """
    if expected_sources == []:
        return True
    return all(source in tool_results for source in expected_sources)

async def run_single_eval(graph, test_case: dict[str, Any]) -> dict[str, Any]:
    """ Run a single evaluation on a test case """
    response = await graph.ainvoke({"messages": [HumanMessage(content=test_case["query"])]})
    tools_used = []
    tool_results = ""
    answer = ""
    retrieved_contexts = []
    for msg in response["messages"]:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tools_used.append(tool_call["name"])
        if isinstance(msg, ToolMessage):
            tool_results += msg.content
            retrieved_contexts.append(msg.content)
    answer = response["messages"][-1].content
    facts_found, facts_missing = check_facts(answer, test_case["expected_answer_contains"])
    tool_correct = check_tool_selection(tools_used, test_case["expected_tool"])
    source_correct = check_sources(tool_results, test_case["expected_sources"])
    return {
        "id":test_case["id"],
        "query": test_case["query"],
        "category": test_case["category"],
        "tools_used": tools_used,
        "expected_tool": test_case["expected_tool"],
        "tool_correct": tool_correct,
        "answer": answer,
        "facts_found": facts_found,
        "facts_missing": facts_missing,
        "facts_correct": len(facts_missing) == 0,
        "source_correct": source_correct,
        "retrieved_contexts": retrieved_contexts,
        "reference": test_case.get("reference", "")
    }

def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """ Compute metrics for a list of evaluation results """
    total = len(results)
    tool_correct_count = 0
    facts_correct_count = 0
    source_correct_count = 0
    scoreable_count = 0

    for result in results:
        if result["tool_correct"]:
            tool_correct_count += 1
        
        if result["category"] not in ["negative", "no_tool"]:
            scoreable_count += 1
            if result["facts_correct"]:
                facts_correct_count += 1
            if result["source_correct"]:
                source_correct_count += 1
    return {
        "tool_accuracy": tool_correct_count / total,
        "faithfulness": facts_correct_count / scoreable_count,
        "source_accuracy": source_correct_count / scoreable_count
    }

def load_test_dataset(path: str = "data/eval/test_dataset.json") -> list[dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)

async def run_evaluation():
    # 1. Load test dataset
    print("Loading test dataset...")
    test_dataset = load_test_dataset()
    # 2. Setup agent (await setup_agent())
    print("Setting up agent...")
    agent = await setup_agent()
    # 3. Loop through test cases, call run_single_eval for each
    print("Running evaluation...")
    results = []
    for test_case in test_dataset:
        print(f"Running test case {test_case['id']}...")
        result = await run_single_eval(agent, test_case)
        results.append(result)
    
    # 4. Convert results to ragas samples
    samples = []
    for result in results:
        if result["category"] not in ["negative", "no_tool"]:
            samples.append(SingleTurnSample(
                user_input=result["query"],
                response=result["answer"],
                retrieved_contexts=result["retrieved_contexts"],
                reference=result["reference"],
            ))
    dataset = EvaluationDataset(samples=samples)
    settings = get_settings()
    llm_as_judge = LangchainLLMWrapper(ChatOpenAI(
        base_url=settings.glm_base_url,
        model=settings.chat_model,
        api_key=settings.glm_api_key.get_secret_value()
    ))
    ragas_results = evaluate(
        dataset=dataset,
        llm=llm_as_judge,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )
    # 5. Call compute_metrics on all results
    print("Computing metrics...")
    metrics = compute_metrics(results)
    for result in results:
        missing = f" (missing: {result['facts_missing']})" if result['facts_missing'] else ""
        print(f"{result['id']} | Category: {result['category']} | Tool: {'✓' if result['tool_correct'] else '✗'} | Facts: {'✓' if result['facts_correct'] else '✗'} {missing} | Source: {'✓' if result['source_correct'] else '✗'}")
    print("\nRAGAS Results:")
    print(ragas_results)
    print("\nMetrics:")
    print(f"Tool Accuracy: {metrics['tool_accuracy']:.2f}")
    print(f"Faithfulness: {metrics['faithfulness']:.2f}")
    print(f"Source Accuracy: {metrics['source_accuracy']:.2f}")
    
    return results, metrics

if __name__ == "__main__":
    import asyncio
    results, metrics = asyncio.run(run_evaluation())