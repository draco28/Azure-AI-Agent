import json
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.config import get_settings
from src.agent.setup import setup_agent

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

async def run_ragas_evaluation(results: list[dict[str, Any]]) -> dict[str, float] | None:
    """Run RAGAS LLM-judged evaluation. Returns None if RAGAS is unavailable or fails."""
    try:
        from openai import AsyncOpenAI, AsyncAzureOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall
    except ImportError:
        print("RAGAS not installed, skipping LLM-judged evaluation.")
        return None

    try:
        settings = get_settings()

        llm_client = AsyncOpenAI(
            base_url=settings.glm_base_url,
            api_key=settings.glm_api_key.get_secret_value(),
        )
        ragas_llm = llm_factory(settings.chat_model, provider="openai", client=llm_client, max_tokens=8192)

        embedding_client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key.get_secret_value(),
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
        ragas_embeddings = RagasOpenAIEmbeddings(
            client=embedding_client,
            model=settings.embedding_deployment,
        )

        faith_metric = Faithfulness(llm=ragas_llm)
        relevancy_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
        recall_metric = ContextRecall(llm=ragas_llm)

        faith_samples = []
        relevancy_samples = []
        recall_samples = []
        for result in results:
            if result["category"] not in ["negative", "no_tool"]:
                faith_samples.append({
                    "user_input": result["query"],
                    "response": result["answer"],
                    "retrieved_contexts": result["retrieved_contexts"],
                })
                relevancy_samples.append({
                    "user_input": result["query"],
                    "response": result["answer"],
                })
                recall_samples.append({
                    "user_input": result["query"],
                    "retrieved_contexts": result["retrieved_contexts"],
                    "reference": result["reference"],
                })

        faith_results = []
        relevancy_results = []
        recall_results = []
        for i in range(len(faith_samples)):
            print(f"  RAGAS scoring sample {i+1}/{len(faith_samples)}...")
            faith_results.append(await faith_metric.ascore(**faith_samples[i]))
            relevancy_results.append(await relevancy_metric.ascore(**relevancy_samples[i]))
            recall_results.append(await recall_metric.ascore(**recall_samples[i]))

        return {
            "ragas_faithfulness": sum(r.value for r in faith_results) / len(faith_results),
            "ragas_answer_relevancy": sum(r.value for r in relevancy_results) / len(relevancy_results),
            "ragas_context_recall": sum(r.value for r in recall_results) / len(recall_results),
        }
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        return None

async def run_evaluation():
    # 1. Load test dataset
    print("Loading test dataset...")
    test_dataset = load_test_dataset()
    # 2. Setup agent (await setup_agent())
    print("Setting up agent...")
    agent, _ = await setup_agent()
    # 3. Loop through test cases, call run_single_eval for each
    print("Running evaluation...")
    results = []
    for test_case in test_dataset:
        print(f"Running test case {test_case['id']}...")
        result = await run_single_eval(agent, test_case)
        results.append(result)

    # 4. Compute manual metrics
    print("Computing metrics...")
    metrics = compute_metrics(results)
    for result in results:
        missing = f" (missing: {result['facts_missing']})" if result['facts_missing'] else ""
        print(f"{result['id']} | Category: {result['category']} | Tool: {'✓' if result['tool_correct'] else '✗'} | Facts: {'✓' if result['facts_correct'] else '✗'}{missing} | Source: {'✓' if result['source_correct'] else '✗'}")

    print("\nManual Metrics:")
    print(f"  Tool Accuracy:   {metrics['tool_accuracy']:.2f}")
    print(f"  Faithfulness:    {metrics['faithfulness']:.2f}")
    print(f"  Source Accuracy:  {metrics['source_accuracy']:.2f}")

    # 5. Run RAGAS evaluation (optional)
    print("\nRunning RAGAS evaluation...")
    ragas_metrics = await run_ragas_evaluation(results)
    if ragas_metrics:
        metrics.update(ragas_metrics)
        print("\nRAGAS Metrics (LLM-judged):")
        print(f"  Faithfulness:      {ragas_metrics['ragas_faithfulness']:.2f}")
        print(f"  Answer Relevancy:  {ragas_metrics['ragas_answer_relevancy']:.2f}")
        print(f"  Context Recall:    {ragas_metrics['ragas_context_recall']:.2f}")
    else:
        print("RAGAS evaluation skipped.")

    return results, metrics

if __name__ == "__main__":
    import asyncio
    results, metrics = asyncio.run(run_evaluation())
