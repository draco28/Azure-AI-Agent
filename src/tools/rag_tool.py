import logging
import time

from langchain_core.tools import BaseTool, tool

from src.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

def create_rag_tool(retriever: HybridRetriever) -> BaseTool:
    @tool
    def rag_tool(query: str) -> str:
        """Search the document knowledge base for information. Use this when the user asks questions about company documents, policies, handbooks, or any uploaded
        content."""
        start_time = time.perf_counter()
        results = retriever.retrieve(query)
        result = retriever.format_citation(results)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "tool.rag.invoked",
            extra={
                "query": query,
                "latency_ms": latency_ms,
            },
        )
        return result
    return rag_tool

