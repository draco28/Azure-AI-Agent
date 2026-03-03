from langchain_core.tools import BaseTool, tool

from src.rag.retriever import HybridRetriever

def create_rag_tool(retriever: HybridRetriever) -> BaseTool:
    @tool
    def rag_tool(query: str) -> str:
        """Search the document knowledge base for information. Use this when the user asks questions about company documents, policies, handbooks, or any uploaded
        content."""
        results = retriever.retrieve(query)
        return retriever.format_citation(results)
    return rag_tool

