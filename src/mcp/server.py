from src.config import get_settings, Settings

from src.rag.embeddings import create_embeddings
from src.rag.store import load_index
from src.rag.sparse import BM25Retriever
from src.rag.retriever import HybridRetriever, AzureSearchRetriever

from src.db.connection import init_db
from src.db.seed import seed_database

from src.tools.file_status import query_files

from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP, Context

@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Lifespan context manager for the MCP server."""
    # Startup

    settings: Settings = get_settings()
    
    embedding = create_embeddings(settings)

    retriever = None
    if settings.search_backend == "azure":
        retriever = AzureSearchRetriever(embedding,settings)
    elif settings.search_backend == "faiss":
        try:
           faiss_index = load_index(settings.vector_store_path, embedding)
        except FileNotFoundError:
            # Index doesn't exist yet - we'll create it later when needed
            faiss_index = None

        if faiss_index is not None:
            docs = list(faiss_index.docstore._dict.values())
            bm25 = BM25Retriever(docs)
            retriever = HybridRetriever(faiss_index, bm25, embedding, settings)
    
    engine, session_factory = await init_db()
    await seed_database(session_factory)

    yield {
        "engine": engine,
        "session_factory": session_factory,
        "retriever": retriever
    }

mcp = FastMCP("Azure AI Agent", lifespan=app_lifespan, host="0.0.0.0", port=8001)


@mcp.tool()
async def rag_retrieve(query: str, ctx: Context) -> str:
    """Retrieve relevant documents from the knowledge base using hybrid search (vector + keyword matching).
    
    This tool searches through indexed documents to find information relevant to the user's query.
    Use this when you need to find specific information from the knowledge base or answer questions
    that require context from stored documents.
    
    Args:
        query: The search query or question to find relevant documents for
        
    Returns:
        Formatted citations of relevant documents, or a message if no documents are indexed yet
        
    Example usage:
        - "What are the main features of the product?"
        - "Find information about pricing policies"
        - "Search for troubleshooting steps for error XYZ"
        :param query:
        :param ctx:
    """
    retriever = ctx.request_context.lifespan_context["retriever"]
    if retriever is None:
        return "No documents have been indexed yet."
    results = retriever.retrieve(query)
    return retriever.format_citation(results)


@mcp.tool()
async def file_status_query(ctx: Context, filename: str = None, file_id: str = None, status: str = None) -> str:
    """Query and filter files in the system based on their properties.
    
    This tool allows you to search for files and check their processing status in the document management system.
    Use this when you need to:
    - Find files by their exact filename
    - Look up a specific file using its unique identifier (file_id)
    - List all files that have a certain processing status
    - Check if a file has been successfully processed or encountered errors
    
    Args:
        filename: (Optional) The exact name of the file to search for (e.g., "report.pdf", "data.csv")
        file_id: (Optional) The unique identifier of a specific file in the system
        status: (Optional) Filter files by their processing status. Common values include:
                - "pending": Files waiting to be processed
                - "processing": Files currently being processed
                - "approved": Successfully processed files
                - "rejected": Files that encountered errors during processing
    
    Returns:
        A list of files matching the specified criteria with their details (name, ID, status, timestamps)
        
    Example usage:
        - "What is the status of report.pdf?"
        - "Show me all files that rejected processing"
        - "Find the file with ID abc-123-def"
        - "List all approved files"
        
    Note: At least one parameter should be provided. You can combine parameters to narrow down results.
    :param status:
    :param file_id:
    :param filename:
    :param ctx:
    """

    session_factory = ctx.request_context.lifespan_context["session_factory"]

    return await query_files(session_factory, filename, file_id, status)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")