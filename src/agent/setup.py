from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.config import get_settings, Settings
from src.rag.embeddings import create_embeddings
from src.rag.store import create_index, load_index, save_index
from src.rag.sparse import BM25Retriever
from src.rag.retriever import HybridRetriever, AzureSearchRetriever

from src.db.connection import init_db
from src.tools.file_status import create_file_status_tool
from src.db.seed import seed_database

from src.tools.rag_tool import create_rag_tool
from src.agent.graph import build_graph
from src.agent.cache import CacheManager


async def setup_agent(settings: Settings | None = None):
    """
    Initialize and configure the AI agent with all necessary components.

    Sets up: LLM, RAG pipeline, database, tools, semantic cache, guardrails,
    and compiles the LangGraph state graph.

    Returns:
        A compiled LangGraph state graph ready to process user queries.
    """
    if settings is None:
        settings = get_settings()

    # Initialize the LLM with GLM configuration
    llm = ChatOpenAI(
        base_url=settings.glm_base_url,
        model=settings.chat_model,
        api_key=settings.glm_api_key.get_secret_value()
    )

    # Create embeddings for vector similarity search
    embedding = create_embeddings(settings)

    # Attempt to load existing FAISS index, or set to None if not found
    try:
        faiss_index = load_index(settings.vector_store_path, embedding)
    except FileNotFoundError:
        faiss_index = None

    retriever = None
    if settings.search_backend == "azure":
        retriever = AzureSearchRetriever(embedding, settings)
    elif settings.search_backend == "faiss":
        # Set up hybrid retriever combining dense (FAISS) and sparse (BM25) retrieval
        if faiss_index is not None:
           docs = list(faiss_index.docstore._dict.values())
           bm25 = BM25Retriever(docs)
           retriever = HybridRetriever(faiss_index, bm25, embedding, settings)

    # Build tools list based on available components
    tools = []
    if retriever is not None:
        rag_tool = create_rag_tool(retriever)
        tools.append(rag_tool)

    # Initialize database connection and seed with sample data
    engine, session_factory = await init_db()
    await seed_database(session_factory)

    # Initialize PostgreSQL checkpoint saver
    checkpointer = AsyncPostgresSaver(url=f"postgresql://{settings.postgres_user}:{settings.postgres_password.get_secret_value()}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
    await checkpointer.setup()

    # File status tool for querying document workflow status
    file_status_tool = create_file_status_tool(session_factory)
    tools.append(file_status_tool)

    # Initialize semantic cache with role-based filtering
    cache_manager = CacheManager(
        embeddings=embedding,
        redis_url=settings.redis_url,
    )

    # Bind tools to LLM for function calling capabilities
    llm_with_tools = llm.bind_tools(tools)

    # Compile the agent graph with all nodes
    graph = build_graph(llm_with_tools, tools, cache_manager, checkpointer)

    return graph, session_factory
