from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.agent.setup import setup_agent
from src.api.routes.health import router as health_router
from src.api.routes.chat import router as chat_router
from src.api.routes.upload import router as upload_router
from src.api.routes.evaluate import router as evaluate_router
from src.api.middleware import LoggingMiddleware
from src.config.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging("api")
    graph, _ = await setup_agent()
    app.state.graph = graph
    yield

app = FastAPI(
    title= "Azure AI Agent",
    version= "0.1.0",
    lifespan=lifespan
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(evaluate_router)
app.add_middleware(LoggingMiddleware)