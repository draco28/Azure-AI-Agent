from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text
from redis.asyncio import from_url as redis_from_url

from src.db.connection import get_database_url
from src.config import get_settings

router = APIRouter(prefix="/api")


@router.get("/health")
async def health_check():
    settings = get_settings()
    checks = {}

    # PostgreSQL connectivity check
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine(get_database_url(settings))
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        checks["postgres"] = "healthy"
    except Exception as e:
        checks["postgres"] = f"unhealthy: {e}"

    # Redis connectivity check
    try:
        redis = redis_from_url(settings.redis_url)
        await redis.ping()
        await redis.aclose()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {e}"

    overall = "healthy" if all(v == "healthy" for v in checks.values()) else "degraded"

    status_code = 200 if overall == "healthy" else 503
    return JSONResponse(status_code=status_code, content={"status": overall, "dependencies": checks})
