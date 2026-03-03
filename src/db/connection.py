from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from src.config import get_settings, Settings
from src.db.models import Base


def get_database_url(settings: Settings | None = None) -> str:
    if settings is None:
        settings = get_settings()
    return f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password.get_secret_value()}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"


async def init_db(settings: Settings | None = None):
    """Called once at startup. Creates engine, session factory, and tables."""
    if settings is None:
        settings = get_settings()

    engine = create_async_engine(get_database_url(settings))
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # Create tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    return engine, session_factory


async def get_session(session_factory: async_sessionmaker) -> AsyncSession:
    """Yields one session for a single database operation."""
    async with session_factory() as session:
        yield session
