import logging

from redisvl.extensions.llmcache import SemanticCache
from redisvl.query.filter import Tag

logger = logging.getLogger(__name__)

DEFAULT_TTL = 3600  # 1 hour


class CacheManager:
    def __init__(self, embeddings, redis_url: str):
        self.cache = SemanticCache(
            name="azure_ai_agent_cache",
            redis_url=redis_url,
            distance_threshold=0.2,
            filterable_fields=[{"name": "user_role", "type": "tag"}],
        )
        self.embeddings = embeddings

    async def check(self, query: str, user_role: str) -> str | None:
        """Check cache for semantically similar query with same role."""
        role_filter = Tag("user_role") == user_role

        results = await self.cache.acheck(
            prompt=query,
            filter_expression=role_filter,
        )

        if results:
            logger.info(
                "cache.hit",
                extra={
                    "user_role": user_role,
                    "distance": results[0].get("vector_distance"),
                },
            )
            return results[0].get("response")

        logger.info("cache.miss", extra={"user_role": user_role})
        return None

    async def store(self, query: str, response: str, user_role: str) -> None:
        """Store query-response pair with role tag and TTL."""
        await self.cache.astore(
            prompt=query,
            response=response,
            filters={"user_role": user_role},
            ttl=DEFAULT_TTL,
        )
        logger.info(
            "cache.stored",
            extra={"user_role": user_role, "ttl": DEFAULT_TTL},
        )
