"""Embedding generation module for the RAG pipeline.

Provides a cached embedding wrapper around AI providers.
Caches embeddings in Redis to avoid redundant API calls for
the same text content, significantly reducing costs and latency.
"""

import json
from typing import Any

from app.common.constants import (
    CACHE_EMBEDDING_TTL,
    CACHE_PREFIX_EMBEDDING,
)
from app.common.utils import compute_md5
from app.core.exception import RAGException
from app.core.logger import get_logger
from app.interfaces.ai_provider import AIProvider
from app.schemas.ai import EmbeddingRequest

logger = get_logger(__name__)


class EmbeddingService:
    """Embedding service with Redis caching.

    Wraps the AI provider's embedding capability with transparent
    caching. Identical text inputs return cached vectors without
    hitting the API.

    Attributes:
        _provider: AI provider implementing the embedding capability.
        _redis: Optional Redis client for caching.
        _cache_ttl: Cache time-to-live in seconds.
    """

    def __init__(
        self,
        provider: AIProvider,
        redis: Any | None = None,
        cache_ttl: int = CACHE_EMBEDDING_TTL,
    ) -> None:
        """Initialize the embedding service.

        Args:
            provider: AI provider instance for generating embeddings.
            redis: Optional async Redis client for caching.
            cache_ttl: Cache TTL in seconds.
        """
        self._provider = provider
        self._redis = redis
        self._cache_ttl = cache_ttl

    async def embed_texts(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Checks the cache for each text before calling the API.
        Uncached texts are batched into a single API call.

        Args:
            texts: List of strings to embed.
            model: Optional model override.

        Returns:
            list[list[float]]: Embedding vectors, one per input text.

        Raises:
            RAGException: If embedding generation fails.
        """
        if not texts:
            return []

        # Check cache for each text
        cached_results: dict[int, list[float]] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        if self._redis:
            for idx, text in enumerate(texts):
                cache_key = self._make_cache_key(text, model)
                cached = await self._redis.get(cache_key)
                if cached:
                    cached_results[idx] = json.loads(cached)
                else:
                    uncached_indices.append(idx)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        # Fetch uncached embeddings from provider
        new_embeddings: list[list[float]] = []
        if uncached_texts:
            logger.debug(
                "Generating embeddings",
                provider=self._provider.provider_name,
                count=len(uncached_texts),
                cached=len(cached_results),
            )

            try:
                request = EmbeddingRequest(texts=uncached_texts, model=model)
                response = await self._provider.embedding(request)
                new_embeddings = response.embeddings
            except Exception as exc:
                raise RAGException(
                    f"Embedding generation failed: {exc}",
                    stage="embedding",
                ) from exc

            # Store new embeddings in cache
            if self._redis:
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    cache_key = self._make_cache_key(texts[idx], model)
                    await self._redis.setex(
                        cache_key,
                        self._cache_ttl,
                        json.dumps(embedding),
                    )

        # Merge cached and newly generated results in correct order
        final_embeddings: list[list[float]] = []
        new_idx = 0
        for original_idx in range(len(texts)):
            if original_idx in cached_results:
                final_embeddings.append(cached_results[original_idx])
            else:
                if new_idx < len(new_embeddings):
                    final_embeddings.append(new_embeddings[new_idx])
                    new_idx += 1

        return final_embeddings

    async def embed_single(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: Text to embed.
            model: Optional model override.

        Returns:
            list[float]: Embedding vector.
        """
        results = await self.embed_texts([text], model=model)
        return results[0] if results else []

    def _make_cache_key(self, text: str, model: str | None) -> str:
        """Build a Redis cache key for a text/model combination.

        Args:
            text: The text being embedded.
            model: Model name (or None for default).

        Returns:
            str: Redis key string.
        """
        content_hash = compute_md5(text)
        model_suffix = model or self._provider.provider_name
        return f"{CACHE_PREFIX_EMBEDDING}{model_suffix}:{content_hash}"
