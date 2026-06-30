"""Tests for the embedding service (caching layer)."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.rag.embedding import EmbeddingService
from app.schemas.ai import EmbeddingResponse


class TestEmbeddingService:
    """Tests for EmbeddingService caching behavior."""

    @pytest.fixture
    def mock_provider(self):
        """Mock AI provider with embedding capability."""
        provider = MagicMock()
        provider.provider_name = "mock"
        provider.embedding = AsyncMock(
            return_value=EmbeddingResponse(
                embeddings=[[0.1] * 10],
                model="mock-embed",
                provider="mock",
                dimensions=10,
            )
        )
        return provider

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client that simulates cache misses."""
        redis = MagicMock()
        redis.get = AsyncMock(return_value=None)  # Cache miss
        redis.setex = AsyncMock(return_value=True)
        return redis

    @pytest.mark.asyncio
    async def test_embed_single_returns_vector(self, mock_provider, mock_redis):
        """Test that embed_single returns a vector."""
        service = EmbeddingService(provider=mock_provider, redis=mock_redis)
        embedding = await service.embed_single("Hello world")

        assert isinstance(embedding, list)
        assert len(embedding) == 10

    @pytest.mark.asyncio
    async def test_cache_hit_skips_provider(self, mock_provider, mock_redis):
        """Test that a cache hit doesn't call the AI provider."""
        cached_vector = [0.5] * 10
        mock_redis.get = AsyncMock(return_value=json.dumps(cached_vector))

        service = EmbeddingService(provider=mock_provider, redis=mock_redis)
        embedding = await service.embed_single("Cached text")

        assert embedding == cached_vector
        mock_provider.embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_provider(self, mock_provider, mock_redis):
        """Test that a cache miss triggers the provider."""
        mock_redis.get = AsyncMock(return_value=None)

        service = EmbeddingService(provider=mock_provider, redis=mock_redis)
        await service.embed_single("Uncached text")

        mock_provider.embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_texts_batch(self, mock_provider):
        """Test batch embedding of multiple texts."""
        provider = MagicMock()
        provider.provider_name = "mock"
        provider.embedding = AsyncMock(
            return_value=EmbeddingResponse(
                embeddings=[[0.1] * 10, [0.2] * 10, [0.3] * 10],
                model="mock-embed",
                provider="mock",
                dimensions=10,
            )
        )

        service = EmbeddingService(provider=provider, redis=None)
        embeddings = await service.embed_texts(["text1", "text2", "text3"])

        assert len(embeddings) == 3
        assert embeddings[0] == [0.1] * 10

    @pytest.mark.asyncio
    async def test_empty_texts_returns_empty(self, mock_provider):
        """Test that empty input returns empty list without API call."""
        service = EmbeddingService(provider=mock_provider, redis=None)
        result = await service.embed_texts([])

        assert result == []
        mock_provider.embedding.assert_not_called()
