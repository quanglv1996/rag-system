"""Tests for the OpenAI provider implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.exception import ProviderException, RateLimitException
from app.schemas.ai import (
    ChatMessage,
    ChatRequest,
    EmbeddingRequest,
)
from app.common.enums import MessageRole


class TestOpenAIProvider:
    """Test suite for OpenAIProvider."""

    @pytest.fixture
    def settings_with_openai_key(self, monkeypatch):
        """Patch settings to provide a fake OpenAI API key."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-testing")
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("SECRET_KEY", "test-secret-key-at-least-32-characters-long-test")

    @pytest.fixture
    def openai_provider(self, settings_with_openai_key):
        """Create an OpenAIProvider instance with mocked API key."""
        from app.core.config import get_settings
        get_settings.cache_clear()
        
        with patch("openai.AsyncOpenAI"):
            from app.providers.ai.openai_provider import OpenAIProvider
            provider = OpenAIProvider()
            return provider

    def test_provider_name(self, openai_provider):
        """Test that provider_name returns 'openai'."""
        assert openai_provider.provider_name == "openai"

    def test_get_capabilities(self, openai_provider):
        """Test that all capabilities are enabled for OpenAI."""
        caps = openai_provider.get_capabilities()
        assert caps["chat"] is True
        assert caps["stream"] is True
        assert caps["embedding"] is True
        assert caps["image_generation"] is True
        assert caps["vision"] is True

    @pytest.mark.asyncio
    async def test_chat_success(self, openai_provider):
        """Test successful chat completion."""
        from app.schemas.ai import ChatUsage

        # Mock the OpenAI client response
        mock_choice = MagicMock()
        mock_choice.message.content = "4"
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "gpt-4o"

        openai_provider._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        request = ChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="What is 2+2?")]
        )
        response = await openai_provider.chat(request)

        assert response.content == "4"
        assert response.provider == "openai"
        assert response.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_chat_rate_limit_raises(self, openai_provider):
        """Test that rate limit errors are mapped to RateLimitException."""
        import openai as openai_lib

        openai_provider._client.chat.completions.create = AsyncMock(
            side_effect=openai_lib.RateLimitError(
                "Rate limit exceeded", response=MagicMock(), body={}
            )
        )

        request = ChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="test")]
        )

        with pytest.raises(RateLimitException):
            await openai_provider.chat(request)

    @pytest.mark.asyncio
    async def test_embedding_success(self, openai_provider):
        """Test successful embedding generation."""
        mock_item = MagicMock()
        mock_item.embedding = [0.1, 0.2, 0.3]

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.total_tokens = 5

        mock_response = MagicMock()
        mock_response.data = [mock_item]
        mock_response.model = "text-embedding-3-small"
        mock_response.usage = mock_usage

        openai_provider._client.embeddings.create = AsyncMock(
            return_value=mock_response
        )

        request = EmbeddingRequest(texts=["Hello world"])
        response = await openai_provider.embedding(request)

        assert len(response.embeddings) == 1
        assert response.embeddings[0] == [0.1, 0.2, 0.3]
        assert response.provider == "openai"

    @pytest.mark.asyncio
    async def test_embedding_api_error_raises(self, openai_provider):
        """Test that API errors are mapped to ProviderException."""
        import openai as openai_lib

        openai_provider._client.embeddings.create = AsyncMock(
            side_effect=openai_lib.APIError("Test error", request=MagicMock(), body={})
        )

        request = EmbeddingRequest(texts=["test"])

        with pytest.raises(ProviderException) as exc_info:
            await openai_provider.embedding(request)

        assert exc_info.value.provider == "openai"
