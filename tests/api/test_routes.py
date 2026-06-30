"""Tests for the AI service layer."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.schemas.ai import ChatMessage, ChatResponse, ChatUsage
from app.common.enums import MessageRole
from app.core.exception import ValidationException


class TestAIService:
    """Tests for AIService business logic."""

    @pytest.fixture
    def ai_service(self, mock_ai_provider):
        """Create an AIService with a mock provider."""
        from app.services.ai_service import AIService

        return AIService(provider=mock_ai_provider)

    @pytest.mark.asyncio
    async def test_chat_success(self, ai_service, sample_chat_messages):
        """Test successful chat completion."""
        response = await ai_service.chat(messages=sample_chat_messages)
        assert response.content == "This is a test response."
        assert response.provider == "mock"

    @pytest.mark.asyncio
    async def test_chat_empty_messages_raises(self, ai_service):
        """Test that empty messages list raises ValidationException."""
        with pytest.raises(ValidationException) as exc_info:
            await ai_service.chat(messages=[])
        assert exc_info.value.error_code == "VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_chat_with_system_prompt(self, ai_service, mock_ai_provider):
        """Test that system prompt is prepended to messages."""
        messages = [ChatMessage(role=MessageRole.USER, content="Hi")]
        await ai_service.chat(messages=messages, system_prompt="Be helpful.")

        call_args = mock_ai_provider.chat.call_args
        request = call_args[0][0]  # First positional arg

        # First message should be the system prompt
        assert request.messages[0].role == MessageRole.SYSTEM
        assert request.messages[0].content == "Be helpful."

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, ai_service):
        """Test successful embedding generation."""
        response = await ai_service.generate_embeddings(["hello", "world"])
        assert len(response.embeddings) == 1  # Mock returns 1

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_raises(self, ai_service):
        """Test that empty texts raise ValidationException."""
        with pytest.raises(ValidationException):
            await ai_service.generate_embeddings([])

    @pytest.mark.asyncio
    async def test_generate_embeddings_all_empty_strings(self, ai_service):
        """Test that all-whitespace texts raise ValidationException."""
        with pytest.raises(ValidationException):
            await ai_service.generate_embeddings(["", "  ", "\t"])

    @pytest.mark.asyncio
    async def test_analyze_image_no_source_raises(self, ai_service):
        """Test that missing image source raises ValidationException."""
        with pytest.raises(ValidationException):
            await ai_service.analyze_image(prompt="Describe this image")

    def test_provider_name(self, ai_service):
        """Test that provider_name reflects the mock provider."""
        assert ai_service.provider_name == "mock"

    def test_get_capabilities(self, ai_service):
        """Test that capabilities are returned from the provider."""
        caps = ai_service.get_provider_capabilities()
        assert "chat" in caps
        assert "embedding" in caps
