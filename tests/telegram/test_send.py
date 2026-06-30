"""Tests for Telegram service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.exception import ValidationException


class TestTelegramService:
    """Tests for TelegramService business logic."""

    @pytest.fixture
    def telegram_service(self, monkeypatch):
        """Create a TelegramService with a mocked provider."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:ABC-test")
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("SECRET_KEY", "test-secret-key-at-least-32-characters-long-test")

        with patch("app.providers.social.telegram_provider.TelegramProvider.__init__") as mock_init:
            mock_init.return_value = None
            from app.services.telegram_service import TelegramService
            from app.core.config import get_settings
            get_settings.cache_clear()
            settings = get_settings()
            service = TelegramService(settings=settings)
            service._provider = MagicMock()
            service._provider.send_message = AsyncMock(
                return_value={"message_id": 101, "chat": {"id": 12345}}
            )
            service._provider.get_me = AsyncMock(
                return_value={"id": 99, "username": "TestBot", "is_bot": True}
            )
            service._provider.set_webhook = AsyncMock(return_value={"ok": True})
            return service

    @pytest.mark.asyncio
    async def test_send_text_success(self, telegram_service):
        """Test successful text message sending."""
        result = await telegram_service.send_text(chat_id=12345, text="Hello!")
        assert result["message_id"] == 101

    @pytest.mark.asyncio
    async def test_send_text_empty_raises(self, telegram_service):
        """Test that empty text raises ValidationException."""
        with pytest.raises(ValidationException):
            await telegram_service.send_text(chat_id=12345, text="")

    @pytest.mark.asyncio
    async def test_get_bot_info(self, telegram_service):
        """Test bot info retrieval."""
        result = await telegram_service.get_bot_info()
        assert result["username"] == "TestBot"
        assert result["is_bot"] is True

    @pytest.mark.asyncio
    async def test_setup_webhook_success(self, telegram_service):
        """Test webhook setup with valid HTTPS URL."""
        result = await telegram_service.setup_webhook("https://example.com/webhook")
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_setup_webhook_http_raises(self, telegram_service):
        """Test that HTTP (non-HTTPS) webhook URL raises ValidationException."""
        with pytest.raises(ValidationException):
            await telegram_service.setup_webhook("http://insecure.com/webhook")
