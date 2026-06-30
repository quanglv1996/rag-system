"""Tests for the Facebook service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestFacebookService:
    """Tests for FacebookService business logic."""

    @pytest.fixture
    def facebook_service(self, monkeypatch):
        """Create a FacebookService with a mocked provider."""
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("SECRET_KEY", "test-secret-key-at-least-32-characters-long-test")

        with patch("app.providers.social.facebook_provider.FacebookProvider.__init__") as mock_init:
            mock_init.return_value = None
            from app.services.facebook_service import FacebookService
            from app.core.config import get_settings
            get_settings.cache_clear()
            settings = get_settings()
            service = FacebookService(settings=settings)
            service._provider = MagicMock()
            service._provider.post = AsyncMock(return_value={"id": "post_123"})
            service._provider.edit_post = AsyncMock(return_value={"success": True})
            service._provider.delete_post = AsyncMock(return_value={"success": True})
            service._provider.send_message = AsyncMock(return_value={"message_id": "msg_456"})
            service._provider.create_comment = AsyncMock(return_value={"id": "comment_789"})
            service._provider.get_page_info = AsyncMock(return_value={"id": "page_1", "name": "Test Page"})
            return service

    @pytest.mark.asyncio
    async def test_publish_post_success(self, facebook_service):
        """Test successful post publication."""
        result = await facebook_service.publish_post(content="Hello, world!")
        assert result["id"] == "post_123"

    @pytest.mark.asyncio
    async def test_publish_post_empty_content_raises(self, facebook_service):
        """Test that empty content raises ValidationException."""
        from app.core.exception import ValidationException

        with pytest.raises(ValidationException):
            await facebook_service.publish_post(content="")

    @pytest.mark.asyncio
    async def test_publish_post_whitespace_raises(self, facebook_service):
        """Test that whitespace-only content raises ValidationException."""
        from app.core.exception import ValidationException

        with pytest.raises(ValidationException):
            await facebook_service.publish_post(content="   ")

    @pytest.mark.asyncio
    async def test_send_message_success(self, facebook_service):
        """Test successful Messenger message send."""
        result = await facebook_service.send_messenger_message(
            recipient_id="user_123", text="Hello!"
        )
        assert result["message_id"] == "msg_456"

    @pytest.mark.asyncio
    async def test_update_post_success(self, facebook_service):
        """Test successful post update."""
        result = await facebook_service.update_post("post_123", "Updated content")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_update_post_empty_content_raises(self, facebook_service):
        """Test that updating with empty content raises ValidationException."""
        from app.core.exception import ValidationException

        with pytest.raises(ValidationException):
            await facebook_service.update_post("post_123", "")

    def test_verify_webhook_signature(self, facebook_service):
        """Test webhook signature verification delegates to provider."""
        facebook_service._provider.verify_webhook_signature = MagicMock(return_value=True)
        result = facebook_service.verify_webhook_signature(b"payload", "sha256=abc")
        assert result is True

    def test_handle_webhook_challenge_valid(self, facebook_service):
        """Test valid webhook challenge returns challenge string."""
        facebook_service._provider.verify_webhook_challenge = MagicMock(
            return_value="challenge_string"
        )
        result = facebook_service.handle_webhook_challenge(
            "subscribe", "valid_token", "challenge_string"
        )
        assert result == "challenge_string"
