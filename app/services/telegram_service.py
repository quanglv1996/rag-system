"""Telegram Service — business logic for Telegram Bot integration."""

from typing import Any

from app.core.config import Settings
from app.core.exception import ValidationException
from app.core.logger import get_logger
from app.interfaces.social_provider import SocialMessage, SocialPost
from app.providers.social.telegram_provider import TelegramProvider

logger = get_logger(__name__)


class TelegramService:
    """Business logic service for Telegram Bot API integration."""

    def __init__(self, settings: Settings) -> None:
        """Initialize Telegram service.

        Args:
            settings: Application settings.
        """
        self._provider = TelegramProvider()
        self._settings = settings
        logger.info("TelegramService initialized")

    async def send_text(
        self,
        chat_id: str | int,
        text: str,
        parse_mode: str = "HTML",
    ) -> dict[str, Any]:
        """Send a text message to a Telegram chat.

        Args:
            chat_id: Target chat ID.
            text: Message content (supports HTML or Markdown).
            parse_mode: 'HTML' or 'Markdown'.

        Returns:
            dict[str, Any]: Sent message object.

        Raises:
            ValidationException: If text is empty.
        """
        if not text.strip():
            raise ValidationException("Message text must not be empty", field="text")

        message = SocialMessage(
            recipient_id=str(chat_id),
            content=text,
            metadata={"parse_mode": parse_mode},
        )
        return await self._provider.send_message(message)

    async def send_to_channel(
        self,
        channel_id: str,
        text: str,
        parse_mode: str = "HTML",
    ) -> dict[str, Any]:
        """Broadcast a message to a Telegram channel.

        Args:
            channel_id: Channel username (e.g., '@mychannel') or ID.
            text: Message content.
            parse_mode: 'HTML' or 'Markdown'.

        Returns:
            dict[str, Any]: Sent message object.
        """
        post = SocialPost(
            content=text,
            metadata={"chat_id": channel_id, "parse_mode": parse_mode},
        )
        return await self._provider.post(post)

    async def send_photo(
        self,
        chat_id: str | int,
        photo_url: str,
        caption: str = "",
    ) -> dict[str, Any]:
        """Send a photo to a Telegram chat.

        Args:
            chat_id: Target chat ID.
            photo_url: URL of the photo.
            caption: Optional photo caption.

        Returns:
            dict[str, Any]: Sent message object.
        """
        return await self._provider.send_photo(chat_id, photo_url, caption)

    async def send_video(
        self,
        chat_id: str | int,
        video_url: str,
        caption: str = "",
    ) -> dict[str, Any]:
        """Send a video to a Telegram chat.

        Args:
            chat_id: Target chat ID.
            video_url: URL of the video.
            caption: Optional caption.

        Returns:
            dict[str, Any]: Sent message object.
        """
        return await self._provider.send_video(chat_id, video_url, caption)

    async def send_document(
        self, chat_id: str | int, document_url: str, caption: str = ""
    ) -> dict[str, Any]:
        """Send a document to a Telegram chat.

        Args:
            chat_id: Target chat ID.
            document_url: URL of the document.
            caption: Optional caption.

        Returns:
            dict[str, Any]: Sent message object.
        """
        return await self._provider.send_document(chat_id, document_url, caption)

    async def send_inline_keyboard(
        self,
        chat_id: str | int,
        text: str,
        buttons: list[list[dict[str, str]]],
    ) -> dict[str, Any]:
        """Send a message with inline keyboard buttons.

        Args:
            chat_id: Target chat ID.
            text: Message text.
            buttons: 2D button array, e.g.:
                     [[{"text": "Yes", "callback_data": "yes"}]].

        Returns:
            dict[str, Any]: Sent message object.
        """
        return await self._provider.send_with_inline_keyboard(chat_id, text, buttons)

    async def answer_callback(
        self,
        callback_query_id: str,
        text: str | None = None,
        show_alert: bool = False,
    ) -> dict[str, Any]:
        """Answer a callback query from an inline keyboard press.

        Args:
            callback_query_id: ID of the callback query.
            text: Optional notification text.
            show_alert: Show as alert dialog if True.

        Returns:
            dict[str, Any]: API response.
        """
        return await self._provider.answer_callback_query(
            callback_query_id, text, show_alert
        )

    async def setup_webhook(self, webhook_url: str, secret_token: str | None = None) -> dict[str, Any]:
        """Register the webhook URL with Telegram.

        Args:
            webhook_url: HTTPS URL for Telegram to send updates.
            secret_token: Optional signature token.

        Returns:
            dict[str, Any]: Registration result.
        """
        if not webhook_url.startswith("https://"):
            raise ValidationException(
                "Telegram webhook URL must use HTTPS", field="webhook_url"
            )

        return await self._provider.set_webhook(webhook_url, secret_token)

    async def remove_webhook(self) -> dict[str, Any]:
        """Unregister the webhook (switch to polling mode).

        Returns:
            dict[str, Any]: API response.
        """
        return await self._provider.delete_webhook()

    async def get_updates(
        self, offset: int | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Retrieve pending updates via long polling.

        Args:
            offset: Update ID offset.
            limit: Maximum updates to return.

        Returns:
            list[dict[str, Any]]: Update objects.
        """
        return await self._provider.get_updates(offset=offset, limit=limit)

    async def get_bot_info(self) -> dict[str, Any]:
        """Get basic information about the bot.

        Returns:
            dict[str, Any]: Bot user object.
        """
        return await self._provider.get_me()
