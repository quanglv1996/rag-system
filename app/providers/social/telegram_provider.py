"""Telegram Bot API provider.

Wraps the Telegram Bot API for sending messages, managing webhooks,
handling inline keyboards, sending media (photos, videos, documents,
voice), managing groups and channels, and scheduling.

Provider layer only — no business logic.
"""

from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.common.constants import TELEGRAM_BASE_URL
from app.core.config import get_settings
from app.core.exception import ProviderException, RateLimitException
from app.core.logger import get_logger
from app.interfaces.social_provider import SocialMessage, SocialPost, SocialProvider

logger = get_logger(__name__)

_RETRY_CONFIG = dict(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)


class TelegramProvider(SocialProvider):
    """Telegram Bot API provider implementation.

    Attributes:
        _base_url: Full Telegram Bot API URL including bot token.
        _client: Shared async HTTP client.
    """

    def __init__(self) -> None:
        """Initialize the Telegram provider from application settings."""
        settings = get_settings()

        if not settings.telegram_bot_token:
            raise ProviderException(
                "Telegram bot token is not configured",
                provider="telegram",
                operation="init",
            )

        self._base_url = f"{TELEGRAM_BASE_URL}{settings.telegram_bot_token}"
        self._webhook_url = settings.telegram_webhook_url

        self._client = httpx.AsyncClient(timeout=settings.http_timeout)
        logger.info("Telegram provider initialized")

    @property
    def platform_name(self) -> str:
        """Return platform identifier.

        Returns:
            str: 'telegram'
        """
        return "telegram"

    async def _call(
        self, method: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call a Telegram Bot API method.

        Args:
            method: Telegram API method name (e.g., 'sendMessage').
            payload: Optional request body parameters.

        Returns:
            dict[str, Any]: API result field from the response.

        Raises:
            RateLimitException: On 429 Too Many Requests.
            ProviderException: On API errors.
        """
        url = f"{self._base_url}/{method}"

        try:
            response = await self._client.post(url, json=payload or {})
            data: dict[str, Any] = response.json()

            if response.status_code == 429:
                retry_after = (
                    response.headers.get("Retry-After")
                    or data.get("parameters", {}).get("retry_after", 30)
                )
                raise RateLimitException(
                    "Telegram rate limit exceeded",
                    retry_after=int(retry_after),
                )

            if not data.get("ok"):
                raise ProviderException(
                    message=data.get("description", "Telegram API error"),
                    provider="telegram",
                    operation=method,
                    details={"error_code": data.get("error_code")},
                )

            return data.get("result", {})  # type: ignore[return-value]

        except (RateLimitException, ProviderException):
            raise
        except httpx.RequestError as exc:
            raise ProviderException(
                f"Telegram connection error: {exc}",
                provider="telegram",
                operation=method,
            ) from exc

    # =========================================================================
    # SocialProvider interface
    # =========================================================================

    @retry(**_RETRY_CONFIG)
    async def post(self, post: SocialPost) -> dict[str, Any]:
        """Send a message to a Telegram chat/channel.

        Args:
            post: Post data. metadata must contain 'chat_id'.
                  metadata may contain 'parse_mode' ('Markdown', 'HTML').

        Returns:
            dict[str, Any]: Sent message object.
        """
        payload: dict[str, Any] = {
            "chat_id": post.metadata.get("chat_id"),
            "text": post.content,
            "parse_mode": post.metadata.get("parse_mode", "HTML"),
        }

        if post.metadata.get("reply_markup"):
            payload["reply_markup"] = post.metadata["reply_markup"]

        return await self._call("sendMessage", payload)

    @retry(**_RETRY_CONFIG)
    async def send_message(self, message: SocialMessage) -> dict[str, Any]:
        """Send a direct message to a Telegram user or chat.

        Args:
            message: Message with recipient_id as chat_id.

        Returns:
            dict[str, Any]: Sent message object.
        """
        payload: dict[str, Any] = {
            "chat_id": message.recipient_id,
            "text": message.content,
            "parse_mode": message.metadata.get("parse_mode", "HTML"),
        }

        return await self._call("sendMessage", payload)

    # =========================================================================
    # Telegram-specific extended operations
    # =========================================================================

    async def get_me(self) -> dict[str, Any]:
        """Retrieve information about the bot itself.

        Returns:
            dict[str, Any]: Bot user object.
        """
        return await self._call("getMe")

    async def send_photo(
        self,
        chat_id: str | int,
        photo_url: str,
        caption: str = "",
        parse_mode: str = "HTML",
    ) -> dict[str, Any]:
        """Send a photo to a chat.

        Args:
            chat_id: Target chat or channel ID.
            photo_url: URL of the photo to send.
            caption: Optional caption text.
            parse_mode: Caption parse mode.

        Returns:
            dict[str, Any]: Sent message object.
        """
        return await self._call(
            "sendPhoto",
            {
                "chat_id": chat_id,
                "photo": photo_url,
                "caption": caption,
                "parse_mode": parse_mode,
            },
        )

    async def send_video(
        self,
        chat_id: str | int,
        video_url: str,
        caption: str = "",
        parse_mode: str = "HTML",
    ) -> dict[str, Any]:
        """Send a video to a chat.

        Args:
            chat_id: Target chat or channel ID.
            video_url: URL of the video to send.
            caption: Optional caption.
            parse_mode: Caption parse mode.

        Returns:
            dict[str, Any]: Sent message object.
        """
        return await self._call(
            "sendVideo",
            {
                "chat_id": chat_id,
                "video": video_url,
                "caption": caption,
                "parse_mode": parse_mode,
            },
        )

    async def send_document(
        self,
        chat_id: str | int,
        document_url: str,
        caption: str = "",
    ) -> dict[str, Any]:
        """Send a document to a chat.

        Args:
            chat_id: Target chat or channel ID.
            document_url: URL of the file to send.
            caption: Optional caption.

        Returns:
            dict[str, Any]: Sent message object.
        """
        return await self._call(
            "sendDocument",
            {"chat_id": chat_id, "document": document_url, "caption": caption},
        )

    async def send_voice(
        self, chat_id: str | int, voice_url: str
    ) -> dict[str, Any]:
        """Send a voice message to a chat.

        Args:
            chat_id: Target chat or channel ID.
            voice_url: URL of the OGG voice file.

        Returns:
            dict[str, Any]: Sent message object.
        """
        return await self._call(
            "sendVoice", {"chat_id": chat_id, "voice": voice_url}
        )

    async def send_with_inline_keyboard(
        self,
        chat_id: str | int,
        text: str,
        keyboard_buttons: list[list[dict[str, str]]],
        parse_mode: str = "HTML",
    ) -> dict[str, Any]:
        """Send a message with an inline keyboard.

        Args:
            chat_id: Target chat ID.
            text: Message text.
            keyboard_buttons: 2D array of button objects
                  e.g. [[{"text": "Yes", "callback_data": "yes"}]].
            parse_mode: Text parse mode.

        Returns:
            dict[str, Any]: Sent message object.
        """
        return await self._call(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": {"inline_keyboard": keyboard_buttons},
            },
        )

    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: str | None = None,
        show_alert: bool = False,
    ) -> dict[str, Any]:
        """Answer a callback query from an inline keyboard button press.

        Args:
            callback_query_id: ID of the callback query to answer.
            text: Optional notification text to show.
            show_alert: If True, show an alert dialog instead of a toast.

        Returns:
            dict[str, Any]: API response (True on success).
        """
        payload: dict[str, Any] = {
            "callback_query_id": callback_query_id,
            "show_alert": show_alert,
        }
        if text:
            payload["text"] = text

        return await self._call("answerCallbackQuery", payload)

    async def set_webhook(
        self, url: str, secret_token: str | None = None
    ) -> dict[str, Any]:
        """Register the webhook URL with Telegram.

        Args:
            url: HTTPS URL where Telegram will send updates.
            secret_token: Optional secret token for X-Telegram-Bot-Api-Secret-Token header.

        Returns:
            dict[str, Any]: API response.
        """
        payload: dict[str, Any] = {"url": url}
        if secret_token:
            payload["secret_token"] = secret_token

        return await self._call("setWebhook", payload)

    async def delete_webhook(self) -> dict[str, Any]:
        """Remove the current webhook (switch to polling mode).

        Returns:
            dict[str, Any]: API response.
        """
        return await self._call("deleteWebhook")

    async def get_updates(
        self,
        offset: int | None = None,
        limit: int = 100,
        timeout: int = 30,
    ) -> list[dict[str, Any]]:
        """Retrieve pending updates via long polling.

        Args:
            offset: Update ID offset (updates before this are skipped).
            limit: Maximum number of updates to return.
            timeout: Long-poll timeout in seconds.

        Returns:
            list[dict[str, Any]]: List of update objects.
        """
        payload: dict[str, Any] = {"limit": limit, "timeout": timeout}
        if offset is not None:
            payload["offset"] = offset

        result = await self._call("getUpdates", payload)
        if isinstance(result, list):
            return result
        return []

    async def forward_message(
        self,
        chat_id: str | int,
        from_chat_id: str | int,
        message_id: int,
    ) -> dict[str, Any]:
        """Forward a message from one chat to another.

        Args:
            chat_id: Target chat ID.
            from_chat_id: Source chat ID.
            message_id: Message ID to forward.

        Returns:
            dict[str, Any]: Sent message object.
        """
        return await self._call(
            "forwardMessage",
            {
                "chat_id": chat_id,
                "from_chat_id": from_chat_id,
                "message_id": message_id,
            },
        )

    async def get_chat_member_count(
        self, chat_id: str | int
    ) -> int:
        """Get the number of members in a chat/group/channel.

        Args:
            chat_id: Chat, group, or channel ID.

        Returns:
            int: Member count.
        """
        result = await self._call("getChatMemberCount", {"chat_id": chat_id})
        return int(result) if result else 0

    async def close(self) -> None:
        """Close the underlying HTTP client connection pool."""
        await self._client.aclose()
