"""Notification System — multi-channel alerting for workflow events.

Supports:
- Telegram (bot messages)
- Email (SMTP)
- Webhook (HTTP POST)
- Slack
- Discord

Notifiers implement the BaseNotifier ABC. The NotificationManager
dispatches to all configured channels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.logger import get_logger

logger = get_logger(__name__)


class BaseNotifier(ABC):
    """Abstract base for notification channel implementations."""

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Return the channel identifier.

        Returns:
            str: Channel name.
        """
        ...

    @abstractmethod
    async def send(
        self,
        title: str,
        message: str,
        recipients: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Send a notification.

        Args:
            title: Notification title.
            message: Notification body.
            recipients: Channel-specific recipient list.
            extra: Optional channel-specific extra parameters.

        Returns:
            bool: True if sent successfully.
        """
        ...


class TelegramNotifier(BaseNotifier):
    """Notification channel that sends Telegram messages."""

    def __init__(self, telegram_service: Any) -> None:
        """Initialize with a Telegram service.

        Args:
            telegram_service: TelegramService instance.
        """
        self._service = telegram_service

    @property
    def channel_name(self) -> str:
        """Return channel name."""
        return "telegram"

    async def send(
        self,
        title: str,
        message: str,
        recipients: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Send a Telegram notification.

        Args:
            title: Notification title (prepended to message).
            message: Notification body.
            recipients: List of Telegram chat IDs.
            extra: Additional parameters.

        Returns:
            bool: True if all messages sent successfully.
        """
        chat_ids = recipients or []
        text = f"<b>{title}</b>\n\n{message}"

        success = True
        for chat_id in chat_ids:
            try:
                await self._service.send_text(chat_id=chat_id, text=text)
            except Exception as exc:
                logger.error("Telegram notification failed", chat_id=chat_id, error=str(exc))
                success = False

        return success


class WebhookNotifier(BaseNotifier):
    """Notification channel that POSTs to a webhook URL."""

    def __init__(self, webhook_url: str) -> None:
        """Initialize with a webhook URL.

        Args:
            webhook_url: Target HTTP endpoint.
        """
        self._url = webhook_url

    @property
    def channel_name(self) -> str:
        """Return channel name."""
        return "webhook"

    async def send(
        self,
        title: str,
        message: str,
        recipients: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """POST a notification to the webhook URL.

        Args:
            title: Event title.
            message: Event message.
            recipients: Not used for webhook.
            extra: Additional fields to include in the POST body.

        Returns:
            bool: True if HTTP 2xx received.
        """
        import httpx

        payload = {
            "title": title,
            "message": message,
            **(extra or {}),
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(self._url, json=payload)
                return response.status_code < 400
        except Exception as exc:
            logger.error("Webhook notification failed", url=self._url, error=str(exc))
            return False


class EmailNotifier(BaseNotifier):
    """SMTP email notification channel."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        use_tls: bool = True,
    ) -> None:
        """Initialize SMTP email notifier.

        Args:
            smtp_host: SMTP server hostname.
            smtp_port: SMTP server port.
            username: SMTP authentication username.
            password: SMTP authentication password.
            from_email: Sender email address.
            use_tls: Enable STARTTLS.
        """
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._username = username
        self._password = password
        self._from_email = from_email
        self._use_tls = use_tls

    @property
    def channel_name(self) -> str:
        """Return channel name."""
        return "email"

    async def send(
        self,
        title: str,
        message: str,
        recipients: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Send an email notification.

        Args:
            title: Email subject.
            message: Email body (plain text).
            recipients: List of recipient email addresses.
            extra: Not used.

        Returns:
            bool: True if sent successfully.
        """
        import asyncio
        import smtplib
        from email.mime.text import MIMEText

        to_emails = recipients or []
        if not to_emails:
            return False

        def _send_sync() -> bool:
            try:
                msg = MIMEText(message, "plain", "utf-8")
                msg["Subject"] = title
                msg["From"] = self._from_email
                msg["To"] = ", ".join(to_emails)

                with smtplib.SMTP(self._smtp_host, self._smtp_port) as smtp:
                    if self._use_tls:
                        smtp.starttls()
                    smtp.login(self._username, self._password)
                    smtp.sendmail(self._from_email, to_emails, msg.as_string())
                return True
            except Exception as exc:
                logger.error("Email notification failed", error=str(exc))
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _send_sync)


class SlackNotifier(BaseNotifier):
    """Slack webhook notification channel."""

    def __init__(self, webhook_url: str) -> None:
        """Initialize with a Slack incoming webhook URL.

        Args:
            webhook_url: Slack Incoming Webhook URL.
        """
        self._url = webhook_url

    @property
    def channel_name(self) -> str:
        """Return channel name."""
        return "slack"

    async def send(
        self,
        title: str,
        message: str,
        recipients: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Post to a Slack channel via incoming webhook.

        Args:
            title: Block title.
            message: Notification text.
            recipients: Not used (channel set via webhook URL).
            extra: Additional Slack blocks.

        Returns:
            bool: True if successful.
        """
        import httpx

        payload = {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": title},
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": message},
                },
            ]
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(self._url, json=payload)
                return response.status_code == 200
        except Exception as exc:
            logger.error("Slack notification failed", error=str(exc))
            return False


class NotificationManager:
    """Dispatches notifications across multiple configured channels.

    Attributes:
        _notifiers: List of registered notifier instances.
    """

    def __init__(self) -> None:
        """Initialize with an empty notifier list."""
        self._notifiers: list[BaseNotifier] = []

    def add_notifier(self, notifier: BaseNotifier) -> None:
        """Register a notification channel.

        Args:
            notifier: Notifier instance to add.
        """
        self._notifiers.append(notifier)
        logger.info("Notifier registered", channel=notifier.channel_name)

    async def notify(
        self,
        title: str,
        message: str,
        recipients: list[str] | None = None,
        channels: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        """Send a notification to all (or selected) channels.

        Args:
            title: Notification title.
            message: Notification body.
            recipients: Channel-specific recipient list.
            channels: Optional subset of channel names to use.
            extra: Optional extra parameters.

        Returns:
            dict[str, bool]: Map of channel name → success status.
        """
        results: dict[str, bool] = {}

        for notifier in self._notifiers:
            if channels and notifier.channel_name not in channels:
                continue

            try:
                success = await notifier.send(title, message, recipients, extra)
                results[notifier.channel_name] = success
            except Exception as exc:
                logger.error(
                    "Notification failed",
                    channel=notifier.channel_name,
                    error=str(exc),
                )
                results[notifier.channel_name] = False

        return results

    async def notify_error(self, error: str, context: str = "") -> None:
        """Convenience method for error notifications.

        Args:
            error: Error message.
            context: Optional context description.
        """
        title = "⚠️ Error Alert"
        message = f"**Error**: {error}"
        if context:
            message += f"\n**Context**: {context}"

        await self.notify(title=title, message=message)
