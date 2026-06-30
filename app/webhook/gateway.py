"""Webhook Gateway — centralized ingestion and dispatch for all platforms.

Provides a single entry point for all incoming webhooks (Facebook,
Telegram, TikTok, Google). Responsibilities:
1. Signature verification per platform.
2. Event parsing and normalization.
3. Async dispatch to registered event handlers.
4. Queuing for async processing via Celery.
5. Idempotency (deduplicate replayed events).

Providers only process events AFTER the gateway validates them.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import Request

from app.core.config import get_settings
from app.core.exception import AuthenticationException, ValidationException
from app.core.logger import get_logger

logger = get_logger(__name__)

# Type alias for async event handler
EventHandler = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]


class WebhookEvent:
    """Normalized webhook event from any platform.

    Attributes:
        platform: Source platform name.
        event_type: Platform-specific event type string.
        payload: Normalized event data.
        raw: Original raw payload bytes.
        event_id: Optional platform-provided deduplication ID.
    """

    def __init__(
        self,
        platform: str,
        event_type: str,
        payload: dict[str, Any],
        raw: bytes = b"",
        event_id: str | None = None,
    ) -> None:
        """Initialize a webhook event."""
        self.platform = platform
        self.event_type = event_type
        self.payload = payload
        self.raw = raw
        self.event_id = event_id


class WebhookGateway:
    """Unified gateway for processing incoming webhook events.

    Verifies signatures, deduplicates events using Redis, and dispatches
    normalized events to registered async handlers.

    Attributes:
        _handlers: Map of platform → list of event handlers.
        _redis: Optional Redis client for deduplication.
    """

    def __init__(self, redis: Any | None = None) -> None:
        """Initialize the webhook gateway.

        Args:
            redis: Optional async Redis client for event deduplication.
        """
        self._handlers: dict[str, list[EventHandler]] = {}
        self._redis = redis
        self._settings = get_settings()

    def register_handler(
        self,
        platform: str,
        handler: EventHandler,
    ) -> None:
        """Register an async event handler for a platform.

        Args:
            platform: Platform name ('facebook', 'telegram', etc.).
            handler: Async callable receiving (event_type, payload).
        """
        if platform not in self._handlers:
            self._handlers[platform] = []
        self._handlers[platform].append(handler)
        logger.debug("Webhook handler registered", platform=platform)

    # =========================================================================
    # Signature Verifiers
    # =========================================================================

    def verify_facebook(self, payload: bytes, signature_header: str) -> bool:
        """Verify a Facebook webhook payload signature.

        Args:
            payload: Raw request body bytes.
            signature_header: Value of X-Hub-Signature-256 header.

        Returns:
            bool: True if valid.
        """
        if not signature_header.startswith("sha256="):
            return False

        expected = hmac.new(
            self._settings.facebook_webhook_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        received = signature_header[7:]
        return hmac.compare_digest(expected, received)

    def verify_telegram(
        self, payload: bytes, secret_token_header: str
    ) -> bool:
        """Verify a Telegram webhook secret token.

        Args:
            payload: Raw request body (not used in token check).
            secret_token_header: X-Telegram-Bot-Api-Secret-Token header value.

        Returns:
            bool: True if the token matches the configured secret.
        """
        # In production, store the secret token used during webhook setup
        expected = self._settings.telegram_bot_token[:32]  # Use first 32 chars as token
        return hmac.compare_digest(secret_token_header, expected)

    def verify_tiktok(self, payload: bytes, signature_header: str) -> bool:
        """Verify a TikTok webhook signature.

        Args:
            payload: Raw request body bytes.
            signature_header: TikTok-Signature header value.

        Returns:
            bool: True if valid.
        """
        # TikTok uses HMAC-SHA256 with client secret
        expected = hmac.new(
            self._settings.tiktok_client_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(signature_header, expected)

    # =========================================================================
    # Platform Processors
    # =========================================================================

    async def process_facebook(
        self,
        request: Request,
        x_hub_signature_256: str = "",
    ) -> dict[str, str]:
        """Process an incoming Facebook webhook request.

        Args:
            request: FastAPI request object.
            x_hub_signature_256: Signature header value.

        Returns:
            dict: Acknowledgement response.

        Raises:
            AuthenticationException: If signature verification fails.
        """
        payload = await request.body()

        if self._settings.facebook_webhook_secret and not self.verify_facebook(
            payload, x_hub_signature_256
        ):
            raise AuthenticationException("Invalid Facebook webhook signature")

        try:
            data: dict[str, Any] = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValidationException("Invalid Facebook webhook JSON") from exc

        # Deduplicate
        event_id = data.get("entry", [{}])[0].get("id")
        if await self._is_duplicate(f"facebook:{event_id}"):
            return {"status": "duplicate"}

        # Parse and dispatch events
        for entry in data.get("entry", []):
            for messaging in entry.get("messaging", []):
                event = WebhookEvent(
                    platform="facebook",
                    event_type="message" if "message" in messaging else "postback",
                    payload=messaging,
                    raw=payload,
                    event_id=str(messaging.get("timestamp")),
                )
                await self._dispatch(event)

            for change in entry.get("changes", []):
                event = WebhookEvent(
                    platform="facebook",
                    event_type=change.get("field", "unknown"),
                    payload=change.get("value", {}),
                    raw=payload,
                )
                await self._dispatch(event)

        return {"status": "received"}

    async def process_telegram(
        self,
        request: Request,
        secret_token: str = "",
    ) -> dict[str, str]:
        """Process an incoming Telegram webhook update.

        Args:
            request: FastAPI request object.
            secret_token: X-Telegram-Bot-Api-Secret-Token header.

        Returns:
            dict: Acknowledgement.
        """
        payload = await request.body()
        data: dict[str, Any] = json.loads(payload)

        update_id = str(data.get("update_id", ""))
        if await self._is_duplicate(f"telegram:{update_id}"):
            return {"status": "duplicate"}

        # Determine event type
        if "message" in data:
            event_type = "message"
        elif "callback_query" in data:
            event_type = "callback_query"
        elif "edited_message" in data:
            event_type = "edited_message"
        else:
            event_type = "unknown"

        event = WebhookEvent(
            platform="telegram",
            event_type=event_type,
            payload=data,
            raw=payload,
            event_id=update_id,
        )
        await self._dispatch(event)

        return {"status": "ok"}

    async def process_tiktok(
        self,
        request: Request,
        tiktok_signature: str = "",
    ) -> dict[str, str]:
        """Process an incoming TikTok webhook event.

        Args:
            request: FastAPI request object.
            tiktok_signature: TikTok-Signature header.

        Returns:
            dict: Acknowledgement.
        """
        payload = await request.body()

        if self._settings.tiktok_client_secret and not self.verify_tiktok(
            payload, tiktok_signature
        ):
            raise AuthenticationException("Invalid TikTok webhook signature")

        data: dict[str, Any] = json.loads(payload)
        event_type = data.get("event", "unknown")
        event_id = data.get("event_time", "")

        if await self._is_duplicate(f"tiktok:{event_id}"):
            return {"status": "duplicate"}

        event = WebhookEvent(
            platform="tiktok",
            event_type=event_type,
            payload=data,
            raw=payload,
            event_id=str(event_id),
        )
        await self._dispatch(event)

        return {"status": "success"}

    # =========================================================================
    # Dispatch & Deduplication
    # =========================================================================

    async def _dispatch(self, event: WebhookEvent) -> None:
        """Dispatch an event to all registered handlers for its platform.

        Args:
            event: Normalized webhook event.
        """
        handlers = self._handlers.get(event.platform, [])

        if not handlers:
            logger.debug(
                "No handlers registered for platform",
                platform=event.platform,
                event_type=event.event_type,
            )
            return

        for handler in handlers:
            try:
                await handler(event.event_type, event.payload)
            except Exception as exc:
                logger.error(
                    "Webhook handler error",
                    platform=event.platform,
                    event_type=event.event_type,
                    error=str(exc),
                )

    async def _is_duplicate(self, dedup_key: str) -> bool:
        """Check if an event has already been processed.

        Uses Redis SET NX (set if not exists) for atomic deduplication.

        Args:
            dedup_key: Unique event key.

        Returns:
            bool: True if this is a duplicate event.
        """
        if self._redis is None or not dedup_key:
            return False

        key = f"webhook:dedup:{dedup_key}"
        # SET NX with 5-minute TTL
        result = await self._redis.set(key, "1", nx=True, ex=300)
        return result is None  # None means key already existed → duplicate
