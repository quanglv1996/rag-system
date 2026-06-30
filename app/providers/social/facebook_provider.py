"""Facebook Graph API provider.

Wraps the Facebook Graph API for page management, posting, messaging,
comments, media uploads, webhooks, and insights. Implements the
SocialProvider interface plus additional Facebook-specific operations.

Business logic must NOT reside here — this provider only wraps raw API calls.
All orchestration is handled in FacebookService.
"""

import hashlib
import hmac
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.common.constants import FACEBOOK_BASE_URL
from app.core.config import get_settings
from app.core.exception import ProviderException, RateLimitException
from app.core.logger import get_logger
from app.interfaces.social_provider import SocialMessage, SocialPost, SocialProvider

logger = get_logger(__name__)

# Retry for transient network errors
_RETRY_CONFIG = dict(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)


class FacebookProvider(SocialProvider):
    """Facebook Graph API provider implementation.

    Provides raw API access for all Facebook operations.
    Services handle business logic; this class handles HTTP mechanics.

    Attributes:
        _base_url: Facebook Graph API base URL with version.
        _page_token: Facebook Page access token.
        _app_secret: Facebook App secret for webhook verification.
        _client: Shared async HTTP client.
    """

    def __init__(self) -> None:
        """Initialize the Facebook provider from application settings."""
        settings = get_settings()
        self._base_url = f"{FACEBOOK_BASE_URL}/{settings.facebook_api_version}"
        self._page_token = settings.facebook_page_token
        self._app_secret = settings.facebook_app_secret
        self._app_id = settings.facebook_app_id
        self._verify_token = settings.facebook_verify_token
        self._webhook_secret = settings.facebook_webhook_secret

        self._client = httpx.AsyncClient(
            timeout=settings.http_timeout,
            headers={"Content-Type": "application/json"},
        )

        logger.info("Facebook provider initialized", api_version=settings.facebook_api_version)

    @property
    def platform_name(self) -> str:
        """Return platform identifier.

        Returns:
            str: 'facebook'
        """
        return "facebook"

    def _get_auth_params(self, token: str | None = None) -> dict[str, str]:
        """Build authentication query parameters.

        Args:
            token: Optional access token override. Uses page token if None.

        Returns:
            dict[str, str]: Auth parameters for API requests.
        """
        return {"access_token": token or self._page_token}

    async def _request(
        self,
        method: str,
        endpoint: str,
        token: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an authenticated Facebook Graph API request.

        Args:
            method: HTTP method (GET, POST, DELETE).
            endpoint: API endpoint path (without base URL).
            token: Optional access token override.
            **kwargs: Additional arguments passed to httpx.

        Returns:
            dict[str, Any]: Parsed JSON response.

        Raises:
            RateLimitException: If the API returns a rate limit error.
            ProviderException: For any other API error.
        """
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        params = self._get_auth_params(token)

        if "params" in kwargs:
            params.update(kwargs.pop("params"))

        try:
            response = await self._client.request(
                method, url, params=params, **kwargs
            )

            data: dict[str, Any] = response.json()

            # Check for Facebook API-level errors
            if "error" in data:
                error = data["error"]
                error_code = error.get("code", 0)
                error_msg = error.get("message", "Unknown Facebook API error")

                # Rate limiting error codes
                if error_code in (4, 17, 32, 613):
                    raise RateLimitException(
                        message=f"Facebook rate limit: {error_msg}",
                        retry_after=300,
                    )

                raise ProviderException(
                    message=error_msg,
                    provider="facebook",
                    operation=endpoint,
                    details={"fb_error_code": error_code},
                )

            return data

        except (RateLimitException, ProviderException):
            raise
        except httpx.HTTPStatusError as exc:
            raise ProviderException(
                f"Facebook HTTP error {exc.response.status_code}",
                provider="facebook",
                operation=endpoint,
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderException(
                f"Facebook connection error: {exc}",
                provider="facebook",
                operation=endpoint,
            ) from exc

    # =========================================================================
    # SocialProvider interface methods
    # =========================================================================

    @retry(**_RETRY_CONFIG)
    async def post(self, post: SocialPost) -> dict[str, Any]:
        """Publish a text post to a Facebook Page.

        Args:
            post: Post data with content and optional link.

        Returns:
            dict[str, Any]: API response containing the post ID.
        """
        payload: dict[str, Any] = {"message": post.content}

        if post.metadata.get("link"):
            payload["link"] = post.metadata["link"]

        return await self._request("POST", "me/feed", json=payload)

    @retry(**_RETRY_CONFIG)
    async def send_message(self, message: SocialMessage) -> dict[str, Any]:
        """Send a message via Facebook Messenger.

        Args:
            message: Message data with recipient ID and content.

        Returns:
            dict[str, Any]: API response.
        """
        payload: dict[str, Any] = {
            "recipient": {"id": message.recipient_id},
            "message": {"text": message.content},
        }

        if message.media_url:
            payload["message"] = {
                "attachment": {
                    "type": "image",
                    "payload": {"url": message.media_url, "is_reusable": True},
                }
            }

        return await self._request("POST", "me/messages", json=payload)

    # =========================================================================
    # Facebook-specific extended operations
    # =========================================================================

    async def get_page_info(self, fields: list[str] | None = None) -> dict[str, Any]:
        """Retrieve information about the configured Facebook Page.

        Args:
            fields: Optional list of fields to retrieve.

        Returns:
            dict[str, Any]: Page information.
        """
        params: dict[str, str] = {}
        if fields:
            params["fields"] = ",".join(fields)

        return await self._request("GET", "me", params=params)

    async def create_photo_post(
        self, page_id: str, image_url: str, caption: str = ""
    ) -> dict[str, Any]:
        """Upload and post a photo to a Facebook Page.

        Args:
            page_id: Facebook Page ID.
            image_url: Publicly accessible URL of the image.
            caption: Optional photo caption.

        Returns:
            dict[str, Any]: API response with post ID.
        """
        payload = {"url": image_url, "caption": caption}
        return await self._request("POST", f"{page_id}/photos", json=payload)

    async def delete_post(self, post_id: str) -> dict[str, Any]:
        """Delete a Facebook post by ID.

        Args:
            post_id: ID of the post to delete.

        Returns:
            dict[str, Any]: API response ({"success": true}).
        """
        return await self._request("DELETE", post_id)

    async def edit_post(self, post_id: str, message: str) -> dict[str, Any]:
        """Edit the text of an existing Facebook post.

        Args:
            post_id: ID of the post to edit.
            message: New text content for the post.

        Returns:
            dict[str, Any]: API response.
        """
        return await self._request("POST", post_id, json={"message": message})

    async def create_comment(
        self, object_id: str, message: str
    ) -> dict[str, Any]:
        """Add a comment to a post or other object.

        Args:
            object_id: ID of the post/photo/etc. to comment on.
            message: Comment text content.

        Returns:
            dict[str, Any]: API response with comment ID.
        """
        return await self._request(
            "POST", f"{object_id}/comments", json={"message": message}
        )

    async def reply_to_comment(
        self, comment_id: str, message: str
    ) -> dict[str, Any]:
        """Reply to an existing comment.

        Args:
            comment_id: ID of the comment to reply to.
            message: Reply text content.

        Returns:
            dict[str, Any]: API response with reply ID.
        """
        return await self._request(
            "POST", f"{comment_id}/comments", json={"message": message}
        )

    async def get_post_insights(
        self, post_id: str, metrics: list[str] | None = None
    ) -> dict[str, Any]:
        """Retrieve insights/analytics for a post.

        Args:
            post_id: ID of the post.
            metrics: List of metric names to retrieve.

        Returns:
            dict[str, Any]: Insights data from the API.
        """
        default_metrics = [
            "post_impressions",
            "post_engaged_users",
            "post_reactions_by_type_total",
        ]
        metric_str = ",".join(metrics or default_metrics)

        return await self._request(
            "GET",
            f"{post_id}/insights",
            params={"metric": metric_str},
        )

    async def get_conversations(
        self, folder: str = "inbox"
    ) -> dict[str, Any]:
        """Retrieve conversations from a Messenger inbox.

        Args:
            folder: Conversation folder ('inbox' or 'other').

        Returns:
            dict[str, Any]: Paginated list of conversations.
        """
        return await self._request(
            "GET",
            "me/conversations",
            params={"folder": folder, "fields": "id,snippet,updated_time,participants"},
        )

    async def get_messages(
        self, conversation_id: str
    ) -> dict[str, Any]:
        """Retrieve messages in a conversation.

        Args:
            conversation_id: ID of the conversation.

        Returns:
            dict[str, Any]: Paginated list of messages.
        """
        return await self._request(
            "GET",
            f"{conversation_id}/messages",
            params={"fields": "id,message,from,created_time"},
        )

    def verify_webhook_signature(
        self, payload: bytes, signature: str
    ) -> bool:
        """Verify a Facebook webhook payload signature.

        Facebook signs webhook payloads with HMAC-SHA256 using
        the app secret. This must be verified before processing.

        Args:
            payload: Raw request body bytes.
            signature: Value of the X-Hub-Signature-256 header.

        Returns:
            bool: True if the signature is valid.
        """
        if not signature.startswith("sha256="):
            return False

        expected = hmac.new(
            self._app_secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()

        received = signature[7:]  # Strip "sha256=" prefix
        return hmac.compare_digest(expected, received)

    def verify_webhook_challenge(
        self, mode: str, verify_token: str, challenge: str
    ) -> str | None:
        """Verify a Facebook webhook subscription challenge.

        Args:
            mode: Must be 'subscribe'.
            verify_token: Token from the webhook request.
            challenge: Challenge string to echo back.

        Returns:
            str | None: Challenge string if valid, None otherwise.
        """
        if mode == "subscribe" and verify_token == self._verify_token:
            return challenge
        return None

    async def get_long_lived_token(self, short_lived_token: str) -> dict[str, Any]:
        """Exchange a short-lived user token for a long-lived one.

        Args:
            short_lived_token: Short-lived (60-minute) user access token.

        Returns:
            dict[str, Any]: Response containing long-lived access_token.
        """
        return await self._request(
            "GET",
            "oauth/access_token",
            params={
                "grant_type": "fb_exchange_token",
                "client_id": self._app_id,
                "client_secret": self._app_secret,
                "fb_exchange_token": short_lived_token,
            },
        )

    async def close(self) -> None:
        """Close the underlying HTTP client connection pool."""
        await self._client.aclose()
