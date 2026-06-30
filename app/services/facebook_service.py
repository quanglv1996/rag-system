"""Facebook Service — business logic for Facebook integration.

Coordinates page management, post lifecycle, messaging, comments,
webhooks, and analytics. All Facebook API calls go through the
FacebookProvider; this service adds scheduling, validation, and
business rules on top.
"""

from typing import Any

from app.core.config import Settings
from app.core.exception import ValidationException
from app.core.logger import get_logger
from app.interfaces.social_provider import SocialMessage, SocialPost
from app.providers.social.facebook_provider import FacebookProvider

logger = get_logger(__name__)


class FacebookService:
    """Business logic service for Facebook integration.

    Attributes:
        _provider: Facebook Graph API provider.
        _settings: Application settings.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize Facebook service with a provider instance.

        Args:
            settings: Application settings.
        """
        self._provider = FacebookProvider()
        self._settings = settings
        logger.info("FacebookService initialized")

    async def publish_post(
        self,
        content: str,
        link: str | None = None,
        media_urls: list[str] | None = None,
    ) -> dict[str, Any]:
        """Publish a text or media post to the Facebook Page.

        Args:
            content: Post text content.
            link: Optional URL to attach to the post.
            media_urls: Optional list of image URLs to attach.

        Returns:
            dict[str, Any]: Published post data including ID.

        Raises:
            ValidationException: If content is empty.
        """
        if not content.strip():
            raise ValidationException("Post content must not be empty", field="content")

        metadata: dict[str, Any] = {}
        if link:
            metadata["link"] = link

        if media_urls:
            # For multi-image posts, publish the first image with caption
            post = SocialPost(
                content=content,
                media_urls=media_urls,
                metadata=metadata,
            )
            # Use photo upload for media posts
            page_info = await self._provider.get_page_info(fields=["id"])
            page_id = page_info.get("id", "me")
            return await self._provider.create_photo_post(
                page_id=page_id,
                image_url=media_urls[0],
                caption=content,
            )

        post = SocialPost(content=content, metadata=metadata)
        result = await self._provider.post(post)

        logger.info("Facebook post published", post_id=result.get("id"))
        return result

    async def update_post(self, post_id: str, content: str) -> dict[str, Any]:
        """Edit an existing Facebook post.

        Args:
            post_id: ID of the post to edit.
            content: New text content.

        Returns:
            dict[str, Any]: API response.

        Raises:
            ValidationException: If content is empty.
        """
        if not content.strip():
            raise ValidationException("Post content must not be empty", field="content")

        result = await self._provider.edit_post(post_id, content)
        logger.info("Facebook post updated", post_id=post_id)
        return result

    async def remove_post(self, post_id: str) -> dict[str, Any]:
        """Delete a Facebook post.

        Args:
            post_id: ID of the post to delete.

        Returns:
            dict[str, Any]: API response ({"success": true}).
        """
        result = await self._provider.delete_post(post_id)
        logger.info("Facebook post deleted", post_id=post_id)
        return result

    async def send_messenger_message(
        self,
        recipient_id: str,
        text: str,
        media_url: str | None = None,
    ) -> dict[str, Any]:
        """Send a message via Facebook Messenger.

        Args:
            recipient_id: PSID of the recipient.
            text: Message text.
            media_url: Optional image/attachment URL.

        Returns:
            dict[str, Any]: API response.

        Raises:
            ValidationException: If text is empty.
        """
        if not text.strip() and not media_url:
            raise ValidationException(
                "Message must have text or media", field="text"
            )

        message = SocialMessage(
            recipient_id=recipient_id,
            content=text,
            media_url=media_url,
        )
        return await self._provider.send_message(message)

    async def add_comment(
        self, object_id: str, text: str
    ) -> dict[str, Any]:
        """Add a comment to a Facebook post or photo.

        Args:
            object_id: ID of the object to comment on.
            text: Comment text.

        Returns:
            dict[str, Any]: Created comment data.
        """
        if not text.strip():
            raise ValidationException("Comment text must not be empty", field="text")

        return await self._provider.create_comment(object_id, text)

    async def reply_to_comment(
        self, comment_id: str, text: str
    ) -> dict[str, Any]:
        """Reply to a comment on a Facebook post.

        Args:
            comment_id: ID of the comment to reply to.
            text: Reply text.

        Returns:
            dict[str, Any]: Created reply data.
        """
        if not text.strip():
            raise ValidationException("Reply text must not be empty", field="text")

        return await self._provider.reply_to_comment(comment_id, text)

    async def get_post_insights(
        self, post_id: str, metrics: list[str] | None = None
    ) -> dict[str, Any]:
        """Retrieve analytics/insights for a Facebook post.

        Args:
            post_id: Facebook post ID.
            metrics: List of metric names to retrieve.

        Returns:
            dict[str, Any]: Insights data.
        """
        return await self._provider.get_post_insights(post_id, metrics)

    async def get_inbox(self) -> dict[str, Any]:
        """Retrieve the page's Messenger inbox conversations.

        Returns:
            dict[str, Any]: List of conversations.
        """
        return await self._provider.get_conversations(folder="inbox")

    async def get_page_info(self) -> dict[str, Any]:
        """Retrieve basic information about the configured Facebook Page.

        Returns:
            dict[str, Any]: Page name, ID, category, and other details.
        """
        return await self._provider.get_page_info(
            fields=["id", "name", "category", "fan_count", "followers_count"]
        )

    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify the signature of an incoming webhook payload.

        Args:
            payload: Raw request body bytes.
            signature: X-Hub-Signature-256 header value.

        Returns:
            bool: True if valid.
        """
        return self._provider.verify_webhook_signature(payload, signature)

    def handle_webhook_challenge(
        self, mode: str, verify_token: str, challenge: str
    ) -> str | None:
        """Handle a Facebook webhook subscription verification challenge.

        Args:
            mode: Must be 'subscribe'.
            verify_token: Token from query parameters.
            challenge: Challenge string to echo back.

        Returns:
            str | None: Challenge if valid, None if invalid.
        """
        return self._provider.verify_webhook_challenge(mode, verify_token, challenge)
