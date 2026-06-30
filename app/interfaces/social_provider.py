"""Abstract interface for social media platform providers.

Defines the minimum contract for social platform integrations.
Each platform (Facebook, TikTok, YouTube, Telegram) implements
this interface to provide a unified abstraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SocialPost:
    """Represents a social media post to be created or returned.

    Attributes:
        content: Text content of the post.
        media_urls: Optional list of media attachment URLs.
        platform_post_id: Platform-assigned post ID (set after creation).
        metadata: Platform-specific additional fields.
    """

    content: str
    media_urls: list[str] = field(default_factory=list)
    platform_post_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SocialMessage:
    """Represents a direct/private message to be sent or received.

    Attributes:
        recipient_id: Platform-specific recipient identifier.
        content: Message text content.
        media_url: Optional media attachment URL.
        metadata: Platform-specific additional fields.
    """

    recipient_id: str
    content: str
    media_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SocialProvider(ABC):
    """Abstract base class for all social media provider implementations.

    Concrete providers must implement post() and send_message().
    Additional platform-specific capabilities are optional.
    """

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the name of this social platform.

        Returns:
            str: Platform identifier (e.g., 'facebook', 'telegram').
        """
        ...

    @abstractmethod
    async def post(self, post: SocialPost) -> dict[str, Any]:
        """Publish a post to the social platform.

        Args:
            post: Post data including content and optional media.

        Returns:
            dict[str, Any]: Platform API response with post ID.

        Raises:
            ProviderException: If the API call fails.
            RateLimitException: If rate limit is exceeded.
        """
        ...

    @abstractmethod
    async def send_message(self, message: SocialMessage) -> dict[str, Any]:
        """Send a direct/private message via the platform.

        Args:
            message: Message data including recipient and content.

        Returns:
            dict[str, Any]: Platform API response.

        Raises:
            ProviderException: If the API call fails.
        """
        ...

    async def get_analytics(
        self, post_id: str, metrics: list[str] | None = None
    ) -> dict[str, Any]:
        """Retrieve analytics data for a post.

        Args:
            post_id: Platform-specific post identifier.
            metrics: Optional list of specific metrics to retrieve.

        Returns:
            dict[str, Any]: Analytics data from the platform.

        Raises:
            NotImplementedError: If not supported by this platform.
        """
        raise NotImplementedError(
            f"Platform '{self.platform_name}' does not support analytics via this interface"
        )
