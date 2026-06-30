"""TikTok Service — business logic for TikTok integration."""

from typing import Any

from app.core.config import Settings
from app.core.exception import ValidationException
from app.core.logger import get_logger
from app.interfaces.social_provider import SocialPost
from app.providers.social.tiktok_provider import TikTokProvider

logger = get_logger(__name__)


class TikTokService:
    """Business logic service for TikTok Content Posting API integration."""

    def __init__(self, settings: Settings) -> None:
        """Initialize TikTok service.

        Args:
            settings: Application settings.
        """
        self._provider = TikTokProvider()
        self._settings = settings
        logger.info("TikTokService initialized")

    async def publish_video(
        self,
        access_token: str,
        video_url: str,
        title: str,
        privacy_level: str = "PUBLIC_TO_EVERYONE",
        disable_comment: bool = False,
        disable_duet: bool = False,
        disable_stitch: bool = False,
    ) -> dict[str, Any]:
        """Publish a video to TikTok.

        Args:
            access_token: User access token.
            video_url: Publicly accessible video URL.
            title: Video title/caption.
            privacy_level: Privacy setting for the video.
            disable_comment: Disable comments.
            disable_duet: Disable duet.
            disable_stitch: Disable stitch.

        Returns:
            dict[str, Any]: Publish result with publish_id.

        Raises:
            ValidationException: If title or video_url is missing.
        """
        if not title.strip():
            raise ValidationException("Video title must not be empty", field="title")

        if not video_url.strip():
            raise ValidationException("Video URL must not be empty", field="video_url")

        post = SocialPost(
            content=title,
            media_urls=[video_url],
            metadata={
                "access_token": access_token,
                "privacy_level": privacy_level,
                "disable_comment": disable_comment,
                "disable_duet": disable_duet,
                "disable_stitch": disable_stitch,
            },
        )

        result = await self._provider.post(post)
        logger.info("TikTok video publish initiated", publish_id=result.get("publish_id"))
        return result

    async def exchange_auth_code(self, code: str) -> dict[str, Any]:
        """Exchange OAuth authorization code for tokens.

        Args:
            code: Authorization code from the OAuth callback.

        Returns:
            dict[str, Any]: Token data with access_token and refresh_token.
        """
        return await self._provider.exchange_code_for_token(code)

    async def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        """Refresh an expired access token.

        Args:
            refresh_token: Valid refresh token.

        Returns:
            dict[str, Any]: New token data.
        """
        return await self._provider.refresh_access_token(refresh_token)

    async def get_creator_info(self, access_token: str) -> dict[str, Any]:
        """Get information about the authenticated TikTok creator.

        Args:
            access_token: User access token.

        Returns:
            dict[str, Any]: Creator profile data.
        """
        return await self._provider.get_creator_info(access_token)

    async def get_video_list(
        self, access_token: str, max_count: int = 20
    ) -> dict[str, Any]:
        """List videos for the authenticated creator.

        Args:
            access_token: User access token.
            max_count: Maximum number of videos to return.

        Returns:
            dict[str, Any]: Paginated video list.
        """
        return await self._provider.list_videos(access_token, max_count)

    async def get_video_analytics(
        self,
        access_token: str,
        video_ids: list[str],
    ) -> dict[str, Any]:
        """Get analytics/data for specific TikTok videos.

        Args:
            access_token: User access token.
            video_ids: List of video IDs.

        Returns:
            dict[str, Any]: Video analytics data.
        """
        if not video_ids:
            raise ValidationException("At least one video ID is required", field="video_ids")

        return await self._provider.get_video_query(access_token, video_ids)
