"""YouTube Service — business logic for YouTube integration."""

from typing import Any

from app.core.config import Settings
from app.core.exception import ValidationException
from app.core.logger import get_logger
from app.interfaces.social_provider import SocialPost
from app.providers.social.youtube_provider import YouTubeProvider

logger = get_logger(__name__)


class YouTubeService:
    """Business logic service for YouTube Data API integration."""

    def __init__(self, settings: Settings) -> None:
        """Initialize YouTube service.

        Args:
            settings: Application settings.
        """
        self._provider = YouTubeProvider()
        self._settings = settings
        logger.info("YouTubeService initialized")

    async def upload_video(
        self,
        access_token: str,
        title: str,
        description: str,
        video_url: str | None = None,
        video_bytes: bytes | None = None,
        tags: list[str] | None = None,
        privacy_status: str = "public",
        category_id: str = "22",
    ) -> dict[str, Any]:
        """Upload a video to YouTube.

        Args:
            access_token: OAuth2 access token.
            title: Video title.
            description: Video description.
            video_url: Public URL of the video file (for pull-type upload).
            video_bytes: Raw video bytes (for push-type upload).
            tags: List of tags.
            privacy_status: 'public', 'private', or 'unlisted'.
            category_id: YouTube category ID.

        Returns:
            dict[str, Any]: Upload result with video ID.

        Raises:
            ValidationException: If neither video_url nor video_bytes provided.
        """
        if not title.strip():
            raise ValidationException("Video title must not be empty", field="title")

        if not video_url and not video_bytes:
            raise ValidationException(
                "Either video_url or video_bytes must be provided", field="video"
            )

        post = SocialPost(
            content=description,
            media_urls=[video_url] if video_url else [],
            metadata={
                "access_token": access_token,
                "title": title,
                "tags": tags or [],
                "privacy_status": privacy_status,
                "category_id": category_id,
            },
        )

        result = await self._provider.post(post)

        # If video_bytes provided, upload the actual video data
        if video_bytes and "upload_uri" in result:
            result = await self._provider.upload_video_bytes(
                upload_uri=result["upload_uri"],
                video_bytes=video_bytes,
            )

        logger.info("YouTube video upload initiated", video_id=result.get("id"))
        return result

    async def set_thumbnail(
        self,
        access_token: str,
        video_id: str,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        """Set a custom thumbnail for a YouTube video.

        Args:
            access_token: OAuth2 access token.
            video_id: YouTube video ID.
            image_bytes: JPEG thumbnail image bytes.

        Returns:
            dict[str, Any]: Thumbnail resource data.
        """
        return await self._provider.set_thumbnail(access_token, video_id, image_bytes)

    async def create_playlist(
        self,
        access_token: str,
        title: str,
        description: str = "",
        privacy_status: str = "public",
    ) -> dict[str, Any]:
        """Create a new YouTube playlist.

        Args:
            access_token: OAuth2 access token.
            title: Playlist title.
            description: Playlist description.
            privacy_status: Privacy setting.

        Returns:
            dict[str, Any]: Created playlist resource.
        """
        if not title.strip():
            raise ValidationException("Playlist title must not be empty", field="title")

        return await self._provider.create_playlist(
            access_token, title, description, privacy_status
        )

    async def add_to_playlist(
        self, access_token: str, playlist_id: str, video_id: str
    ) -> dict[str, Any]:
        """Add a video to a YouTube playlist.

        Args:
            access_token: OAuth2 access token.
            playlist_id: Playlist to add to.
            video_id: Video to add.

        Returns:
            dict[str, Any]: PlaylistItem resource.
        """
        return await self._provider.add_video_to_playlist(
            access_token, playlist_id, video_id
        )

    async def get_comments(
        self, access_token: str, video_id: str, max_results: int = 20
    ) -> dict[str, Any]:
        """Retrieve top-level comment threads for a video.

        Args:
            access_token: OAuth2 access token.
            video_id: Video ID.
            max_results: Maximum number of comment threads.

        Returns:
            dict[str, Any]: Comment thread list response.
        """
        return await self._provider.list_comments(access_token, video_id, max_results)

    async def reply_to_comment(
        self, access_token: str, parent_comment_id: str, text: str
    ) -> dict[str, Any]:
        """Post a reply to a YouTube comment.

        Args:
            access_token: OAuth2 access token.
            parent_comment_id: ID of the comment to reply to.
            text: Reply text.

        Returns:
            dict[str, Any]: Created comment resource.
        """
        if not text.strip():
            raise ValidationException("Reply text must not be empty", field="text")

        return await self._provider.reply_to_comment(access_token, parent_comment_id, text)

    async def get_analytics(
        self, access_token: str, video_id: str
    ) -> dict[str, Any]:
        """Get analytics data for a YouTube video.

        Args:
            access_token: OAuth2 access token.
            video_id: Video ID.

        Returns:
            dict[str, Any]: Analytics report.
        """
        return await self._provider.get_video_analytics(access_token, video_id)

    async def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        """Refresh an expired YouTube OAuth2 access token.

        Args:
            refresh_token: Valid refresh token.

        Returns:
            dict[str, Any]: New token response.
        """
        return await self._provider.refresh_token(refresh_token)
