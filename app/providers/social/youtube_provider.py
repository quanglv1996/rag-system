"""YouTube Data API v3 provider.

Wraps the YouTube Data API v3 for video uploads, thumbnails, playlists,
comments, analytics, captions, live streams, and more.

Provider layer only — no business logic.
"""

import json
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.common.constants import (
    GOOGLE_TOKEN_URL,
    YOUTUBE_BASE_URL,
    YOUTUBE_UPLOAD_URL,
)
from app.core.config import get_settings
from app.core.exception import ProviderException, RateLimitException
from app.core.logger import get_logger
from app.interfaces.social_provider import SocialMessage, SocialPost, SocialProvider

logger = get_logger(__name__)

_RETRY_CONFIG = dict(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True,
)


class YouTubeProvider(SocialProvider):
    """YouTube Data API v3 provider implementation.

    Attributes:
        _base_url: YouTube API base URL.
        _upload_url: YouTube resumable upload endpoint.
        _client: Shared async HTTP client.
    """

    def __init__(self) -> None:
        """Initialize the YouTube provider from application settings."""
        settings = get_settings()
        self._base_url = YOUTUBE_BASE_URL
        self._upload_url = YOUTUBE_UPLOAD_URL
        self._client_id = settings.youtube_client_id
        self._client_secret = settings.youtube_client_secret
        self._redirect_uri = settings.youtube_redirect_uri

        self._client = httpx.AsyncClient(timeout=settings.http_timeout)
        logger.info("YouTube provider initialized")

    @property
    def platform_name(self) -> str:
        """Return platform identifier.

        Returns:
            str: 'youtube'
        """
        return "youtube"

    def _auth_headers(self, access_token: str) -> dict[str, str]:
        """Build authorization headers for API requests.

        Args:
            access_token: OAuth2 access token.

        Returns:
            dict[str, str]: Headers dict with Authorization.
        """
        return {"Authorization": f"Bearer {access_token}"}

    async def _request(
        self,
        method: str,
        endpoint: str,
        access_token: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an authenticated YouTube API request.

        Args:
            method: HTTP method.
            endpoint: API endpoint path.
            access_token: OAuth2 access token.
            **kwargs: Additional httpx arguments.

        Returns:
            dict[str, Any]: Parsed JSON response.

        Raises:
            RateLimitException: On 429 responses.
            ProviderException: On API errors.
        """
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        headers = self._auth_headers(access_token)

        try:
            response = await self._client.request(
                method, url, headers=headers, **kwargs
            )

            if response.status_code == 429:
                raise RateLimitException(
                    "YouTube API quota exceeded",
                    retry_after=86400,  # YouTube quotas reset daily
                )

            if response.status_code >= 400:
                error_data = response.json() if response.content else {}
                error_msg = (
                    error_data.get("error", {}).get("message", "YouTube API error")
                )
                raise ProviderException(
                    error_msg,
                    provider="youtube",
                    operation=endpoint,
                    details={"status_code": response.status_code},
                )

            if not response.content:
                return {}

            return response.json()  # type: ignore[no-any-return]

        except (RateLimitException, ProviderException):
            raise
        except httpx.RequestError as exc:
            raise ProviderException(
                f"YouTube connection error: {exc}",
                provider="youtube",
                operation=endpoint,
            ) from exc

    # =========================================================================
    # SocialProvider interface
    # =========================================================================

    @retry(**_RETRY_CONFIG)
    async def post(self, post: SocialPost) -> dict[str, Any]:
        """Initiate a YouTube video upload.

        Args:
            post: Post data. metadata must contain 'access_token'.
                  post.content is used as the video description.
                  metadata['title'] is required for YouTube.

        Returns:
            dict[str, Any]: Upload response with video ID.
        """
        access_token = post.metadata.get("access_token", "")
        video_body = {
            "snippet": {
                "title": post.metadata.get("title", "Untitled Video"),
                "description": post.content,
                "tags": post.metadata.get("tags", []),
                "categoryId": post.metadata.get("category_id", "22"),
            },
            "status": {
                "privacyStatus": post.metadata.get("privacy_status", "public"),
                "selfDeclaredMadeForKids": False,
            },
        }

        # For resumable upload, we initiate here and return the upload URI
        headers = {
            **self._auth_headers(access_token),
            "Content-Type": "application/json",
            "X-Upload-Content-Type": "video/*",
        }

        response = await self._client.post(
            f"{self._upload_url}?uploadType=resumable&part=snippet,status",
            headers=headers,
            content=json.dumps(video_body),
        )

        if response.status_code not in (200, 201):
            raise ProviderException(
                "Failed to initiate YouTube upload",
                provider="youtube",
                operation="upload_initiate",
                details={"status_code": response.status_code},
            )

        upload_uri = response.headers.get("Location", "")
        return {"upload_uri": upload_uri, "video_body": video_body}

    async def send_message(self, message: SocialMessage) -> dict[str, Any]:
        """YouTube does not support direct messages via the API.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "YouTube Data API does not support direct messages"
        )

    # =========================================================================
    # YouTube-specific operations
    # =========================================================================

    async def upload_video_bytes(
        self,
        upload_uri: str,
        video_bytes: bytes,
        content_type: str = "video/mp4",
    ) -> dict[str, Any]:
        """Upload video bytes to a resumable upload URI.

        Args:
            upload_uri: URI returned by the initiate upload call.
            video_bytes: Raw video file bytes.
            content_type: MIME type of the video.

        Returns:
            dict[str, Any]: Completed video resource from YouTube.
        """
        try:
            response = await self._client.put(
                upload_uri,
                content=video_bytes,
                headers={
                    "Content-Type": content_type,
                    "Content-Length": str(len(video_bytes)),
                },
            )

            if response.status_code not in (200, 201):
                raise ProviderException(
                    f"YouTube video upload failed: {response.status_code}",
                    provider="youtube",
                    operation="upload_bytes",
                )

            return response.json()  # type: ignore[no-any-return]

        except (ProviderException, RateLimitException):
            raise
        except httpx.RequestError as exc:
            raise ProviderException(
                f"YouTube upload connection error: {exc}",
                provider="youtube",
                operation="upload_bytes",
            ) from exc

    async def set_thumbnail(
        self, access_token: str, video_id: str, image_bytes: bytes
    ) -> dict[str, Any]:
        """Set a custom thumbnail for a video.

        Args:
            access_token: OAuth2 access token.
            video_id: YouTube video ID.
            image_bytes: JPEG image bytes for the thumbnail.

        Returns:
            dict[str, Any]: Thumbnail resource.
        """
        url = f"{self._base_url}/thumbnails/set"
        headers = {
            **self._auth_headers(access_token),
            "Content-Type": "image/jpeg",
        }
        params = {"videoId": video_id}

        response = await self._client.post(
            url, headers=headers, params=params, content=image_bytes
        )

        if response.status_code >= 400:
            raise ProviderException(
                "Failed to set YouTube thumbnail",
                provider="youtube",
                operation="set_thumbnail",
                details={"status_code": response.status_code},
            )

        return response.json()  # type: ignore[no-any-return]

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
            privacy_status: Privacy setting (public/private/unlisted).

        Returns:
            dict[str, Any]: Created playlist resource.
        """
        body = {
            "snippet": {"title": title, "description": description},
            "status": {"privacyStatus": privacy_status},
        }
        return await self._request(
            "POST",
            "playlists?part=snippet,status",
            access_token=access_token,
            json=body,
        )

    async def add_video_to_playlist(
        self, access_token: str, playlist_id: str, video_id: str
    ) -> dict[str, Any]:
        """Add a video to a playlist.

        Args:
            access_token: OAuth2 access token.
            playlist_id: Target playlist ID.
            video_id: Video to add.

        Returns:
            dict[str, Any]: PlaylistItem resource.
        """
        body = {
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": {"kind": "youtube#video", "videoId": video_id},
            }
        }
        return await self._request(
            "POST",
            "playlistItems?part=snippet",
            access_token=access_token,
            json=body,
        )

    async def list_comments(
        self, access_token: str, video_id: str, max_results: int = 20
    ) -> dict[str, Any]:
        """List top-level comment threads for a video.

        Args:
            access_token: OAuth2 access token.
            video_id: Video ID to retrieve comments for.
            max_results: Maximum number of comment threads.

        Returns:
            dict[str, Any]: CommentThreadListResponse.
        """
        return await self._request(
            "GET",
            f"commentThreads?part=snippet&videoId={video_id}&maxResults={max_results}",
            access_token=access_token,
        )

    async def reply_to_comment(
        self, access_token: str, parent_id: str, text: str
    ) -> dict[str, Any]:
        """Reply to a comment.

        Args:
            access_token: OAuth2 access token.
            parent_id: ID of the parent comment.
            text: Reply text.

        Returns:
            dict[str, Any]: Created comment resource.
        """
        body = {
            "snippet": {
                "parentId": parent_id,
                "textOriginal": text,
            }
        }
        return await self._request(
            "POST",
            "comments?part=snippet",
            access_token=access_token,
            json=body,
        )

    async def get_video_analytics(
        self, access_token: str, video_id: str
    ) -> dict[str, Any]:
        """Retrieve analytics for a video using YouTube Analytics API.

        Args:
            access_token: OAuth2 access token.
            video_id: Video ID.

        Returns:
            dict[str, Any]: Analytics report data.
        """
        analytics_url = (
            f"https://youtubeanalytics.googleapis.com/v2/reports"
            f"?ids=channel==MINE"
            f"&startDate=2024-01-01"
            f"&endDate=2024-12-31"
            f"&metrics=views,likes,dislikes,shares"
            f"&dimensions=video"
            f"&filters=video=={video_id}"
        )

        headers = self._auth_headers(access_token)
        response = await self._client.get(analytics_url, headers=headers)

        if response.status_code >= 400:
            raise ProviderException(
                "YouTube Analytics API error",
                provider="youtube",
                operation="get_analytics",
            )

        return response.json()  # type: ignore[no-any-return]

    async def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        """Refresh an expired OAuth2 access token.

        Args:
            refresh_token: Valid refresh token.

        Returns:
            dict[str, Any]: New token response.
        """
        payload = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }

        response = await self._client.post(GOOGLE_TOKEN_URL, data=payload)
        return response.json()  # type: ignore[no-any-return]

    async def close(self) -> None:
        """Close the underlying HTTP client connection pool."""
        await self._client.aclose()
