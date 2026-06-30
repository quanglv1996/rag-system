"""TikTok Content Posting API provider.

Wraps the TikTok Content Posting API v2 for video uploads, drafts,
publishing, comments, creator info, analytics, and webhook handling.

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

from app.common.constants import TIKTOK_BASE_URL, TIKTOK_TOKEN_URL
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


class TikTokProvider(SocialProvider):
    """TikTok Content Posting API v2 provider.

    Attributes:
        _base_url: TikTok API base URL.
        _client_id: TikTok app client key.
        _client_secret: TikTok app client secret.
        _client: Shared async HTTP client.
    """

    def __init__(self) -> None:
        """Initialize the TikTok provider from application settings."""
        settings = get_settings()
        self._base_url = TIKTOK_BASE_URL
        self._client_id = settings.tiktok_client_id
        self._client_secret = settings.tiktok_client_secret
        self._redirect_uri = settings.tiktok_redirect_uri

        self._client = httpx.AsyncClient(
            timeout=settings.http_timeout,
            headers={"Content-Type": "application/json; charset=UTF-8"},
        )

        logger.info("TikTok provider initialized")

    @property
    def platform_name(self) -> str:
        """Return platform identifier.

        Returns:
            str: 'tiktok'
        """
        return "tiktok"

    async def _request(
        self,
        method: str,
        endpoint: str,
        access_token: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an authenticated TikTok API request.

        Args:
            method: HTTP method.
            endpoint: API endpoint path.
            access_token: User access token.
            **kwargs: Additional httpx arguments.

        Returns:
            dict[str, Any]: Parsed API response data.

        Raises:
            RateLimitException: If TikTok rate limits are hit.
            ProviderException: For any other API error.
        """
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        headers = {"Authorization": f"Bearer {access_token}"}

        try:
            response = await self._client.request(
                method, url, headers=headers, **kwargs
            )
            data: dict[str, Any] = response.json()

            # TikTok wraps responses in {"data": {...}, "error": {...}}
            if "error" in data and data["error"].get("code") != "ok":
                error = data["error"]
                code = error.get("code", "unknown")
                msg = error.get("message", "Unknown TikTok error")

                if "rate_limit" in code.lower() or "quota" in code.lower():
                    raise RateLimitException(
                        message=f"TikTok rate limit: {msg}",
                        retry_after=600,
                    )

                raise ProviderException(
                    message=msg,
                    provider="tiktok",
                    operation=endpoint,
                    details={"tiktok_error_code": code},
                )

            return data.get("data", data)

        except (RateLimitException, ProviderException):
            raise
        except httpx.RequestError as exc:
            raise ProviderException(
                f"TikTok connection error: {exc}",
                provider="tiktok",
                operation=endpoint,
            ) from exc

    # =========================================================================
    # SocialProvider interface
    # =========================================================================

    async def post(self, post: SocialPost) -> dict[str, Any]:
        """Initiate a video upload via TikTok Content Posting API.

        TikTok requires a multi-step process: initialize upload,
        upload chunks, then publish.

        Args:
            post: Post data. metadata must contain 'access_token'
                  and 'video_url' or 'video_bytes'.

        Returns:
            dict[str, Any]: Upload initialization response.
        """
        access_token = post.metadata.get("access_token", "")
        video_title = post.content[:2200]  # TikTok max title length

        payload = {
            "post_info": {
                "title": video_title,
                "privacy_level": post.metadata.get("privacy_level", "PUBLIC_TO_EVERYONE"),
                "disable_duet": post.metadata.get("disable_duet", False),
                "disable_comment": post.metadata.get("disable_comment", False),
                "disable_stitch": post.metadata.get("disable_stitch", False),
            },
            "source_info": {
                "source": "PULL_FROM_URL",
                "video_url": post.media_urls[0] if post.media_urls else "",
            },
        }

        return await self._request(
            "POST",
            "post/video/publish",
            access_token=access_token,
            json=payload,
        )

    async def send_message(self, message: SocialMessage) -> dict[str, Any]:
        """TikTok does not support direct messages via API.

        Raises:
            NotImplementedError: Always — DMs are not supported in TikTok API.
        """
        raise NotImplementedError(
            "TikTok Content Posting API does not support direct messages"
        )

    # =========================================================================
    # TikTok-specific operations
    # =========================================================================

    async def exchange_code_for_token(self, code: str) -> dict[str, Any]:
        """Exchange OAuth authorization code for access + refresh tokens.

        Args:
            code: Authorization code from the OAuth callback.

        Returns:
            dict[str, Any]: Token response with access_token and refresh_token.
        """
        payload = {
            "client_key": self._client_id,
            "client_secret": self._client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self._redirect_uri,
        }

        try:
            response = await self._client.post(TIKTOK_TOKEN_URL, json=payload)
            data: dict[str, Any] = response.json()

            if "error" in data and data["error"] != "ok":
                raise ProviderException(
                    f"TikTok token exchange failed: {data.get('error_description')}",
                    provider="tiktok",
                    operation="exchange_code",
                )

            return data

        except (RateLimitException, ProviderException):
            raise
        except httpx.RequestError as exc:
            raise ProviderException(
                f"TikTok token exchange connection error: {exc}",
                provider="tiktok",
                operation="exchange_code",
            ) from exc

    async def refresh_access_token(self, refresh_token: str) -> dict[str, Any]:
        """Refresh an expired access token.

        Args:
            refresh_token: Valid refresh token.

        Returns:
            dict[str, Any]: New token response.
        """
        payload = {
            "client_key": self._client_id,
            "client_secret": self._client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }

        response = await self._client.post(TIKTOK_TOKEN_URL, json=payload)
        return response.json()  # type: ignore[no-any-return]

    async def get_creator_info(self, access_token: str) -> dict[str, Any]:
        """Retrieve information about the authenticated creator.

        Args:
            access_token: User access token.

        Returns:
            dict[str, Any]: Creator profile data.
        """
        return await self._request("GET", "post/video/creator_info/", access_token=access_token)

    async def get_video_query(
        self, access_token: str, video_ids: list[str], fields: list[str] | None = None
    ) -> dict[str, Any]:
        """Retrieve information about specific videos.

        Args:
            access_token: User access token.
            video_ids: List of video IDs to query.
            fields: Optional list of fields to retrieve.

        Returns:
            dict[str, Any]: Video data from TikTok.
        """
        default_fields = [
            "id", "title", "video_description", "create_time",
            "cover_image_url", "share_url", "view_count", "like_count",
        ]
        payload = {
            "filters": {"video_ids": video_ids},
            "fields": fields or default_fields,
        }
        return await self._request(
            "POST", "video/query/", access_token=access_token, json=payload
        )

    async def list_videos(
        self, access_token: str, max_count: int = 20
    ) -> dict[str, Any]:
        """List videos for the authenticated creator.

        Args:
            access_token: User access token.
            max_count: Maximum number of videos to return.

        Returns:
            dict[str, Any]: Paginated video list.
        """
        payload = {"max_count": max_count}
        return await self._request(
            "POST", "video/list/", access_token=access_token, json=payload
        )

    async def close(self) -> None:
        """Close the underlying HTTP client connection pool."""
        await self._client.aclose()
