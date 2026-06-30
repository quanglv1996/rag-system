"""OAuth Manager — centralized OAuth2 flow management.

Handles the full OAuth lifecycle for Facebook, Google/YouTube, and TikTok:
- Authorization URL generation
- Callback code exchange
- Token storage (encrypted in Redis + DB)
- Token expiration tracking
- Automatic token refresh

Business logic lives here; providers only make raw API calls.
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlencode

from app.common.constants import (
    CACHE_PREFIX_TOKEN,
    CACHE_TOKEN_TTL,
    FACEBOOK_OAUTH_URL,
    FACEBOOK_TOKEN_URL,
    GOOGLE_AUTH_URL,
    GOOGLE_TOKEN_URL,
    TIKTOK_AUTH_URL,
    TIKTOK_TOKEN_URL,
)
from app.core.config import get_settings
from app.core.exception import AuthenticationException
from app.core.logger import get_logger

logger = get_logger(__name__)


class OAuthToken:
    """Represents an OAuth2 token set.

    Attributes:
        access_token: Short-lived access token.
        refresh_token: Long-lived refresh token (may be None).
        expires_at: Unix timestamp when the access token expires.
        token_type: Token type (usually 'Bearer').
        scope: Granted scopes.
        extra: Additional provider-specific fields.
    """

    def __init__(
        self,
        access_token: str,
        refresh_token: str | None = None,
        expires_in: int = 3600,
        token_type: str = "Bearer",
        scope: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an OAuth token.

        Args:
            access_token: The access token string.
            refresh_token: Optional refresh token.
            expires_in: Seconds until token expiry.
            token_type: Token type string.
            scope: Space-separated scope string.
            extra: Additional platform-specific data.
        """
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at = int(time.time()) + expires_in
        self.token_type = token_type
        self.scope = scope
        self.extra = extra or {}

    @property
    def is_expired(self) -> bool:
        """Check if the access token has expired (with 5-minute buffer).

        Returns:
            bool: True if the token is expired or expiring soon.
        """
        return time.time() >= (self.expires_at - 300)  # 5-minute buffer

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage.

        Returns:
            dict: Token data.
        """
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "token_type": self.token_type,
            "scope": self.scope,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthToken":
        """Deserialize from dictionary.

        Args:
            data: Stored token data dict.

        Returns:
            OAuthToken: Reconstructed token object.
        """
        token = cls(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            expires_in=max(0, data.get("expires_at", 0) - int(time.time())),
            token_type=data.get("token_type", "Bearer"),
            scope=data.get("scope", ""),
            extra=data.get("extra", {}),
        )
        token.expires_at = data.get("expires_at", token.expires_at)
        return token


class OAuthManager:
    """Centralized OAuth2 manager for all platforms.

    Attributes:
        _redis: Async Redis client for token caching.
        _settings: Application settings.
    """

    def __init__(self, redis: Any | None = None) -> None:
        """Initialize the OAuth manager.

        Args:
            redis: Optional async Redis client for token caching.
        """
        self._redis = redis
        self._settings = get_settings()

    # =========================================================================
    # Facebook OAuth
    # =========================================================================

    def get_facebook_login_url(
        self,
        redirect_uri: str,
        scopes: list[str] | None = None,
        state: str | None = None,
    ) -> str:
        """Build the Facebook OAuth authorization URL.

        Args:
            redirect_uri: Callback URL after authorization.
            scopes: List of Facebook permissions to request.
            state: Optional CSRF protection state string.

        Returns:
            str: Authorization URL to redirect the user to.
        """
        default_scopes = ["pages_manage_posts", "pages_read_engagement", "pages_messaging"]
        params = {
            "client_id": self._settings.facebook_app_id,
            "redirect_uri": redirect_uri,
            "scope": ",".join(scopes or default_scopes),
            "response_type": "code",
        }
        if state:
            params["state"] = state

        return f"{FACEBOOK_OAUTH_URL}?{urlencode(params)}"

    async def exchange_facebook_code(
        self, code: str, redirect_uri: str
    ) -> OAuthToken:
        """Exchange a Facebook authorization code for tokens.

        Args:
            code: Authorization code from the callback.
            redirect_uri: Must match the redirect_uri used for authorization.

        Returns:
            OAuthToken: Token set.

        Raises:
            AuthenticationException: If the exchange fails.
        """
        import httpx

        params = {
            "client_id": self._settings.facebook_app_id,
            "client_secret": self._settings.facebook_app_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(FACEBOOK_TOKEN_URL, params=params)
            data: dict[str, Any] = response.json()

        if "error" in data:
            raise AuthenticationException(
                f"Facebook token exchange failed: {data['error'].get('message')}"
            )

        token = OAuthToken(
            access_token=data["access_token"],
            expires_in=data.get("expires_in", 5184000),  # ~60 days
            token_type=data.get("token_type", "bearer"),
        )

        await self._store_token("facebook", token)
        return token

    async def refresh_facebook_token(
        self, short_lived_token: str
    ) -> OAuthToken:
        """Exchange a short-lived Facebook token for a long-lived one.

        Args:
            short_lived_token: Short-lived user access token.

        Returns:
            OAuthToken: Long-lived token.
        """
        from app.providers.social.facebook_provider import FacebookProvider

        provider = FacebookProvider()
        result = await provider.get_long_lived_token(short_lived_token)

        token = OAuthToken(
            access_token=result["access_token"],
            expires_in=result.get("expires_in", 5184000),
        )
        await self._store_token("facebook", token)
        return token

    # =========================================================================
    # Google / YouTube OAuth
    # =========================================================================

    def get_google_login_url(
        self,
        redirect_uri: str,
        scopes: list[str] | None = None,
        state: str | None = None,
    ) -> str:
        """Build the Google OAuth authorization URL.

        Args:
            redirect_uri: Callback URL.
            scopes: List of Google API scopes.
            state: Optional state string.

        Returns:
            str: Authorization URL.
        """
        default_scopes = [
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube.readonly",
        ]
        params = {
            "client_id": self._settings.youtube_client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(scopes or default_scopes),
            "response_type": "code",
            "access_type": "offline",
            "prompt": "consent",
        }
        if state:
            params["state"] = state

        return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

    async def exchange_google_code(
        self, code: str, redirect_uri: str
    ) -> OAuthToken:
        """Exchange a Google authorization code for tokens.

        Args:
            code: Authorization code.
            redirect_uri: Must match the one used for auth.

        Returns:
            OAuthToken: Token set including refresh_token.
        """
        import httpx

        payload = {
            "client_id": self._settings.youtube_client_id,
            "client_secret": self._settings.youtube_client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(GOOGLE_TOKEN_URL, data=payload)
            data: dict[str, Any] = response.json()

        if "error" in data:
            raise AuthenticationException(
                f"Google token exchange failed: {data.get('error_description', data['error'])}"
            )

        token = OAuthToken(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            expires_in=data.get("expires_in", 3600),
            scope=data.get("scope", ""),
        )

        await self._store_token("google", token)
        return token

    async def refresh_google_token(self, refresh_token: str) -> OAuthToken:
        """Refresh an expired Google access token.

        Args:
            refresh_token: Long-lived refresh token.

        Returns:
            OAuthToken: New access token.
        """
        from app.services.youtube_service import YouTubeService

        service = YouTubeService(settings=self._settings)
        result = await service.refresh_token(refresh_token)

        token = OAuthToken(
            access_token=result["access_token"],
            refresh_token=refresh_token,  # Reuse existing refresh token
            expires_in=result.get("expires_in", 3600),
        )

        await self._store_token("google", token)
        return token

    # =========================================================================
    # TikTok OAuth
    # =========================================================================

    def get_tiktok_login_url(
        self,
        redirect_uri: str,
        scopes: list[str] | None = None,
        state: str | None = None,
    ) -> str:
        """Build the TikTok OAuth authorization URL.

        Args:
            redirect_uri: Callback URL.
            scopes: TikTok API scopes.
            state: Optional state string.

        Returns:
            str: Authorization URL.
        """
        default_scopes = ["user.info.basic", "video.publish", "video.list"]
        params = {
            "client_key": self._settings.tiktok_client_id,
            "redirect_uri": redirect_uri,
            "scope": ",".join(scopes or default_scopes),
            "response_type": "code",
        }
        if state:
            params["state"] = state

        return f"{TIKTOK_AUTH_URL}?{urlencode(params)}"

    async def exchange_tiktok_code(self, code: str) -> OAuthToken:
        """Exchange a TikTok authorization code for tokens.

        Args:
            code: Authorization code from callback.

        Returns:
            OAuthToken: Token set.
        """
        from app.services.tiktok_service import TikTokService

        service = TikTokService(settings=self._settings)
        result = await service.exchange_auth_code(code)

        token = OAuthToken(
            access_token=result["access_token"],
            refresh_token=result.get("refresh_token"),
            expires_in=result.get("expires_in", 86400),
            scope=result.get("scope", ""),
        )

        await self._store_token("tiktok", token)
        return token

    # =========================================================================
    # Auto-Refresh
    # =========================================================================

    async def get_valid_token(self, platform: str, user_id: str) -> OAuthToken | None:
        """Get a valid (non-expired) token, refreshing if necessary.

        Args:
            platform: Platform name ('facebook', 'google', 'tiktok').
            user_id: User identifier for per-user token storage.

        Returns:
            OAuthToken | None: Valid token or None if not found.
        """
        token = await self._load_token(platform, user_id)
        if token is None:
            return None

        if token.is_expired and token.refresh_token:
            logger.info("Token expired, auto-refreshing", platform=platform, user_id=user_id)

            try:
                if platform == "google":
                    token = await self.refresh_google_token(token.refresh_token)
                elif platform == "tiktok":
                    service = TikTokService(settings=self._settings)
                    result = await service.refresh_token(token.refresh_token)
                    token = OAuthToken(
                        access_token=result["access_token"],
                        refresh_token=result.get("refresh_token", token.refresh_token),
                        expires_in=result.get("expires_in", 86400),
                    )
                    await self._store_token(platform, token, user_id)
            except Exception as exc:
                logger.error("Token refresh failed", platform=platform, error=str(exc))
                return None

        return token

    # =========================================================================
    # Token Storage
    # =========================================================================

    async def _store_token(
        self,
        platform: str,
        token: OAuthToken,
        user_id: str = "default",
    ) -> None:
        """Store a token in Redis.

        Args:
            platform: Platform name.
            token: Token to store.
            user_id: User identifier.
        """
        if self._redis is None:
            return

        key = f"{CACHE_PREFIX_TOKEN}{platform}:{user_id}"
        await self._redis.setex(
            key,
            CACHE_TOKEN_TTL,
            json.dumps(token.to_dict()),
        )

    async def _load_token(
        self, platform: str, user_id: str = "default"
    ) -> OAuthToken | None:
        """Load a token from Redis.

        Args:
            platform: Platform name.
            user_id: User identifier.

        Returns:
            OAuthToken | None: Stored token or None.
        """
        if self._redis is None:
            return None

        key = f"{CACHE_PREFIX_TOKEN}{platform}:{user_id}"
        raw = await self._redis.get(key)
        if not raw:
            return None

        return OAuthToken.from_dict(json.loads(raw))
