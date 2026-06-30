"""Rate limiting middleware using Redis sliding window algorithm.

Implements a token-bucket / sliding window rate limiter per client IP.
Configurable via RATE_LIMIT_REQUESTS and RATE_LIMIT_WINDOW settings.
"""

import time
from typing import Any

import redis.asyncio as aioredis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.common.constants import CACHE_PREFIX_RATE_LIMIT
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiter using Redis.

    Limits requests per client IP using the Redis INCR + TTL pattern.
    Exceeding the limit returns 429 Too Many Requests.

    Attributes:
        _redis: Async Redis client.
        _max_requests: Maximum requests per window.
        _window_seconds: Window duration in seconds.
    """

    def __init__(self, app: Any, redis_url: str | None = None) -> None:
        """Initialize the rate limiter.

        Args:
            app: The ASGI application.
            redis_url: Redis connection URL. Defaults to settings value.
        """
        super().__init__(app)
        settings = get_settings()
        self._max_requests = settings.rate_limit_requests
        self._window_seconds = settings.rate_limit_window

        url = redis_url or settings.redis_url
        self._redis: aioredis.Redis = aioredis.from_url(  # type: ignore[type-arg]
            str(url), encoding="utf-8", decode_responses=True
        )

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Apply rate limiting and forward the request.

        Skips rate limiting for health check endpoints.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            Response: HTTP response or 429 if rate limited.
        """
        # Skip health check endpoints
        if request.url.path in ("/health", "/metrics"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        key = f"{CACHE_PREFIX_RATE_LIMIT}{client_ip}"

        try:
            current = await self._redis.incr(key)
            if current == 1:
                await self._redis.expire(key, self._window_seconds)

            if current > self._max_requests:
                ttl = await self._redis.ttl(key)
                logger.warning(
                    "Rate limit exceeded",
                    client_ip=client_ip,
                    requests=current,
                    retry_after=ttl,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "code": "RATE_LIMIT_EXCEEDED",
                            "message": "Too many requests. Please retry later.",
                            "details": {"retry_after_seconds": max(ttl, 1)},
                        }
                    },
                    headers={"Retry-After": str(max(ttl, 1))},
                )

        except Exception as exc:
            # If Redis is unavailable, fail open (don't block requests)
            logger.error("Rate limiter Redis error", error=str(exc))

        return await call_next(request)
