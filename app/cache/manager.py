"""Unified cache manager with Redis backend.

Provides:
- get/set/delete with type-safe deserialization
- TTL management
- Namespace prefixing to avoid key collisions
- @cached decorator for async functions
- Cache invalidation by pattern
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from app.core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CacheManager:
    """Centralized cache manager using Redis.

    All keys are automatically prefixed with a namespace to prevent
    collisions between different subsystems.

    Attributes:
        _redis: Async Redis client.
        _namespace: Key prefix applied to all cache operations.
        _default_ttl: Default TTL in seconds.
    """

    def __init__(
        self,
        redis: Any,
        namespace: str = "cache",
        default_ttl: int = 3600,
    ) -> None:
        """Initialize the cache manager.

        Args:
            redis: Async Redis client instance.
            namespace: Key namespace prefix.
            default_ttl: Default cache TTL in seconds.
        """
        self._redis = redis
        self._namespace = namespace
        self._default_ttl = default_ttl

    def _make_key(self, key: str) -> str:
        """Prepend namespace to a cache key.

        Args:
            key: Raw cache key.

        Returns:
            str: Namespaced key.
        """
        return f"{self._namespace}:{key}"

    async def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from cache.

        Args:
            key: Cache key.
            default: Value to return on cache miss.

        Returns:
            Any: Cached value or default.
        """
        raw = await self._redis.get(self._make_key(key))
        if raw is None:
            return default

        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Store a value in cache.

        Args:
            key: Cache key.
            value: Value to cache (must be JSON-serializable).
            ttl: TTL in seconds (uses default if None).
        """
        cache_ttl = ttl if ttl is not None else self._default_ttl
        serialized = json.dumps(value, default=str)
        await self._redis.setex(self._make_key(key), cache_ttl, serialized)

    async def delete(self, key: str) -> bool:
        """Delete a cache entry.

        Args:
            key: Cache key.

        Returns:
            bool: True if the key existed and was deleted.
        """
        result = await self._redis.delete(self._make_key(key))
        return bool(result)

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache.

        Args:
            key: Cache key.

        Returns:
            bool: True if key exists.
        """
        return bool(await self._redis.exists(self._make_key(key)))

    async def invalidate_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern.

        Args:
            pattern: Glob pattern (e.g., 'user:*').

        Returns:
            int: Number of deleted keys.
        """
        full_pattern = self._make_key(pattern)
        keys = await self._redis.keys(full_pattern)

        if not keys:
            return 0

        count = await self._redis.delete(*keys)
        logger.debug("Cache invalidated", pattern=full_pattern, count=count)
        return int(count)

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Coroutine[Any, Any, Any]],
        ttl: int | None = None,
    ) -> Any:
        """Get a cached value, computing and caching it if absent.

        Args:
            key: Cache key.
            factory: Async callable that computes the value on cache miss.
            ttl: Optional custom TTL.

        Returns:
            Any: Cached or freshly computed value.
        """
        cached = await self.get(key)
        if cached is not None:
            return cached

        value = await factory()
        await self.set(key, value, ttl=ttl)
        return value


def cached(
    key_template: str,
    ttl: int = 3600,
    namespace: str = "fn",
) -> Callable:
    """Decorator for caching async function results in Redis.

    The cache key is built from key_template with the function's arguments
    substituted in (using Python format syntax).

    Example:
        >>> @cached("user:{user_id}", ttl=600)
        ... async def get_user(user_id: str, redis) -> dict:
        ...     return await db.query(user_id)

    Args:
        key_template: Cache key template with {arg_name} placeholders.
        ttl: Cache TTL in seconds.
        namespace: Key namespace prefix.

    Returns:
        Callable: Decorated function with caching.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Find redis in kwargs
            redis = kwargs.get("redis")
            if redis is None:
                # Try positional args (convention: redis is last positional)
                for arg in args:
                    if hasattr(arg, "get") and hasattr(arg, "set"):
                        redis = arg
                        break

            if redis is None:
                # No Redis available — call function directly
                return await func(*args, **kwargs)

            # Build cache key
            try:
                cache_key = key_template.format(**kwargs)
            except KeyError:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            manager = CacheManager(redis, namespace=namespace, default_ttl=ttl)

            cached_value = await manager.get(cache_key)
            if cached_value is not None:
                logger.debug("Cache hit", key=cache_key)
                return cached_value

            result = await func(*args, **kwargs)
            await manager.set(cache_key, result, ttl=ttl)
            logger.debug("Cache set", key=cache_key)
            return result

        return wrapper

    return decorator
