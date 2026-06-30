"""FastAPI dependency injection providers.

Centralizes all dependency factories for FastAPI's Depends() system.
Services, database sessions, Redis clients, and provider instances are
all obtained through these functions — never instantiated directly in routes.

Example:
    >>> from fastapi import Depends
    >>> from app.core.dependency import get_ai_service
    >>>
    >>> @router.post("/chat")
    ... async def chat(
    ...     request: ChatRequest,
    ...     ai_service: AIService = Depends(get_ai_service),
    ... ) -> ChatResponse:
    ...     return await ai_service.chat(request)
"""

from collections.abc import AsyncGenerator
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.exception import AuthenticationException
from app.core.security import decode_token
from app.database.session import get_async_session

# Bearer token security scheme for Swagger UI
_bearer_scheme = HTTPBearer(auto_error=False)

# =============================================================================
# Settings
# =============================================================================


def get_settings_dependency() -> Settings:
    """Provide application settings as a FastAPI dependency.

    Returns:
        Settings: Application settings instance.
    """
    return get_settings()


# =============================================================================
# Database
# =============================================================================


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async database session per request.

    Yields:
        AsyncSession: SQLAlchemy async database session.

    Example:
        >>> async def endpoint(db: AsyncSession = Depends(get_db)):
        ...     result = await db.execute(select(User))
    """
    async for session in get_async_session():
        yield session


# =============================================================================
# Redis
# =============================================================================


async def get_redis() -> aioredis.Redis:  # type: ignore[type-arg]
    """Provide a Redis client instance.

    Returns:
        aioredis.Redis: Async Redis client.

    Raises:
        RuntimeError: If Redis connection fails.
    """
    settings = get_settings()
    client: aioredis.Redis = aioredis.from_url(  # type: ignore[type-arg]
        str(settings.redis_url),
        encoding="utf-8",
        decode_responses=True,
    )
    return client


# =============================================================================
# Authentication
# =============================================================================


async def get_current_user_id(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_bearer_scheme),
    ],
) -> str:
    """Extract and validate the current user ID from a Bearer JWT token.

    Args:
        credentials: HTTP Bearer credentials from the Authorization header.

    Returns:
        str: The authenticated user's subject (ID).

    Raises:
        HTTPException: 401 if credentials are missing or token is invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_token(credentials.credentials)
        user_id: str | None = payload.get("sub")
        if not user_id:
            raise AuthenticationException("Token missing subject claim")
        return user_id
    except (ValueError, AuthenticationException) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# =============================================================================
# AI Providers (lazy imports to avoid circular dependencies)
# =============================================================================


def get_ai_provider(
    settings: Annotated[Settings, Depends(get_settings_dependency)],
) -> "AIProvider":  # type: ignore[name-defined]  # noqa: F821
    """Provide an AI provider based on configuration.

    Uses the Factory pattern to return the correct provider
    (OpenAI or Google) based on the LLM_PROVIDER setting.

    Args:
        settings: Application settings.

    Returns:
        AIProvider: Configured AI provider instance.
    """
    from app.providers.ai.factory import AIProviderFactory

    return AIProviderFactory.create(settings.llm_provider)


def get_vector_provider(
    settings: Annotated[Settings, Depends(get_settings_dependency)],
) -> "VectorDatabase":  # type: ignore[name-defined]  # noqa: F821
    """Provide a vector database provider based on configuration.

    Args:
        settings: Application settings.

    Returns:
        VectorDatabase: Configured vector DB provider instance.
    """
    from app.providers.vector.factory import VectorProviderFactory

    return VectorProviderFactory.create(settings.vector_db)


# =============================================================================
# Services
# =============================================================================


def get_ai_service(
    provider: Annotated["AIProvider", Depends(get_ai_provider)],  # type: ignore[name-defined]  # noqa: F821
) -> "AIService":  # type: ignore[name-defined]  # noqa: F821
    """Provide an AIService instance with injected AI provider.

    Args:
        provider: AI provider resolved by get_ai_provider.

    Returns:
        AIService: Service instance ready for use.
    """
    from app.services.ai_service import AIService

    return AIService(provider=provider)


def get_rag_service(
    settings: Annotated[Settings, Depends(get_settings_dependency)],
    vector_provider: Annotated["VectorDatabase", Depends(get_vector_provider)],  # type: ignore[name-defined]  # noqa: F821
    ai_provider: Annotated["AIProvider", Depends(get_ai_provider)],  # type: ignore[name-defined]  # noqa: F821
) -> "RAGService":  # type: ignore[name-defined]  # noqa: F821
    """Provide a RAGService instance with all required dependencies.

    Args:
        settings: Application settings.
        vector_provider: Vector database provider.
        ai_provider: AI provider for embeddings and generation.

    Returns:
        RAGService: Service instance ready for use.
    """
    from app.services.rag_service import RAGService

    return RAGService(
        settings=settings,
        vector_provider=vector_provider,
        ai_provider=ai_provider,
    )


def get_facebook_service(
    settings: Annotated[Settings, Depends(get_settings_dependency)],
) -> "FacebookService":  # type: ignore[name-defined]  # noqa: F821
    """Provide a FacebookService instance.

    Args:
        settings: Application settings.

    Returns:
        FacebookService: Service instance ready for use.
    """
    from app.services.facebook_service import FacebookService

    return FacebookService(settings=settings)


def get_youtube_service(
    settings: Annotated[Settings, Depends(get_settings_dependency)],
) -> "YouTubeService":  # type: ignore[name-defined]  # noqa: F821
    """Provide a YouTubeService instance.

    Args:
        settings: Application settings.

    Returns:
        YouTubeService: Service instance ready for use.
    """
    from app.services.youtube_service import YouTubeService

    return YouTubeService(settings=settings)


def get_telegram_service(
    settings: Annotated[Settings, Depends(get_settings_dependency)],
) -> "TelegramService":  # type: ignore[name-defined]  # noqa: F821
    """Provide a TelegramService instance.

    Args:
        settings: Application settings.

    Returns:
        TelegramService: Service instance ready for use.
    """
    from app.services.telegram_service import TelegramService

    return TelegramService(settings=settings)


def get_tiktok_service(
    settings: Annotated[Settings, Depends(get_settings_dependency)],
) -> "TikTokService":  # type: ignore[name-defined]  # noqa: F821
    """Provide a TikTokService instance.

    Args:
        settings: Application settings.

    Returns:
        TikTokService: Service instance ready for use.
    """
    from app.services.tiktok_service import TikTokService

    return TikTokService(settings=settings)


# =============================================================================
# Convenience type aliases for route annotations
# =============================================================================

DBSession = Annotated[AsyncSession, Depends(get_db)]
RedisClient = Annotated[aioredis.Redis, Depends(get_redis)]  # type: ignore[type-arg]
CurrentUserId = Annotated[str, Depends(get_current_user_id)]
AppSettings = Annotated[Settings, Depends(get_settings_dependency)]
