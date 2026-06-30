"""Async SQLAlchemy session management.

Provides async engine and session factory configured from application
settings. Uses connection pooling with sensible production defaults.

Example:
    >>> async for session in get_async_session():
    ...     result = await session.execute(select(User))
    ...     users = result.scalars().all()
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# Module-level engine and session factory (singletons per process)
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create the async SQLAlchemy engine.

    The engine is created once and reused across requests. Configuration
    is read from application settings.

    Returns:
        AsyncEngine: The configured async database engine.
    """
    global _engine

    if _engine is None:
        settings = get_settings()

        logger.info(
            "Creating database engine",
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
        )

        _engine = create_async_engine(
            settings.database_url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_timeout=settings.database_pool_timeout,
            pool_pre_ping=True,  # Verify connections before use
            echo=settings.database_echo,
            future=True,
        )

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory.

    Returns:
        async_sessionmaker: Configured session factory.
    """
    global _session_factory

    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Avoid lazy loading after commit
            autocommit=False,
            autoflush=False,
        )

    return _session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Async generator that yields a database session per request.

    Handles commit on success and rollback on exception.
    Used as a FastAPI dependency via Depends(get_async_session).

    Yields:
        AsyncSession: A scoped async database session.

    Example:
        >>> @router.get("/users")
        ... async def get_users(db: AsyncSession = Depends(get_async_session)):
        ...     result = await db.execute(select(User))
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Dispose the database engine, closing all pooled connections.

    Should be called during application shutdown to clean up resources.
    """
    global _engine, _session_factory

    if _engine is not None:
        logger.info("Disposing database engine")
        await _engine.dispose()
        _engine = None
        _session_factory = None
