"""Application lifecycle event handlers.

Handles startup and shutdown events:
- Startup: Initialize database engine, Redis, run Alembic migrations.
- Shutdown: Dispose database pool, close Redis connections.
"""

from app.core.logger import get_logger

logger = get_logger(__name__)


async def on_startup() -> None:
    """Execute all startup tasks when the FastAPI application starts.

    - Configures logging from settings.
    - Verifies database connectivity.
    - Verifies Redis connectivity.
    - Logs startup confirmation.
    """
    from app.core.config import get_settings
    from app.core.logger import setup_logging

    settings = get_settings()

    # Initialize logging first
    setup_logging(
        log_level=settings.log_level,
        log_format=settings.log_format,
        log_file=settings.log_file,
    )

    logger.info(
        "Application starting",
        name=settings.app_name,
        version=settings.app_version,
        env=settings.app_env,
    )

    # Verify database connectivity
    try:
        from app.database.session import get_engine
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        logger.info("Database connection verified")
    except Exception as exc:
        logger.error("Database connection failed on startup", error=str(exc))
        # Don't crash — let the health check surface the issue

    # Verify Redis connectivity
    try:
        import redis.asyncio as aioredis

        redis_client = aioredis.from_url(str(settings.redis_url), decode_responses=True)
        await redis_client.ping()
        await redis_client.aclose()
        logger.info("Redis connection verified")
    except Exception as exc:
        logger.error("Redis connection failed on startup", error=str(exc))

    logger.info("Application startup complete")


async def on_shutdown() -> None:
    """Execute cleanup tasks when the FastAPI application shuts down.

    - Disposes the database engine and connection pool.
    - Logs shutdown confirmation.
    """
    logger.info("Application shutting down")

    try:
        from app.database.session import dispose_engine
        await dispose_engine()
        logger.info("Database engine disposed")
    except Exception as exc:
        logger.error("Error disposing database engine", error=str(exc))

    logger.info("Application shutdown complete")
