"""FastAPI application entry point.

Configures and creates the FastAPI application with:
- Lifecycle handlers (startup/shutdown)
- Exception handlers mapping domain exceptions to HTTP responses
- Middleware (logging, rate limiting, CORS)
- API v1 router
- OpenAPI documentation
- Health check endpoint
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import api_router
from app.core.config import get_settings
from app.core.exception import BaseAppException
from app.events.handlers import on_shutdown, on_startup
from app.middleware.logging_middleware import RequestLoggingMiddleware
from app.middleware.rate_limit_middleware import RateLimitMiddleware


# =============================================================================
# Application Factory
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events.

    Args:
        app: FastAPI application instance.

    Yields:
        None: Suspends here while the application runs.
    """
    await on_startup()
    yield
    await on_shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Fully configured application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI Automation Platform — multi-provider AI, RAG pipeline, "
            "and social media integration."
        ),
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ==========================================================================
    # Exception Handlers
    # ==========================================================================

    @app.exception_handler(BaseAppException)
    async def app_exception_handler(
        request: Request, exc: BaseAppException
    ) -> JSONResponse:
        """Convert domain exceptions to structured JSON error responses.

        Args:
            request: Incoming request.
            exc: Domain exception.

        Returns:
            JSONResponse: Structured error response.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler for unexpected exceptions.

        Args:
            request: Incoming request.
            exc: Unhandled exception.

        Returns:
            JSONResponse: Generic 500 error response.
        """
        from app.core.logger import get_logger

        logger = get_logger(__name__)
        logger.exception("Unhandled exception", error=str(exc), path=request.url.path)

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "details": {},
                }
            },
        )

    # ==========================================================================
    # Middleware (applied in reverse order — last added = outermost)
    # ==========================================================================

    # CORS — must be before custom middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_hosts,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting
    app.add_middleware(RateLimitMiddleware)

    # Request logging (innermost — runs last)
    app.add_middleware(RequestLoggingMiddleware)

    # ==========================================================================
    # Routers
    # ==========================================================================

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    # ==========================================================================
    # Built-in endpoints
    # ==========================================================================

    @app.get(
        "/health",
        tags=["Health"],
        summary="Health check",
        description="Returns the application health status and dependency checks.",
    )
    async def health_check() -> dict[str, Any]:
        """Health check endpoint.

        Checks connectivity to database and Redis.

        Returns:
            dict: Health status map.
        """
        from app.database.session import get_engine
        import redis.asyncio as aioredis

        services: dict[str, str] = {}

        # Database check
        try:
            from sqlalchemy import text
            engine = get_engine()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            services["database"] = "healthy"
        except Exception:
            services["database"] = "unhealthy"

        # Redis check
        try:
            redis_client = aioredis.from_url(str(settings.redis_url), decode_responses=True)
            await redis_client.ping()
            await redis_client.aclose()
            services["redis"] = "healthy"
        except Exception:
            services["redis"] = "unhealthy"

        overall = "healthy" if all(v == "healthy" for v in services.values()) else "degraded"

        return {
            "status": overall,
            "version": settings.app_version,
            "environment": settings.app_env,
            "services": services,
        }

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        """Root endpoint redirects to docs."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
        }

    return app


# Create the global application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
