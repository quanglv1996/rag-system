"""Request logging middleware.

Logs every incoming request with method, path, status code, duration,
and injects a unique request ID into the logging context for tracing.
"""

import time
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logger import get_logger, set_request_id

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs each HTTP request with timing and trace ID.

    Injects a unique request_id into every log record within the
    request context using structlog's contextvars.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process the request, log it, and pass it to the next handler.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            Response: HTTP response.
        """
        # Generate and bind request ID
        request_id = str(uuid.uuid4())
        set_request_id(request_id)

        # Bind request metadata to the structlog context
        import structlog

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
        )

        start = time.perf_counter()

        logger.info("Request started")

        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "Request completed",
                status_code=response.status_code,
                elapsed_ms=round(elapsed_ms, 2),
            )

            # Expose request ID in response header for client-side tracing
            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "Request failed with unhandled exception",
                elapsed_ms=round(elapsed_ms, 2),
                error=str(exc),
            )
            raise
        finally:
            structlog.contextvars.clear_contextvars()
