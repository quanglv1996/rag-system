"""Common Pydantic schemas shared across multiple modules."""

from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field

T = TypeVar("T")


class BaseResponse(BaseModel):
    """Standard API response envelope.

    Wraps all API responses with a consistent structure including
    success status and optional metadata.

    Attributes:
        success: Whether the request succeeded.
        message: Optional human-readable status message.
    """

    success: bool = Field(default=True)
    message: str | None = Field(default=None)


class PaginatedResponse(BaseResponse, Generic[T]):
    """Paginated list response with metadata.

    Attributes:
        data: List of items for the current page.
        total: Total number of items across all pages.
        page: Current page number (1-indexed).
        page_size: Number of items per page.
        total_pages: Total number of pages.
    """

    data: list[T] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    total_pages: int = Field(default=0, ge=0)


class ErrorResponse(BaseModel):
    """Standard API error response.

    Attributes:
        error: Nested error details object.
    """

    class ErrorDetail(BaseModel):
        """Error detail nested schema.

        Attributes:
            code: Machine-readable error code.
            message: Human-readable error description.
            details: Optional additional context.
        """

        code: str
        message: str
        details: dict[str, Any] = Field(default_factory=dict)

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check endpoint response.

    Attributes:
        status: Overall health status (healthy/degraded/unhealthy).
        version: Application version.
        services: Map of service name to health status.
    """

    status: str
    version: str
    services: dict[str, str] = Field(default_factory=dict)


class PaginationParams(BaseModel):
    """Query parameters for paginated list endpoints.

    Attributes:
        page: Page number (1-indexed).
        page_size: Number of items per page.
    """

    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(
        default=20, ge=1, le=100, description="Items per page"
    )

    @property
    def offset(self) -> int:
        """Calculate the SQL OFFSET value.

        Returns:
            int: Zero-based offset for database queries.
        """
        return (self.page - 1) * self.page_size
