"""Custom exception hierarchy for the AI Automation Platform.

Provides a structured set of domain exceptions with HTTP status codes,
error codes, and contextual details. All exceptions extend BaseAppException
to ensure consistent error responses across the API.

Example:
    >>> from app.core.exception import AuthenticationException
    >>> raise AuthenticationException("Invalid credentials")
"""

from typing import Any


class BaseAppException(Exception):
    """Base exception for all application-specific errors.

    Attributes:
        message: Human-readable error message.
        error_code: Machine-readable error code string.
        status_code: HTTP status code for API responses.
        details: Optional additional context about the error.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize base application exception.

        Args:
            message: Human-readable error description.
            error_code: Machine-readable error code (SCREAMING_SNAKE_CASE).
            status_code: HTTP status code for the response.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the exception to a dictionary for API responses.

        Returns:
            dict[str, Any]: Dictionary representation of the error.
        """
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
            }
        }

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            str: Repr string with error_code and message.
        """
        return (
            f"{self.__class__.__name__}("
            f"error_code={self.error_code!r}, "
            f"message={self.message!r})"
        )


class AuthenticationException(BaseAppException):
    """Raised when authentication fails (invalid credentials, expired token)."""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize authentication exception.

        Args:
            message: Human-readable error description.
            details: Optional additional context.
        """
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_FAILED",
            status_code=401,
            details=details,
        )


class AuthorizationException(BaseAppException):
    """Raised when the user lacks permission to perform an action."""

    def __init__(
        self,
        message: str = "Permission denied",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize authorization exception.

        Args:
            message: Human-readable error description.
            details: Optional additional context.
        """
        super().__init__(
            message=message,
            error_code="PERMISSION_DENIED",
            status_code=403,
            details=details,
        )


class RateLimitException(BaseAppException):
    """Raised when an API rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize rate limit exception.

        Args:
            message: Human-readable error description.
            retry_after: Seconds until rate limit resets.
            details: Optional additional context.
        """
        extra = details or {}
        if retry_after is not None:
            extra["retry_after_seconds"] = retry_after

        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details=extra,
        )
        self.retry_after = retry_after


class APIException(BaseAppException):
    """Raised when an external API call fails."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        status_code: int = 502,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize API exception.

        Args:
            message: Human-readable error description.
            provider: Name of the external provider (e.g., "openai").
            status_code: HTTP status code for the response.
            details: Optional additional context.
        """
        extra = details or {}
        extra["provider"] = provider

        super().__init__(
            message=message,
            error_code="EXTERNAL_API_ERROR",
            status_code=status_code,
            details=extra,
        )
        self.provider = provider


class ValidationException(BaseAppException):
    """Raised when input data validation fails."""

    def __init__(
        self,
        message: str = "Validation failed",
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize validation exception.

        Args:
            message: Human-readable error description.
            field: Optional name of the field that failed validation.
            details: Optional additional context.
        """
        extra = details or {}
        if field:
            extra["field"] = field

        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
            details=extra,
        )


class RAGException(BaseAppException):
    """Raised when a RAG pipeline operation fails."""

    def __init__(
        self,
        message: str,
        stage: str = "unknown",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize RAG exception.

        Args:
            message: Human-readable error description.
            stage: Pipeline stage where failure occurred (e.g., "embedding").
            details: Optional additional context.
        """
        extra = details or {}
        extra["stage"] = stage

        super().__init__(
            message=message,
            error_code="RAG_PIPELINE_ERROR",
            status_code=500,
            details=extra,
        )


class ProviderException(BaseAppException):
    """Raised when an AI or social provider operation fails."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        operation: str = "unknown",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize provider exception.

        Args:
            message: Human-readable error description.
            provider: Provider name (e.g., "openai", "facebook").
            operation: Operation that failed (e.g., "chat", "post").
            details: Optional additional context.
        """
        extra = details or {}
        extra["provider"] = provider
        extra["operation"] = operation

        super().__init__(
            message=message,
            error_code="PROVIDER_ERROR",
            status_code=502,
            details=extra,
        )


class DatabaseException(BaseAppException):
    """Raised when a database operation fails."""

    def __init__(
        self,
        message: str = "Database operation failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize database exception.

        Args:
            message: Human-readable error description.
            details: Optional additional context.
        """
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details=details,
        )


class NotFoundException(BaseAppException):
    """Raised when a requested resource is not found."""

    def __init__(
        self,
        resource: str = "Resource",
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize not found exception.

        Args:
            resource: Type of resource that was not found.
            resource_id: Optional identifier of the missing resource.
            details: Optional additional context.
        """
        message = f"{resource} not found"
        if resource_id:
            message = f"{resource} with id '{resource_id}' not found"

        extra = details or {}
        extra["resource"] = resource
        if resource_id:
            extra["resource_id"] = resource_id

        super().__init__(
            message=message,
            error_code="NOT_FOUND",
            status_code=404,
            details=extra,
        )


class ConfigurationException(BaseAppException):
    """Raised when a required configuration value is missing or invalid."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize configuration exception.

        Args:
            message: Human-readable error description.
            config_key: Optional name of the missing/invalid config key.
            details: Optional additional context.
        """
        extra = details or {}
        if config_key:
            extra["config_key"] = config_key

        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            details=extra,
        )
