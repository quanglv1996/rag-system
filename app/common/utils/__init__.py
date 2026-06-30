"""Common utility functions.

General-purpose helpers used across all modules. Each utility is a
standalone function with no side effects, making them easy to test.
"""

import hashlib
import json
import re
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, TypeVar

T = TypeVar("T")


# =============================================================================
# String Utilities
# =============================================================================


def slugify(text: str) -> str:
    """Convert a string to a URL-safe slug.

    Args:
        text: Input string to slugify.

    Returns:
        str: Lowercase, hyphen-separated slug.

    Example:
        >>> slugify("Hello World! 123")
        'hello-world-123'
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate a string to a maximum length with a suffix.

    Args:
        text: Input string to truncate.
        max_length: Maximum length including the suffix.
        suffix: String to append when truncated.

    Returns:
        str: Truncated string or original if within limit.

    Example:
        >>> truncate("This is a long sentence", max_length=15)
        'This is a lo...'
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """Clean text by removing excessive whitespace and control characters.

    Args:
        text: Raw text to clean.

    Returns:
        str: Cleaned, normalized text.
    """
    # Remove control characters except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =============================================================================
# Hashing Utilities
# =============================================================================


def compute_md5(content: str | bytes) -> str:
    """Compute the MD5 hash of a string or bytes object.

    Args:
        content: Content to hash.

    Returns:
        str: Hex-encoded MD5 hash string.

    Note:
        MD5 is used only for content deduplication, not for security.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.md5(content, usedforsecurity=False).hexdigest()


def compute_sha256(content: str | bytes) -> str:
    """Compute the SHA-256 hash of a string or bytes object.

    Args:
        content: Content to hash.

    Returns:
        str: Hex-encoded SHA-256 hash string.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


# =============================================================================
# ID Utilities
# =============================================================================


def generate_uuid() -> str:
    """Generate a new UUID4 as a string.

    Returns:
        str: UUID4 string (e.g., '550e8400-e29b-41d4-a716-446655440000').
    """
    return str(uuid.uuid4())


def generate_short_id(length: int = 8) -> str:
    """Generate a short alphanumeric ID using UUID bytes.

    Args:
        length: Desired length of the short ID. Must be <= 32.

    Returns:
        str: Short alphanumeric ID.

    Example:
        >>> generate_short_id()
        'a3f9b2c1'
    """
    return uuid.uuid4().hex[:length]


# =============================================================================
# JSON Utilities
# =============================================================================


def safe_json_loads(json_str: str | None, default: Any = None) -> Any:
    """Safely parse a JSON string, returning a default on failure.

    Args:
        json_str: JSON string to parse.
        default: Value to return if parsing fails. Defaults to None.

    Returns:
        Parsed JSON value or the default.
    """
    if not json_str:
        return default
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """Safely serialize an object to JSON, returning a default on failure.

    Args:
        obj: Object to serialize.
        default: String to return if serialization fails. Defaults to "{}".

    Returns:
        JSON string representation.
    """
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return default


# =============================================================================
# Timing Utilities
# =============================================================================


def current_timestamp_ms() -> int:
    """Get current UTC timestamp in milliseconds.

    Returns:
        int: Current UTC time in milliseconds since epoch.
    """
    return int(time.time() * 1000)


# =============================================================================
# Chunking Utilities
# =============================================================================


def chunk_list(items: list[T], size: int) -> list[list[T]]:
    """Split a list into fixed-size chunks.

    Args:
        items: List to chunk.
        size: Maximum size of each chunk.

    Returns:
        list[list[T]]: List of chunks.

    Example:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [items[i : i + size] for i in range(0, len(items), size)]


# =============================================================================
# Async Utilities
# =============================================================================


@asynccontextmanager
async def timer() -> AsyncGenerator[dict[str, float], None]:
    """Async context manager that measures elapsed time.

    Yields:
        dict[str, float]: Dictionary that will contain "elapsed_ms" on exit.

    Example:
        >>> async with timer() as t:
        ...     await some_operation()
        >>> print(f"Took {t['elapsed_ms']:.1f}ms")
    """
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000


# =============================================================================
# Validation Utilities
# =============================================================================


def is_valid_url(url: str) -> bool:
    """Check if a string is a valid HTTP/HTTPS URL.

    Args:
        url: URL string to validate.

    Returns:
        bool: True if the URL appears valid.
    """
    pattern = re.compile(
        r"^https?://"
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
        r"localhost|"
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        r"(?::\d+)?"
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return bool(pattern.match(url))


def is_valid_email(email: str) -> bool:
    """Check if a string is a valid email address.

    Args:
        email: Email string to validate.

    Returns:
        bool: True if the email appears valid.
    """
    pattern = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
    return bool(pattern.match(email))
