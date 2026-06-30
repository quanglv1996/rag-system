"""Security utilities module.

Provides JWT token creation/verification, password hashing using bcrypt,
and API key validation. All cryptographic operations use industry-standard
algorithms and libraries.

Example:
    >>> from app.core.security import create_access_token, verify_password
    >>> token = create_access_token({"sub": "user123"})
    >>> is_valid = verify_password("plain_pass", hashed_pass)
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# bcrypt context for password hashing
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# =============================================================================
# Password Hashing
# =============================================================================


def hash_password(plain_password: str) -> str:
    """Hash a plain text password using bcrypt.

    Args:
        plain_password: The plain text password to hash.

    Returns:
        str: The bcrypt hashed password string.
    """
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain text password against a bcrypt hash.

    Args:
        plain_password: The plain text password to verify.
        hashed_password: The bcrypt hash to verify against.

    Returns:
        bool: True if the password matches, False otherwise.
    """
    return _pwd_context.verify(plain_password, hashed_password)


# =============================================================================
# JWT Tokens
# =============================================================================


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT access token.

    Args:
        data: Payload data to encode in the token.
        expires_delta: Optional custom expiry duration.
            Defaults to settings.access_token_expire_minutes.

    Returns:
        str: Encoded JWT access token string.
    """
    settings = get_settings()
    to_encode = data.copy()

    if expires_delta is not None:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(
            minutes=settings.access_token_expire_minutes
        )

    to_encode.update({"exp": expire, "type": "access"})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm,
    )
    return encoded_jwt


def create_refresh_token(data: dict[str, Any]) -> str:
    """Create a signed JWT refresh token with longer expiry.

    Args:
        data: Payload data to encode in the token.

    Returns:
        str: Encoded JWT refresh token string.
    """
    settings = get_settings()
    to_encode = data.copy()

    expire = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)
    to_encode.update({"exp": expire, "type": "refresh"})

    return jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm,
    )


def decode_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT token.

    Args:
        token: JWT token string to decode.

    Returns:
        dict[str, Any]: Decoded token payload.

    Raises:
        ValueError: If the token is invalid or expired.
    """
    settings = get_settings()
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
        return payload
    except JWTError as exc:
        logger.warning("JWT decode failed", error=str(exc))
        raise ValueError("Invalid or expired token") from exc


def extract_subject(token: str) -> str:
    """Extract the subject (user ID) from a JWT token.

    Args:
        token: JWT token string.

    Returns:
        str: The subject claim from the token.

    Raises:
        ValueError: If the token is invalid or missing 'sub' claim.
    """
    payload = decode_token(token)
    subject: str | None = payload.get("sub")
    if subject is None:
        raise ValueError("Token missing 'sub' claim")
    return subject
