"""Credential Manager — encrypted storage and rotation of secrets.

Manages API keys, OAuth tokens, and other secrets with:
- Fernet symmetric encryption before storage
- TTL-based expiration
- Rotation support (create new, revoke old)
- Validation hooks
- Zero hard-coded secrets

All secrets are stored encrypted in Redis. The encryption key
is derived from the application SECRET_KEY.
"""

from __future__ import annotations

import json
import time
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import get_settings
from app.core.exception import AuthenticationException, ValidationException
from app.core.logger import get_logger

logger = get_logger(__name__)

# Credential type registry
CREDENTIAL_TYPES = frozenset(
    {
        "api_key",
        "oauth_access_token",
        "oauth_refresh_token",
        "webhook_secret",
        "encryption_key",
        "service_account",
    }
)


class Credential:
    """Represents an encrypted credential entry.

    Attributes:
        key: Unique storage key for this credential.
        credential_type: Type identifier (see CREDENTIAL_TYPES).
        encrypted_value: Fernet-encrypted secret bytes.
        created_at: Unix creation timestamp.
        expires_at: Optional unix expiry timestamp.
        metadata: Arbitrary non-sensitive metadata.
    """

    def __init__(
        self,
        key: str,
        credential_type: str,
        encrypted_value: bytes,
        created_at: float | None = None,
        expires_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a credential entry."""
        self.key = key
        self.credential_type = credential_type
        self.encrypted_value = encrypted_value
        self.created_at = created_at or time.time()
        self.expires_at = expires_at
        self.metadata = metadata or {}

    @property
    def is_expired(self) -> bool:
        """Check if this credential has expired.

        Returns:
            bool: True if expired.
        """
        return self.expires_at is not None and time.time() >= self.expires_at

    def to_storage_dict(self) -> dict[str, Any]:
        """Serialize for Redis storage.

        Returns:
            dict: Serializable representation.
        """
        return {
            "key": self.key,
            "type": self.credential_type,
            "value": self.encrypted_value.decode("latin-1"),  # bytes → str for JSON
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_storage_dict(cls, data: dict[str, Any]) -> "Credential":
        """Deserialize from Redis storage dict.

        Args:
            data: Stored dict.

        Returns:
            Credential: Reconstructed credential.
        """
        return cls(
            key=data["key"],
            credential_type=data["type"],
            encrypted_value=data["value"].encode("latin-1"),
            created_at=data.get("created_at"),
            expires_at=data.get("expires_at"),
            metadata=data.get("metadata", {}),
        )


class CredentialManager:
    """Manages encrypted storage, retrieval, and rotation of credentials.

    Uses Fernet encryption (AES-128-CBC) derived from the application
    secret key. All values are encrypted before storage; only this
    class decrypts them.

    Attributes:
        _fernet: Fernet encryption instance.
        _redis: Async Redis client.
    """

    _KEY_PREFIX = "creds:"

    def __init__(self, redis: Any) -> None:
        """Initialize with a Redis client.

        Args:
            redis: Async Redis client for credential storage.
        """
        settings = get_settings()

        # Derive a Fernet key from the application secret
        import base64
        import hashlib

        raw = hashlib.sha256(settings.secret_key.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(raw)
        self._fernet = Fernet(fernet_key)
        self._redis = redis

    def _encrypt(self, plaintext: str) -> bytes:
        """Encrypt a plaintext secret.

        Args:
            plaintext: Secret string to encrypt.

        Returns:
            bytes: Fernet-encrypted ciphertext.
        """
        return self._fernet.encrypt(plaintext.encode("utf-8"))

    def _decrypt(self, ciphertext: bytes) -> str:
        """Decrypt a Fernet-encrypted ciphertext.

        Args:
            ciphertext: Encrypted bytes.

        Returns:
            str: Decrypted plaintext.

        Raises:
            AuthenticationException: If decryption fails (tampered data).
        """
        try:
            return self._fernet.decrypt(ciphertext).decode("utf-8")
        except InvalidToken as exc:
            raise AuthenticationException(
                "Credential decryption failed — data may be tampered"
            ) from exc

    async def store(
        self,
        key: str,
        value: str,
        credential_type: str = "api_key",
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Encrypt and store a credential.

        Args:
            key: Unique storage key (e.g., 'openai_api_key').
            value: Plaintext secret value.
            credential_type: One of CREDENTIAL_TYPES.
            ttl_seconds: Optional TTL in seconds.
            metadata: Optional non-sensitive metadata.

        Raises:
            ValidationException: If credential_type is not registered.
        """
        if credential_type not in CREDENTIAL_TYPES:
            raise ValidationException(
                f"Unknown credential type '{credential_type}'. Valid: {CREDENTIAL_TYPES}",
                field="credential_type",
            )

        encrypted = self._encrypt(value)
        expires_at = time.time() + ttl_seconds if ttl_seconds else None

        credential = Credential(
            key=key,
            credential_type=credential_type,
            encrypted_value=encrypted,
            expires_at=expires_at,
            metadata=metadata,
        )

        redis_key = f"{self._KEY_PREFIX}{key}"
        serialized = json.dumps(credential.to_storage_dict())

        if ttl_seconds:
            await self._redis.setex(redis_key, ttl_seconds, serialized)
        else:
            await self._redis.set(redis_key, serialized)

        logger.info("Credential stored", key=key, type=credential_type, has_ttl=ttl_seconds is not None)

    async def retrieve(self, key: str) -> str | None:
        """Retrieve and decrypt a credential.

        Args:
            key: Storage key.

        Returns:
            str | None: Decrypted plaintext value, or None if not found or expired.
        """
        redis_key = f"{self._KEY_PREFIX}{key}"
        raw = await self._redis.get(redis_key)

        if not raw:
            return None

        credential = Credential.from_storage_dict(json.loads(raw))

        if credential.is_expired:
            await self._redis.delete(redis_key)
            logger.warning("Retrieved credential is expired and was deleted", key=key)
            return None

        return self._decrypt(credential.encrypted_value)

    async def rotate(
        self,
        key: str,
        new_value: str,
        ttl_seconds: int | None = None,
    ) -> None:
        """Replace a credential with a new value (rotation).

        The old value is deleted before storing the new one.

        Args:
            key: Storage key to rotate.
            new_value: New plaintext secret value.
            ttl_seconds: Optional new TTL.
        """
        # Load existing to preserve metadata/type
        redis_key = f"{self._KEY_PREFIX}{key}"
        raw = await self._redis.get(redis_key)

        credential_type = "api_key"
        metadata: dict[str, Any] = {}

        if raw:
            old = Credential.from_storage_dict(json.loads(raw))
            credential_type = old.credential_type
            metadata = old.metadata

        await self.store(key, new_value, credential_type, ttl_seconds, metadata)
        logger.info("Credential rotated", key=key)

    async def delete(self, key: str) -> bool:
        """Delete a stored credential.

        Args:
            key: Storage key to delete.

        Returns:
            bool: True if a credential was deleted.
        """
        redis_key = f"{self._KEY_PREFIX}{key}"
        deleted = await self._redis.delete(redis_key)
        return bool(deleted)

    async def exists(self, key: str) -> bool:
        """Check if a credential exists and is not expired.

        Args:
            key: Storage key.

        Returns:
            bool: True if a valid credential exists.
        """
        value = await self.retrieve(key)
        return value is not None

    async def list_keys(self, pattern: str = "*") -> list[str]:
        """List stored credential keys matching a pattern.

        Args:
            pattern: Redis key pattern (e.g., 'openai*').

        Returns:
            list[str]: Matching keys (without the internal prefix).
        """
        redis_pattern = f"{self._KEY_PREFIX}{pattern}"
        raw_keys = await self._redis.keys(redis_pattern)
        prefix_len = len(self._KEY_PREFIX)
        return [k[prefix_len:] for k in raw_keys]
