"""Storage Layer — pluggable object storage interface.

Provides a unified interface for storing and retrieving binary objects
(images, videos, documents, AI outputs) across different backends:
- Local filesystem
- Amazon S3
- Google Cloud Storage
- MinIO (S3-compatible)

New backends can be added by implementing the BaseStorage ABC.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseStorage(ABC):
    """Abstract interface for object storage backends."""

    @abstractmethod
    async def upload(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload binary data to storage.

        Args:
            key: Object key (path/filename).
            data: Raw bytes to store.
            content_type: MIME type of the data.
            metadata: Optional key-value metadata.

        Returns:
            str: URL or path to the stored object.
        """
        ...

    @abstractmethod
    async def download(self, key: str) -> bytes:
        """Download an object from storage.

        Args:
            key: Object key to retrieve.

        Returns:
            bytes: Object content.

        Raises:
            FileNotFoundError: If the key does not exist.
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete an object from storage.

        Args:
            key: Object key to delete.

        Returns:
            bool: True if deleted successfully.
        """
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if an object exists.

        Args:
            key: Object key.

        Returns:
            bool: True if the object exists.
        """
        ...

    @abstractmethod
    def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Get a public or pre-signed URL for an object.

        Args:
            key: Object key.
            expires_in: URL expiry in seconds (for pre-signed URLs).

        Returns:
            str: Accessible URL.
        """
        ...


class LocalStorage(BaseStorage):
    """Local filesystem storage backend.

    Stores objects in a local directory. Suitable for development
    and single-server deployments.

    Attributes:
        _base_path: Root directory for file storage.
        _base_url: URL prefix for generating file URLs.
    """

    def __init__(
        self,
        base_path: str = "./data/storage",
        base_url: str = "http://localhost:8000/files",
    ) -> None:
        """Initialize local storage.

        Args:
            base_path: Root directory path.
            base_url: Base URL for generated file links.
        """
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._base_url = base_url.rstrip("/")

    async def upload(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Save data to a local file.

        Args:
            key: Relative file path.
            data: File content.
            content_type: MIME type (ignored for local storage).
            metadata: Optional metadata (ignored for local storage).

        Returns:
            str: Local URL to the file.
        """
        file_path = self._base_path / key
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)
        return f"{self._base_url}/{key}"

    async def download(self, key: str) -> bytes:
        """Read a local file.

        Args:
            key: Relative file path.

        Returns:
            bytes: File content.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        file_path = self._base_path / key
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {key}")
        return file_path.read_bytes()

    async def delete(self, key: str) -> bool:
        """Delete a local file.

        Args:
            key: Relative file path.

        Returns:
            bool: True if deleted.
        """
        file_path = self._base_path / key
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if a local file exists."""
        return (self._base_path / key).exists()

    def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Return the local URL for a file.

        Args:
            key: File key.
            expires_in: Not used for local storage.

        Returns:
            str: Local HTTP URL.
        """
        return f"{self._base_url}/{key}"


class S3Storage(BaseStorage):
    """Amazon S3 (or compatible) storage backend.

    Uses boto3 async via aioboto3. Supports any S3-compatible service
    (AWS S3, MinIO, Cloudflare R2, DigitalOcean Spaces).

    Attributes:
        _bucket: S3 bucket name.
        _client_kwargs: Boto3 client configuration.
    """

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        public_url_prefix: str | None = None,
    ) -> None:
        """Initialize S3 storage.

        Args:
            bucket: S3 bucket name.
            region: AWS region.
            endpoint_url: Custom endpoint for S3-compatible services.
            access_key: AWS access key ID.
            secret_key: AWS secret access key.
            public_url_prefix: Optional public URL prefix for non-signed URLs.
        """
        self._bucket = bucket
        self._region = region
        self._endpoint_url = endpoint_url
        self._public_url_prefix = public_url_prefix

        self._client_kwargs: dict[str, Any] = {
            "region_name": region,
        }
        if endpoint_url:
            self._client_kwargs["endpoint_url"] = endpoint_url
        if access_key and secret_key:
            self._client_kwargs["aws_access_key_id"] = access_key
            self._client_kwargs["aws_secret_access_key"] = secret_key

    def _get_client(self) -> Any:
        """Create a boto3 S3 client.

        Returns:
            boto3.client: S3 client instance.
        """
        import boto3  # type: ignore[import]

        return boto3.client("s3", **self._client_kwargs)

    async def upload(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload data to S3.

        Args:
            key: S3 object key.
            data: Binary data.
            content_type: Content MIME type.
            metadata: Optional S3 metadata.

        Returns:
            str: S3 object URL.
        """
        import io

        client = self._get_client()
        client.upload_fileobj(
            io.BytesIO(data),
            self._bucket,
            key,
            ExtraArgs={
                "ContentType": content_type,
                "Metadata": metadata or {},
            },
        )

        return self.get_url(key)

    async def download(self, key: str) -> bytes:
        """Download an S3 object.

        Args:
            key: S3 object key.

        Returns:
            bytes: Object content.
        """
        import io

        client = self._get_client()
        buffer = io.BytesIO()
        client.download_fileobj(self._bucket, key, buffer)
        return buffer.getvalue()

    async def delete(self, key: str) -> bool:
        """Delete an S3 object.

        Args:
            key: S3 object key.

        Returns:
            bool: True if deleted.
        """
        try:
            client = self._get_client()
            client.delete_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        """Check if an S3 object exists.

        Args:
            key: S3 object key.

        Returns:
            bool: True if exists.
        """
        try:
            client = self._get_client()
            client.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a pre-signed URL for an S3 object.

        Args:
            key: S3 object key.
            expires_in: URL validity duration in seconds.

        Returns:
            str: Pre-signed URL.
        """
        if self._public_url_prefix:
            return f"{self._public_url_prefix}/{key}"

        client = self._get_client()
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_in,
        )


class StorageFactory:
    """Factory for creating storage backend instances."""

    @staticmethod
    def create(backend: str = "local", **kwargs: Any) -> BaseStorage:
        """Create a storage backend.

        Args:
            backend: Backend type ('local', 's3', 'minio').
            **kwargs: Backend-specific configuration.

        Returns:
            BaseStorage: Configured storage instance.

        Raises:
            ValueError: If backend is not recognized.
        """
        if backend == "local":
            return LocalStorage(
                base_path=kwargs.get("base_path", "./data/storage"),
                base_url=kwargs.get("base_url", "http://localhost:8000/files"),
            )
        elif backend in ("s3", "minio"):
            return S3Storage(
                bucket=kwargs["bucket"],
                region=kwargs.get("region", "us-east-1"),
                endpoint_url=kwargs.get("endpoint_url"),
                access_key=kwargs.get("access_key"),
                secret_key=kwargs.get("secret_key"),
                public_url_prefix=kwargs.get("public_url_prefix"),
            )
        else:
            raise ValueError(f"Unknown storage backend: '{backend}'")
