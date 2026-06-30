"""SQLAlchemy declarative base and mixin classes.

Defines the shared DeclarativeBase for all ORM models, plus reusable
mixins for timestamps, soft-deletion, and UUID primary keys.

Example:
    >>> from app.database.base import Base, TimestampMixin
    >>>
    >>> class User(Base, TimestampMixin):
    ...     __tablename__ = "users"
    ...     id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models.

    All application models must inherit from this class.
    Provides the metadata object shared across all models.
    """

    pass


class UUIDPrimaryKeyMixin:
    """Mixin that adds a UUID primary key column.

    Attributes:
        id: UUID primary key with server-side default generation.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
        index=True,
    )


class TimestampMixin:
    """Mixin that adds created_at and updated_at timestamp columns.

    Attributes:
        created_at: UTC timestamp of record creation.
        updated_at: UTC timestamp of last update, auto-updated on change.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    """Mixin that adds soft-deletion support.

    Attributes:
        deleted_at: Timestamp when the record was soft-deleted, None if active.
        is_deleted: Whether the record is soft-deleted.
    """

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    @property
    def is_deleted(self) -> bool:
        """Check if this record has been soft-deleted.

        Returns:
            bool: True if the record is soft-deleted.
        """
        return self.deleted_at is not None

    def soft_delete(self) -> None:
        """Mark this record as soft-deleted with the current UTC timestamp."""
        self.deleted_at = datetime.now(UTC)


class StringPrimaryKeyMixin:
    """Mixin that adds a string (VARCHAR) primary key column.

    Useful for models where the natural key is a string.
    """

    id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
        index=True,
    )
