"""Generic async repository base class.

Provides CRUD operations over SQLAlchemy 2.x async sessions.
Repositories abstract all database interactions from services,
ensuring business logic never touches SQLAlchemy directly.
"""

from typing import Any, Generic, TypeVar
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exception import DatabaseException, NotFoundException
from app.core.logger import get_logger
from app.database.base import Base

logger = get_logger(__name__)

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Generic CRUD repository for SQLAlchemy ORM models.

    Provides type-safe async database operations for any model
    that inherits from Base.

    Type Parameters:
        ModelT: The SQLAlchemy model class.
    """

    def __init__(self, model: type[ModelT], session: AsyncSession) -> None:
        """Initialize the repository.

        Args:
            model: The SQLAlchemy model class to operate on.
            session: Active async database session.
        """
        self._model = model
        self._session = session

    async def get_by_id(self, record_id: UUID | str) -> ModelT | None:
        """Retrieve a record by its primary key.

        Args:
            record_id: Primary key value (UUID or string).

        Returns:
            ModelT | None: The record if found, None otherwise.
        """
        try:
            result = await self._session.get(self._model, record_id)
            return result
        except Exception as exc:
            raise DatabaseException(
                f"Failed to get {self._model.__name__} by id: {exc}"
            ) from exc

    async def get_by_id_or_raise(self, record_id: UUID | str) -> ModelT:
        """Retrieve a record by primary key or raise NotFoundException.

        Args:
            record_id: Primary key value.

        Returns:
            ModelT: The found record.

        Raises:
            NotFoundException: If the record does not exist.
        """
        record = await self.get_by_id(record_id)
        if record is None:
            raise NotFoundException(
                resource=self._model.__name__,
                resource_id=str(record_id),
            )
        return record

    async def get_all(
        self,
        offset: int = 0,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[ModelT], int]:
        """Retrieve a paginated list of records.

        Args:
            offset: Number of records to skip.
            limit: Maximum number of records to return.
            filters: Optional column filters as {column_name: value}.

        Returns:
            tuple[list[ModelT], int]: Page of records and total count.
        """
        try:
            query = select(self._model)
            count_query = select(func.count()).select_from(self._model)

            if filters:
                for column_name, value in filters.items():
                    col = getattr(self._model, column_name, None)
                    if col is not None:
                        query = query.where(col == value)
                        count_query = count_query.where(col == value)

            total_result = await self._session.execute(count_query)
            total = total_result.scalar() or 0

            result = await self._session.execute(query.offset(offset).limit(limit))
            records = list(result.scalars().all())

            return records, total

        except Exception as exc:
            raise DatabaseException(
                f"Failed to list {self._model.__name__}: {exc}"
            ) from exc

    async def create(self, data: dict[str, Any]) -> ModelT:
        """Create and persist a new record.

        Args:
            data: Dictionary of column values to set on the new record.

        Returns:
            ModelT: The newly created and flushed record.

        Raises:
            DatabaseException: If creation fails.
        """
        try:
            instance = self._model(**data)
            self._session.add(instance)
            await self._session.flush()
            await self._session.refresh(instance)
            return instance
        except Exception as exc:
            raise DatabaseException(
                f"Failed to create {self._model.__name__}: {exc}"
            ) from exc

    async def update(
        self, record_id: UUID | str, data: dict[str, Any]
    ) -> ModelT:
        """Update an existing record by its primary key.

        Args:
            record_id: Primary key value.
            data: Dictionary of fields to update.

        Returns:
            ModelT: The updated record.

        Raises:
            NotFoundException: If the record does not exist.
            DatabaseException: If the update fails.
        """
        record = await self.get_by_id_or_raise(record_id)

        try:
            for key, value in data.items():
                if hasattr(record, key):
                    setattr(record, key, value)

            self._session.add(record)
            await self._session.flush()
            await self._session.refresh(record)
            return record
        except Exception as exc:
            raise DatabaseException(
                f"Failed to update {self._model.__name__}: {exc}"
            ) from exc

    async def delete(self, record_id: UUID | str) -> None:
        """Permanently delete a record by its primary key.

        Args:
            record_id: Primary key value.

        Raises:
            NotFoundException: If the record does not exist.
            DatabaseException: If deletion fails.
        """
        record = await self.get_by_id_or_raise(record_id)

        try:
            await self._session.delete(record)
            await self._session.flush()
        except Exception as exc:
            raise DatabaseException(
                f"Failed to delete {self._model.__name__}: {exc}"
            ) from exc

    async def exists(self, record_id: UUID | str) -> bool:
        """Check whether a record exists by primary key.

        Args:
            record_id: Primary key value.

        Returns:
            bool: True if the record exists.
        """
        record = await self.get_by_id(record_id)
        return record is not None
