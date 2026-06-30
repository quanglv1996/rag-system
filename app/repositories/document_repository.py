"""Document repository — data access for Document and DocumentChunk models."""

from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exception import DatabaseException
from app.database.models.document import Document, DocumentChunk
from app.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    """Repository for Document ORM model operations.

    Provides domain-specific queries in addition to generic CRUD.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with Document model.

        Args:
            session: Active async database session.
        """
        super().__init__(Document, session)

    async def get_by_status(
        self, status: str, limit: int = 100
    ) -> list[Document]:
        """Retrieve documents with a specific processing status.

        Args:
            status: Processing status to filter by.
            limit: Maximum number of results.

        Returns:
            list[Document]: Matching documents.
        """
        try:
            result = await self._session.execute(
                select(Document)
                .where(Document.status == status)
                .limit(limit)
                .order_by(Document.created_at)
            )
            return list(result.scalars().all())
        except Exception as exc:
            raise DatabaseException(f"Failed to get documents by status: {exc}") from exc

    async def update_status(
        self,
        document_id: UUID,
        status: str,
        chunk_count: int | None = None,
    ) -> None:
        """Update a document's processing status.

        Args:
            document_id: Document primary key.
            status: New status value.
            chunk_count: Optional chunk count to update.
        """
        try:
            values: dict = {"status": status}
            if chunk_count is not None:
                values["chunk_count"] = chunk_count

            await self._session.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(**values)
            )
            await self._session.flush()
        except Exception as exc:
            raise DatabaseException(f"Failed to update document status: {exc}") from exc


class DocumentChunkRepository(BaseRepository[DocumentChunk]):
    """Repository for DocumentChunk ORM model operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with DocumentChunk model.

        Args:
            session: Active async database session.
        """
        super().__init__(DocumentChunk, session)

    async def get_by_document(
        self, document_id: UUID
    ) -> list[DocumentChunk]:
        """Retrieve all chunks belonging to a document.

        Args:
            document_id: Parent document primary key.

        Returns:
            list[DocumentChunk]: Ordered list of chunks.
        """
        try:
            result = await self._session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
            )
            return list(result.scalars().all())
        except Exception as exc:
            raise DatabaseException(f"Failed to get chunks for document: {exc}") from exc

    async def get_vector_ids(self, document_id: UUID) -> list[str]:
        """Get vector store IDs for all chunks of a document.

        Args:
            document_id: Parent document primary key.

        Returns:
            list[str]: List of vector IDs (non-null only).
        """
        chunks = await self.get_by_document(document_id)
        return [c.vector_id for c in chunks if c.vector_id]
