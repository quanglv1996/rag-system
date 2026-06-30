"""Document and DocumentChunk ORM models for RAG storage."""

import uuid

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin


class Document(Base, TimestampMixin):
    """Database model for uploaded documents.

    Attributes:
        id: UUID primary key.
        title: Human-readable document title.
        source: Original file path or URL.
        doc_type: Document type (pdf, docx, txt, md, html).
        status: Processing status (pending, processing, completed, failed).
        chunk_count: Number of chunks created from this document.
        metadata_json: JSON string of arbitrary document metadata.
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4, index=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source: Mapped[str] = mapped_column(String(1000), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", index=True
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationship to chunks
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return f"Document(id={self.id!r}, title={self.title!r})"


class DocumentChunk(Base, TimestampMixin):
    """Database model for document chunks used in RAG retrieval.

    Attributes:
        id: UUID primary key.
        document_id: Foreign key to parent Document.
        content: The text content of this chunk.
        chunk_index: Position of this chunk in the document.
        start_char: Character offset in original document.
        end_char: End character offset in original document.
        vector_id: ID of the corresponding vector in the vector store.
        metadata_json: JSON string of chunk-level metadata.
    """

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4, index=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    vector_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationship back to parent document
    document: Mapped[Document] = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"DocumentChunk(id={self.id!r}, "
            f"document_id={self.document_id!r}, "
            f"index={self.chunk_index})"
        )
