"""Pydantic schemas for RAG API endpoints."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.common.enums import ChunkStrategy, DocumentStatus, DocumentType


class DocumentUploadResponse(BaseModel):
    """Response returned after uploading and indexing a document."""

    document_id: str
    filename: str
    chunk_count: int
    collection: str
    status: DocumentStatus = DocumentStatus.COMPLETED


class RAGQueryRequest(BaseModel):
    """Request body for querying the RAG pipeline."""

    question: str = Field(min_length=1, max_length=5000)
    collection: str = Field(default="default")
    top_k: int | None = Field(default=None, ge=1, le=50)
    system_prompt: str | None = Field(default=None)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    filter_metadata: dict[str, Any] | None = Field(default=None)


class SourceChunkSchema(BaseModel):
    """A source chunk included in a RAG query response."""

    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGQueryResponse(BaseModel):
    """Response from a RAG query."""

    answer: str
    question: str
    sources: list[SourceChunkSchema]
    retrieved_count: int
    model: str
    provider: str


class DeleteDocumentResponse(BaseModel):
    """Response after deleting a document from the vector store."""

    document_id: str
    deleted_chunks: int
    success: bool = True
