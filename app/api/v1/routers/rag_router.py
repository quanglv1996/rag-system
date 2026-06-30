"""RAG API router — document upload, indexing, and Q&A endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.core.dependency import AppSettings, get_rag_service
from app.schemas.rag import (
    DeleteDocumentResponse,
    DocumentUploadResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    SourceChunkSchema,
)
from app.services.rag_service import RAGService

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload and index a document",
    description=(
        "Upload a document (PDF, DOCX, TXT, MD, HTML) and index it "
        "into the vector store for RAG retrieval."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    collection: str = Form(default="default", description="Vector store collection"),
    document_id: str | None = Form(default=None, description="Optional document ID"),
    rag_service: RAGService = Depends(get_rag_service),
) -> DocumentUploadResponse:
    """Upload and index a document.

    Args:
        file: Uploaded file object.
        collection: Target vector store collection.
        document_id: Optional document identifier.
        rag_service: Injected RAG service.

    Returns:
        DocumentUploadResponse: Indexing result summary.
    """
    file_bytes = await file.read()
    filename = file.filename or "uploaded_document"

    result = await rag_service.index_document(
        file_bytes=file_bytes,
        filename=filename,
        document_id=document_id,
        collection=collection,
    )

    return DocumentUploadResponse(
        document_id=result.document_id,
        filename=filename,
        chunk_count=result.chunk_count,
        collection=result.collection,
    )


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Query the RAG knowledge base",
    description="Ask a question and get an answer synthesized from indexed documents.",
)
async def query_rag(
    request: RAGQueryRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> RAGQueryResponse:
    """Query the RAG pipeline and generate an answer.

    Args:
        request: Query request with question and retrieval parameters.
        rag_service: Injected RAG service.

    Returns:
        RAGQueryResponse: Generated answer with source citations.
    """
    result = await rag_service.query(
        question=request.question,
        collection=request.collection,
        system_prompt=request.system_prompt,
        top_k=request.top_k,
        filter_metadata=request.filter_metadata,
        temperature=request.temperature,
    )

    sources = [
        SourceChunkSchema(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            score=chunk.score,
            metadata=chunk.metadata,
        )
        for chunk in result.sources
    ]

    return RAGQueryResponse(
        answer=result.answer,
        question=result.query,
        sources=sources,
        retrieved_count=result.retrieved_count,
        model=result.model,
        provider=result.provider,
    )


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document from the vector store",
    description="Remove all chunks belonging to a document from the vector store.",
)
async def delete_document(
    document_id: str,
    chunk_ids: list[str],
    collection: str = "default",
    rag_service: RAGService = Depends(get_rag_service),
) -> DeleteDocumentResponse:
    """Delete a document and all its indexed chunks.

    Args:
        document_id: Document identifier.
        chunk_ids: List of chunk IDs to delete.
        collection: Vector store collection.
        rag_service: Injected RAG service.

    Returns:
        DeleteDocumentResponse: Deletion summary.
    """
    deleted_count = await rag_service.delete_document(
        document_id=document_id,
        chunk_ids=chunk_ids,
        collection=collection,
    )

    return DeleteDocumentResponse(
        document_id=document_id,
        deleted_chunks=deleted_count,
    )
