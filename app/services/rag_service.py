"""RAG Service — business logic for document indexing and Q&A.

Coordinates between the RAG pipeline, database repositories, and
caching layer. Enforces document size limits, deduplication, and
access control at the service level.
"""

from typing import Any

from app.core.config import Settings
from app.core.exception import RAGException, ValidationException
from app.core.logger import get_logger
from app.interfaces.ai_provider import AIProvider
from app.interfaces.vector_database import VectorDatabase
from app.rag.pipeline import IndexResult, RAGPipeline, RAGQueryResult

logger = get_logger(__name__)

# Maximum file size for document upload: 50 MB
MAX_DOCUMENT_SIZE_BYTES = 50 * 1024 * 1024


class RAGService:
    """Service layer for RAG document management and question answering.

    Attributes:
        _pipeline: The RAG pipeline orchestrator.
        _settings: Application settings.
    """

    def __init__(
        self,
        settings: Settings,
        vector_provider: VectorDatabase,
        ai_provider: AIProvider,
        redis: Any | None = None,
    ) -> None:
        """Initialize the RAG service.

        Args:
            settings: Application settings.
            vector_provider: Vector database provider.
            ai_provider: AI provider for embeddings and generation.
            redis: Optional Redis client for embedding caching.
        """
        self._settings = settings
        self._pipeline = RAGPipeline(
            ai_provider=ai_provider,
            vector_db=vector_provider,
            redis=redis,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            top_k=settings.retriever_top_k,
        )
        logger.info("RAGService initialized")

    async def index_document(
        self,
        file_bytes: bytes,
        filename: str,
        document_id: str | None = None,
        collection: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> IndexResult:
        """Index a document uploaded as bytes.

        Validates file size, delegates to the RAG pipeline, and returns
        indexing results for persistence in the database.

        Args:
            file_bytes: Raw file content.
            filename: Original filename with extension.
            document_id: Optional ID prefix for chunks.
            collection: Vector store collection.
            metadata: Optional metadata for all chunks.

        Returns:
            IndexResult: Indexing summary.

        Raises:
            ValidationException: If the file is too large or format unsupported.
            RAGException: If pipeline processing fails.
        """
        # Validate file size
        if len(file_bytes) > MAX_DOCUMENT_SIZE_BYTES:
            raise ValidationException(
                f"File size {len(file_bytes) / 1024 / 1024:.1f} MB exceeds "
                f"the {MAX_DOCUMENT_SIZE_BYTES // 1024 // 1024} MB limit",
                field="file",
            )

        if not filename or "." not in filename:
            raise ValidationException(
                "Filename must include a valid extension",
                field="filename",
            )

        logger.info(
            "Starting document indexing",
            filename=filename,
            size_bytes=len(file_bytes),
            collection=collection,
        )

        return await self._pipeline.index_bytes(
            content=file_bytes,
            filename=filename,
            document_id=document_id,
            collection=collection,
            metadata=metadata,
        )

    async def index_file_path(
        self,
        path: str,
        document_id: str | None = None,
        collection: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> IndexResult:
        """Index a document from a file system path.

        Args:
            path: File system path to the document.
            document_id: Optional chunk ID prefix.
            collection: Vector store collection.
            metadata: Optional metadata.

        Returns:
            IndexResult: Indexing summary.
        """
        return await self._pipeline.index(
            source=path,
            document_id=document_id,
            collection=collection,
            metadata=metadata,
        )

    async def query(
        self,
        question: str,
        collection: str = "default",
        system_prompt: str | None = None,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float = 0.3,
    ) -> RAGQueryResult:
        """Answer a question using the RAG pipeline.

        Args:
            question: Natural language question.
            collection: Vector store collection to search.
            system_prompt: Optional LLM system instruction.
            top_k: Number of chunks to retrieve.
            filter_metadata: Optional retrieval filter.
            model: LLM model override.
            temperature: LLM sampling temperature.

        Returns:
            RAGQueryResult: Answer with source citations.

        Raises:
            ValidationException: If question is empty.
            RAGException: If retrieval or generation fails.
        """
        if not question or not question.strip():
            raise ValidationException("Question must not be empty", field="question")

        return await self._pipeline.query(
            question=question,
            collection=collection,
            system_prompt=system_prompt,
            top_k=top_k,
            filter_metadata=filter_metadata,
            model=model,
            temperature=temperature,
        )

    async def delete_document(
        self,
        document_id: str,
        chunk_ids: list[str],
        collection: str = "default",
    ) -> int:
        """Remove a document's chunks from the vector store.

        Args:
            document_id: Document identifier for logging.
            chunk_ids: List of chunk IDs to delete.
            collection: Vector store collection.

        Returns:
            int: Number of chunks deleted.
        """
        return await self._pipeline.delete_document(
            document_id=document_id,
            chunk_ids=chunk_ids,
            collection=collection,
        )
