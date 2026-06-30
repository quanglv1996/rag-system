"""RAG background tasks — document indexing, querying."""

from __future__ import annotations

from typing import Any

from celery import Task

from app.workers.celery_app import celery_app


@celery_app.task(
    bind=True,
    name="app.workers.tasks.rag_tasks.index_document",
    max_retries=3,
    default_retry_delay=10,
    queue="rag",
)
def index_document(
    self: Task,
    file_bytes_b64: str,
    filename: str,
    document_id: str | None = None,
    collection: str = "default",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Index a document into the vector store in the background.

    Args:
        file_bytes_b64: Base64-encoded file content.
        filename: Original filename with extension.
        document_id: Optional document ID prefix.
        collection: Vector store collection.
        metadata: Optional metadata dict.

    Returns:
        dict: Contains 'document_id', 'chunk_count', 'collection'.
    """
    import asyncio
    import base64

    from app.core.config import get_settings
    from app.core.logger import get_logger
    from app.providers.ai.factory import AIProviderFactory
    from app.providers.vector.factory import VectorProviderFactory
    from app.services.rag_service import RAGService

    logger = get_logger(__name__)
    logger.info("RAG index task started", filename=filename, task_id=self.request.id)

    async def _run() -> dict[str, Any]:
        settings = get_settings()
        ai_provider = AIProviderFactory.create(settings.embedding_provider)
        vector_provider = VectorProviderFactory.create(settings.vector_db)
        service = RAGService(
            settings=settings,
            vector_provider=vector_provider,
            ai_provider=ai_provider,
        )

        file_bytes = base64.b64decode(file_bytes_b64)
        result = await service.index_document(
            file_bytes=file_bytes,
            filename=filename,
            document_id=document_id,
            collection=collection,
            metadata=metadata,
        )
        return {
            "document_id": result.document_id,
            "chunk_count": result.chunk_count,
            "collection": result.collection,
        }

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.error("RAG index task failed", error=str(exc), filename=filename)
        raise self.retry(exc=exc, countdown=10 * (2 ** self.request.retries))


@celery_app.task(
    bind=True,
    name="app.workers.tasks.rag_tasks.rag_query_task",
    max_retries=2,
    queue="rag",
)
def rag_query_task(
    self: Task,
    question: str,
    collection: str = "default",
    top_k: int | None = None,
) -> dict[str, Any]:
    """Run a RAG query in the background.

    Args:
        question: Natural language question.
        collection: Vector store collection.
        top_k: Number of chunks to retrieve.

    Returns:
        dict: Contains 'answer', 'sources_count'.
    """
    import asyncio

    async def _run() -> dict[str, Any]:
        from app.core.config import get_settings
        from app.providers.ai.factory import AIProviderFactory
        from app.providers.vector.factory import VectorProviderFactory
        from app.services.rag_service import RAGService

        settings = get_settings()
        ai_provider = AIProviderFactory.create(settings.llm_provider)
        vector_provider = VectorProviderFactory.create(settings.vector_db)
        service = RAGService(
            settings=settings,
            vector_provider=vector_provider,
            ai_provider=ai_provider,
        )
        result = await service.query(question=question, collection=collection, top_k=top_k)
        return {
            "answer": result.answer,
            "sources_count": result.retrieved_count,
            "model": result.model,
        }

    try:
        return asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc)
