"""Complete RAG pipeline orchestrator.

Wires together all RAG components:
    DocumentLoader → TextChunker → EmbeddingService → VectorDatabase
    Query → EmbeddingService → RAGRetriever → PromptBuilder → LLM → Answer

The pipeline is configurable at every stage via constructor parameters
or application settings. New components can be plugged in without
changing the pipeline logic.
"""

from dataclasses import dataclass, field
from typing import Any

from app.common.enums import ChunkStrategy
from app.core.config import get_settings
from app.core.exception import RAGException
from app.core.logger import TimingLogger, get_logger
from app.interfaces.ai_provider import AIProvider
from app.interfaces.vector_database import VectorDatabase, VectorDocument
from app.rag.chunking import TextChunk, TextChunker
from app.rag.document_loader import DocumentLoader, LoadedDocument
from app.rag.embedding import EmbeddingService
from app.rag.retriever import RAGRetriever, RetrievedChunk
from app.schemas.ai import ChatMessage, ChatRequest
from app.common.enums import MessageRole

logger = get_logger(__name__)


@dataclass
class RAGQueryResult:
    """Result of a RAG query.

    Attributes:
        answer: Generated answer from the LLM.
        sources: Retrieved chunks used to generate the answer.
        query: Original query string.
        model: LLM model used for generation.
        provider: AI provider used.
        retrieved_count: Number of chunks retrieved.
    """

    answer: str
    sources: list[RetrievedChunk]
    query: str
    model: str
    provider: str
    retrieved_count: int


@dataclass
class IndexResult:
    """Result of indexing a document into the RAG pipeline.

    Attributes:
        document_id: Identifier of the indexed document.
        chunk_count: Number of chunks created and indexed.
        collection: Vector store collection used.
    """

    document_id: str
    chunk_count: int
    collection: str
    chunk_ids: list[str] = field(default_factory=list)


class RAGPipeline:
    """Orchestrates the complete Retrieval-Augmented Generation pipeline.

    Provides two main operations:
    1. index() — Load, chunk, embed, and store documents.
    2. query() — Retrieve relevant chunks and generate an LLM answer.

    Attributes:
        _loader: Document loader for multiple file formats.
        _chunker: Text chunker with configurable strategy.
        _embedding_service: Cached embedding generation service.
        _vector_db: Vector database for storage and retrieval.
        _retriever: Semantic retriever over the vector store.
        _ai_provider: AI provider for LLM generation.
    """

    def __init__(
        self,
        ai_provider: AIProvider,
        vector_db: VectorDatabase,
        redis: Any | None = None,
        chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k: int | None = None,
        min_score: float = 0.0,
    ) -> None:
        """Initialize the RAG pipeline with all components.

        Args:
            ai_provider: AI provider for embeddings and generation.
            vector_db: Vector database for chunk storage and retrieval.
            redis: Optional Redis client for embedding caching.
            chunk_strategy: Text splitting strategy.
            chunk_size: Override for chunk size in characters.
            chunk_overlap: Override for chunk overlap in characters.
            top_k: Number of chunks to retrieve per query.
            min_score: Minimum similarity threshold for retrieval.
        """
        settings = get_settings()

        self._loader = DocumentLoader()
        self._chunker = TextChunker(
            strategy=chunk_strategy,
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
        )
        self._embedding_service = EmbeddingService(
            provider=ai_provider,
            redis=redis,
        )
        self._vector_db = vector_db
        self._retriever = RAGRetriever(
            vector_db=vector_db,
            embedding_service=self._embedding_service,
            top_k=top_k or settings.retriever_top_k,
            min_score=min_score,
        )
        self._ai_provider = ai_provider

        logger.info(
            "RAG pipeline initialized",
            ai_provider=ai_provider.provider_name,
            vector_db=vector_db.provider_name,
            chunk_strategy=chunk_strategy,
        )

    async def index(
        self,
        source: str,
        document_id: str | None = None,
        collection: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> IndexResult:
        """Load, chunk, embed, and index a document into the vector store.

        Args:
            source: File path to the document.
            document_id: Optional ID prefix for chunks. Auto-generated if None.
            collection: Vector store collection to index into.
            metadata: Optional metadata attached to all chunks.

        Returns:
            IndexResult: Summary of the indexing operation.

        Raises:
            RAGException: If any pipeline stage fails.
        """
        from app.common.utils import generate_uuid

        doc_id = document_id or generate_uuid()

        with TimingLogger(f"index:{doc_id}", logger):
            # Stage 1: Load document
            logger.info("Indexing document", document_id=doc_id, source=source)
            loaded: LoadedDocument = await self._loader.load(source)

            # Stage 2: Chunk text
            base_metadata: dict[str, Any] = {
                "document_id": doc_id,
                "source": source,
                "doc_type": loaded.doc_type.value,
                **(loaded.metadata or {}),
                **(metadata or {}),
            }
            chunks: list[TextChunk] = self._chunker.chunk(
                loaded.content, base_metadata
            )

            if not chunks:
                raise RAGException(
                    f"Document '{source}' produced no chunks after splitting",
                    stage="chunking",
                )

            # Stage 3: Embed all chunks
            texts = [c.content for c in chunks]
            embeddings = await self._embedding_service.embed_texts(texts)

            if len(embeddings) != len(chunks):
                raise RAGException(
                    "Embedding count mismatch with chunk count",
                    stage="embedding",
                )

            # Stage 4: Build VectorDocument objects
            vector_docs: list[VectorDocument] = []
            chunk_ids: list[str] = []

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{doc_id}-chunk-{i}"
                chunk_ids.append(chunk_id)
                vector_docs.append(
                    VectorDocument(
                        id=chunk_id,
                        content=chunk.content,
                        embedding=embedding,
                        metadata={
                            **chunk.metadata,
                            "chunk_index": chunk.index,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                        },
                    )
                )

            # Stage 5: Store in vector database
            await self._vector_db.add(vector_docs, collection=collection)

            logger.info(
                "Document indexed successfully",
                document_id=doc_id,
                chunk_count=len(chunks),
                collection=collection,
            )

            return IndexResult(
                document_id=doc_id,
                chunk_count=len(chunks),
                collection=collection,
                chunk_ids=chunk_ids,
            )

    async def index_bytes(
        self,
        content: bytes,
        filename: str,
        document_id: str | None = None,
        collection: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> IndexResult:
        """Load from bytes, chunk, embed, and index into the vector store.

        Args:
            content: Raw file bytes.
            filename: Original filename with extension.
            document_id: Optional chunk ID prefix.
            collection: Vector store collection.
            metadata: Optional metadata for all chunks.

        Returns:
            IndexResult: Indexing summary.
        """
        import tempfile
        from pathlib import Path

        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            return await self.index(
                source=tmp_path,
                document_id=document_id,
                collection=collection,
                metadata={**(metadata or {}), "original_filename": filename},
            )
        finally:
            import os
            os.unlink(tmp_path)

    async def query(
        self,
        question: str,
        collection: str = "default",
        system_prompt: str | None = None,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> RAGQueryResult:
        """Retrieve relevant chunks and generate an answer.

        Args:
            question: Natural language question to answer.
            collection: Vector store collection to retrieve from.
            system_prompt: Optional system instruction for the LLM.
            top_k: Number of chunks to retrieve.
            filter_metadata: Metadata filter for retrieval.
            model: LLM model override.
            temperature: LLM sampling temperature.
            max_tokens: Maximum LLM response tokens.

        Returns:
            RAGQueryResult: Generated answer with source chunks.

        Raises:
            RAGException: If retrieval or generation fails.
        """
        with TimingLogger(f"query:{question[:50]}", logger):
            # Stage 1: Retrieve relevant chunks
            retrieved_chunks = await self._retriever.retrieve(
                query=question,
                collection=collection,
                top_k=top_k,
                filter_metadata=filter_metadata,
            )

            if not retrieved_chunks:
                logger.warning(
                    "No relevant chunks found for query",
                    question=question[:100],
                    collection=collection,
                )

            # Stage 2: Build the augmented prompt
            context = self._build_context(retrieved_chunks)
            prompt = self._build_prompt(question, context)

            default_system = (
                "You are a helpful AI assistant. Answer the question based only on "
                "the provided context. If the context does not contain enough information, "
                "say so clearly. Do not make up information."
            )

            # Stage 3: Generate answer via LLM
            try:
                messages = []
                if system_prompt or default_system:
                    messages.append(
                        ChatMessage(
                            role=MessageRole.SYSTEM,
                            content=system_prompt or default_system,
                        )
                    )
                messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

                request = ChatRequest(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response = await self._ai_provider.chat(request)

            except Exception as exc:
                raise RAGException(
                    f"LLM generation failed: {exc}",
                    stage="generation",
                ) from exc

            logger.info(
                "RAG query completed",
                question=question[:100],
                retrieved_chunks=len(retrieved_chunks),
                answer_length=len(response.content),
            )

            return RAGQueryResult(
                answer=response.content,
                sources=retrieved_chunks,
                query=question,
                model=response.model,
                provider=response.provider,
                retrieved_count=len(retrieved_chunks),
            )

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a context block for the prompt.

        Args:
            chunks: Retrieved and scored document chunks.

        Returns:
            str: Formatted context string.
        """
        if not chunks:
            return "No relevant context found."

        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "Unknown")
            score = f"{chunk.score:.3f}"
            parts.append(
                f"[{i}] Source: {source} (similarity: {score})\n{chunk.content}"
            )

        return "\n\n".join(parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the final RAG prompt combining context and question.

        Args:
            question: User's question.
            context: Formatted context from retrieved chunks.

        Returns:
            str: Complete prompt string for the LLM.
        """
        return (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

    async def delete_document(
        self,
        document_id: str,
        chunk_ids: list[str],
        collection: str = "default",
    ) -> int:
        """Delete all chunks belonging to a document.

        Args:
            document_id: Document identifier.
            chunk_ids: List of chunk IDs to delete.
            collection: Vector store collection.

        Returns:
            int: Number of chunks deleted.
        """
        deleted = await self._vector_db.delete(chunk_ids, collection=collection)
        logger.info(
            "Document chunks deleted",
            document_id=document_id,
            deleted_count=deleted,
        )
        return deleted
