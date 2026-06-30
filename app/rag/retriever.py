"""RAG retriever module.

Implements semantic retrieval from the vector store by converting
the user query to an embedding and finding the most similar chunks.
Supports metadata filtering, score thresholds, and result re-ranking.
"""

from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.exception import RAGException
from app.core.logger import get_logger
from app.interfaces.vector_database import VectorDatabase, VectorDocument
from app.rag.embedding import EmbeddingService

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with similarity score.

    Attributes:
        content: Text content of the chunk.
        score: Cosine similarity score (0.0 to 1.0).
        chunk_id: Unique identifier of the chunk.
        metadata: Associated metadata from the vector store.
    """

    content: str
    score: float
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGRetriever:
    """Semantic retriever that queries the vector store.

    Converts queries to embeddings and retrieves the most semantically
    similar document chunks. Supports filtering and score thresholding.

    Attributes:
        _vector_db: Vector database provider for similarity search.
        _embedding_service: Service for converting queries to embeddings.
        _top_k: Default number of results to retrieve.
        _min_score: Minimum similarity threshold.
    """

    def __init__(
        self,
        vector_db: VectorDatabase,
        embedding_service: EmbeddingService,
        top_k: int | None = None,
        min_score: float = 0.0,
    ) -> None:
        """Initialize the retriever.

        Args:
            vector_db: Configured vector database provider.
            embedding_service: Service for generating query embeddings.
            top_k: Number of top results to return.
            min_score: Minimum cosine similarity (0.0 to 1.0).
        """
        settings = get_settings()
        self._vector_db = vector_db
        self._embedding_service = embedding_service
        self._top_k = top_k or settings.retriever_top_k
        self._min_score = min_score

    async def retrieve(
        self,
        query: str,
        collection: str = "default",
        top_k: int | None = None,
        min_score: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: Natural language query string.
            collection: Vector store collection to search.
            top_k: Override for the number of results.
            min_score: Override for the minimum similarity score.
            filter_metadata: Metadata filter applied before scoring.

        Returns:
            list[RetrievedChunk]: Retrieved chunks sorted by score (desc).

        Raises:
            RAGException: If embedding or retrieval fails.
        """
        if not query or not query.strip():
            raise RAGException("Query must not be empty", stage="retrieval")

        k = top_k or self._top_k
        threshold = min_score if min_score is not None else self._min_score

        logger.debug(
            "Retrieving chunks",
            collection=collection,
            top_k=k,
            min_score=threshold,
        )

        # Step 1: Embed the query
        try:
            query_embedding = await self._embedding_service.embed_single(query)
        except Exception as exc:
            raise RAGException(
                f"Failed to embed query: {exc}",
                stage="retrieval",
            ) from exc

        if not query_embedding:
            raise RAGException("Empty embedding returned for query", stage="retrieval")

        # Step 2: Search the vector store
        try:
            results: list[VectorDocument] = await self._vector_db.search(
                query_embedding=query_embedding,
                top_k=k,
                collection=collection,
                min_score=threshold,
                filter_metadata=filter_metadata,
            )
        except Exception as exc:
            raise RAGException(
                f"Vector store search failed: {exc}",
                stage="retrieval",
            ) from exc

        # Step 3: Convert to RetrievedChunk
        chunks = [
            RetrievedChunk(
                content=doc.content,
                score=doc.score or 0.0,
                chunk_id=doc.id,
                metadata=doc.metadata,
            )
            for doc in results
        ]

        logger.info(
            "Retrieval complete",
            query_preview=query[:100],
            retrieved_count=len(chunks),
            collection=collection,
        )

        return chunks

    async def retrieve_with_rerank(
        self,
        query: str,
        collection: str = "default",
        top_k: int | None = None,
        fetch_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks and re-rank by keyword overlap as a second pass.

        Fetches more candidates (fetch_k) then re-scores by a simple
        keyword matching heuristic to improve precision.

        Args:
            query: Natural language query string.
            collection: Vector store collection name.
            top_k: Final number of results to return.
            fetch_k: Initial retrieval count before re-ranking.

        Returns:
            list[RetrievedChunk]: Re-ranked chunks.
        """
        k = top_k or self._top_k
        fetch = fetch_k or min(k * 3, 50)

        candidates = await self.retrieve(query, collection, top_k=fetch)

        if not candidates:
            return []

        # Simple keyword-based re-ranking boost
        query_terms = set(query.lower().split())

        def rerank_score(chunk: RetrievedChunk) -> float:
            """Compute combined score with keyword boost."""
            content_lower = chunk.content.lower()
            keyword_hits = sum(1 for term in query_terms if term in content_lower)
            boost = keyword_hits / max(len(query_terms), 1) * 0.1
            return (chunk.score or 0.0) + boost

        candidates.sort(key=rerank_score, reverse=True)
        return candidates[:k]
