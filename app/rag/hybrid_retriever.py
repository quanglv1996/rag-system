"""Hybrid Retriever — combines dense vector search with BM25 keyword search.

Implements Reciprocal Rank Fusion (RRF) to merge results from two
complementary retrieval strategies:
- Dense retrieval: semantic similarity via embeddings (high recall on meaning)
- Sparse retrieval: BM25 keyword matching (high precision on exact terms)

The combined ranking outperforms either method alone, especially for
queries with rare technical terms or exact phrase requirements.
"""

from __future__ import annotations

from typing import Any

from app.core.logger import get_logger
from app.interfaces.vector_database import VectorDatabase
from app.rag.embedding import EmbeddingService
from app.rag.retriever import RAGRetriever, RetrievedChunk

logger = get_logger(__name__)

# Reciprocal Rank Fusion constant (prevents over-weighting top results)
RRF_K = 60


class BM25Retriever:
    """In-memory BM25 keyword retriever.

    Requires rank-bm25 package. Falls back to simple TF-IDF-like scoring
    if rank-bm25 is not available.

    Attributes:
        _corpus: List of tokenized documents.
        _doc_ids: Corresponding document IDs.
        _contents: Original document contents.
    """

    def __init__(self) -> None:
        """Initialize BM25 retriever."""
        self._corpus: list[list[str]] = []
        self._doc_ids: list[str] = []
        self._contents: list[str] = []
        self._bm25: Any = None

    def index(
        self,
        documents: list[tuple[str, str]],
    ) -> None:
        """Build the BM25 index from (id, content) tuples.

        Args:
            documents: List of (document_id, text_content) tuples.
        """
        self._doc_ids = [d[0] for d in documents]
        self._contents = [d[1] for d in documents]
        self._corpus = [doc.lower().split() for doc in self._contents]

        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import]

            self._bm25 = BM25Okapi(self._corpus)
        except ImportError:
            logger.warning("rank-bm25 not installed; BM25 uses simple term-frequency fallback")
            self._bm25 = None

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Search using BM25 keyword matching.

        Args:
            query: Query string.
            top_k: Number of top results.

        Returns:
            list[RetrievedChunk]: Ranked results.
        """
        if not self._corpus:
            return []

        query_tokens = query.lower().split()

        if self._bm25 is not None:
            scores = self._bm25.get_scores(query_tokens)
        else:
            # Simple TF fallback
            scores = []
            for doc_tokens in self._corpus:
                score = sum(doc_tokens.count(t) for t in query_tokens)
                scores.append(float(score))

        import numpy as np

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []

        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    RetrievedChunk(
                        content=self._contents[idx],
                        score=float(scores[idx]),
                        chunk_id=self._doc_ids[idx],
                    )
                )

        return results


class HybridRetriever:
    """Hybrid retriever combining dense embeddings and BM25.

    Uses Reciprocal Rank Fusion to merge ranked lists from both
    retrievers into a single, reranked result list.

    Attributes:
        _dense_retriever: Embedding-based semantic retriever.
        _bm25: BM25 keyword retriever.
        _alpha: Weight for dense results (1-alpha for BM25).
    """

    def __init__(
        self,
        vector_db: VectorDatabase,
        embedding_service: EmbeddingService,
        bm25_corpus: list[tuple[str, str]] | None = None,
        top_k: int = 5,
        alpha: float = 0.7,  # Favor dense retrieval by default
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            vector_db: Dense vector database provider.
            embedding_service: Embedding generation service.
            bm25_corpus: Optional pre-built corpus for BM25 indexing.
            top_k: Number of final results.
            alpha: Dense retrieval weight (0.0 to 1.0).
        """
        self._dense_retriever = RAGRetriever(
            vector_db=vector_db,
            embedding_service=embedding_service,
            top_k=top_k * 2,  # Fetch more candidates for fusion
        )
        self._bm25 = BM25Retriever()
        self._top_k = top_k
        self._alpha = alpha

        if bm25_corpus:
            self._bm25.index(bm25_corpus)

    def update_bm25_corpus(self, documents: list[tuple[str, str]]) -> None:
        """Update the BM25 index with new documents.

        Args:
            documents: List of (doc_id, content) tuples.
        """
        self._bm25.index(documents)

    async def retrieve(
        self,
        query: str,
        collection: str = "default",
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform hybrid retrieval with RRF fusion.

        Args:
            query: Natural language query.
            collection: Vector store collection.
            filter_metadata: Optional metadata filter (applied to dense only).

        Returns:
            list[RetrievedChunk]: Fused and ranked results.
        """
        # Retrieve from both systems
        dense_results = await self._dense_retriever.retrieve(
            query,
            collection=collection,
            filter_metadata=filter_metadata,
        )
        sparse_results = self._bm25.search(query, top_k=self._top_k * 2)

        # Reciprocal Rank Fusion
        return self._rrf_merge(dense_results, sparse_results)

    def _rrf_merge(
        self,
        dense: list[RetrievedChunk],
        sparse: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Merge ranked lists using Reciprocal Rank Fusion.

        Args:
            dense: Dense retrieval results.
            sparse: BM25 retrieval results.

        Returns:
            list[RetrievedChunk]: Merged and re-ranked results.
        """
        # Build RRF score map: chunk_id → accumulated RRF score
        rrf_scores: dict[str, float] = {}
        all_chunks: dict[str, RetrievedChunk] = {}

        # Dense retrieval contribution (weighted by alpha)
        for rank, chunk in enumerate(dense):
            rrf_score = self._alpha / (RRF_K + rank + 1)
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + rrf_score
            all_chunks[chunk.chunk_id] = chunk

        # Sparse retrieval contribution (weighted by 1-alpha)
        sparse_weight = 1.0 - self._alpha
        for rank, chunk in enumerate(sparse):
            rrf_score = sparse_weight / (RRF_K + rank + 1)
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + rrf_score
            if chunk.chunk_id not in all_chunks:
                all_chunks[chunk.chunk_id] = chunk

        # Sort by RRF score and attach scores
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        results = []
        for chunk_id in sorted_ids[: self._top_k]:
            chunk = all_chunks[chunk_id]
            chunk.score = rrf_scores[chunk_id]
            results.append(chunk)

        return results


class CrossEncoderReranker:
    """Cross-encoder reranker for improved result precision.

    Uses a small cross-encoder model (sentence-transformers) to
    re-score retrieved chunks by computing query-document relevance
    jointly (rather than independently as in bi-encoder retrieval).

    This is computationally expensive — only apply to top-K candidates.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Initialize the cross-encoder.

        Args:
            model_name: HuggingFace cross-encoder model identifier.
        """
        self._model_name = model_name
        self._model: Any = None

    def _load_model(self) -> Any:
        """Lazily load the cross-encoder model.

        Returns:
            CrossEncoder: Loaded model instance.
        """
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore[import]

                self._model = CrossEncoder(self._model_name)
                logger.info("Cross-encoder loaded", model=self._model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc

        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks using the cross-encoder.

        Args:
            query: Original query string.
            chunks: Candidate chunks to rerank.
            top_k: Number of results after reranking.

        Returns:
            list[RetrievedChunk]: Reranked chunks with updated scores.
        """
        if not chunks:
            return []

        model = self._load_model()
        pairs = [(query, chunk.content) for chunk in chunks]
        scores = model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        reranked = sorted(chunks, key=lambda c: c.score or 0.0, reverse=True)
        return reranked[: top_k] if top_k else reranked
