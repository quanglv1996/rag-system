"""Abstract interface for vector database providers.

Defines the contract for all vector store implementations (FAISS, ChromaDB,
Pinecone, Weaviate, etc.). The RAG pipeline depends on this interface only,
enabling transparent switching between backends via configuration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorDocument:
    """A document to be stored in or retrieved from the vector store.

    Attributes:
        id: Unique identifier for this document.
        content: Original text content of the chunk.
        embedding: Vector representation (may be None before indexing).
        metadata: Arbitrary key-value metadata.
        score: Similarity score from retrieval (None for indexed docs).
    """

    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


class VectorDatabase(ABC):
    """Abstract base class for all vector database implementations.

    Provides a uniform interface for:
    - Adding documents with their pre-computed embeddings.
    - Searching by embedding vector (ANN retrieval).
    - Deleting documents by ID.
    - Clearing the entire store.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the vector database provider name.

        Returns:
            str: Provider identifier (e.g., 'faiss', 'chroma').
        """
        ...

    @abstractmethod
    async def add(
        self,
        documents: list[VectorDocument],
        collection: str = "default",
    ) -> list[str]:
        """Add documents with pre-computed embeddings to the store.

        Args:
            documents: List of VectorDocument objects with embeddings set.
            collection: Optional collection/namespace name.

        Returns:
            list[str]: List of inserted document IDs.

        Raises:
            ProviderException: If the upsert operation fails.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        collection: str = "default",
        min_score: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorDocument]:
        """Find the top-k most similar documents by embedding.

        Args:
            query_embedding: Query vector to search against.
            top_k: Maximum number of results to return.
            collection: Optional collection/namespace name.
            min_score: Minimum similarity score threshold (0.0 to 1.0).
            filter_metadata: Optional metadata filters to apply.

        Returns:
            list[VectorDocument]: Matching documents sorted by score descending.

        Raises:
            ProviderException: If the search operation fails.
        """
        ...

    @abstractmethod
    async def delete(
        self,
        document_ids: list[str],
        collection: str = "default",
    ) -> int:
        """Delete documents by their IDs.

        Args:
            document_ids: List of document IDs to delete.
            collection: Optional collection/namespace name.

        Returns:
            int: Number of documents actually deleted.

        Raises:
            ProviderException: If the delete operation fails.
        """
        ...

    @abstractmethod
    async def clear(self, collection: str = "default") -> None:
        """Remove all documents from a collection.

        Args:
            collection: Collection to clear. Defaults to 'default'.

        Raises:
            ProviderException: If the clear operation fails.
        """
        ...

    @abstractmethod
    async def count(self, collection: str = "default") -> int:
        """Count the number of documents in a collection.

        Args:
            collection: Collection to count.

        Returns:
            int: Number of documents in the collection.
        """
        ...

    async def health_check(self) -> bool:
        """Check if the vector database is reachable and operational.

        Returns:
            bool: True if the store is healthy.
        """
        try:
            await self.count()
            return True
        except Exception:
            return False
