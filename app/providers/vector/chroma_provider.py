"""ChromaDB vector database provider.

Implements the VectorDatabase interface using ChromaDB for persistent,
metadata-filtered vector storage. Suitable for production workloads
requiring metadata filtering and collection management.
"""

from typing import Any

from app.core.config import get_settings
from app.core.exception import ProviderException
from app.core.logger import get_logger
from app.interfaces.vector_database import VectorDatabase, VectorDocument

logger = get_logger(__name__)


class ChromaProvider(VectorDatabase):
    """ChromaDB vector database provider implementation.

    Connects to a running ChromaDB instance (local or remote).
    Uses cosine similarity for nearest-neighbor search.

    Attributes:
        _client: ChromaDB async HTTP client.
        _embedding_function: None (embeddings provided externally).
    """

    def __init__(self) -> None:
        """Initialize the ChromaDB provider from application settings."""
        try:
            import chromadb

            settings = get_settings()
            self._client = chromadb.AsyncHttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
            logger.info(
                "ChromaDB provider initialized",
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
        except ImportError as exc:
            raise ProviderException(
                "chromadb package is not installed. Run: pip install chromadb",
                provider="chroma",
                operation="init",
            ) from exc

    @property
    def provider_name(self) -> str:
        """Return provider identifier.

        Returns:
            str: 'chroma'
        """
        return "chroma"

    async def _get_or_create_collection(
        self, collection: str
    ) -> Any:
        """Get or create a ChromaDB collection.

        Args:
            collection: Collection name.

        Returns:
            chromadb.Collection: The collection instance.
        """
        return await self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(
        self,
        documents: list[VectorDocument],
        collection: str = "default",
    ) -> list[str]:
        """Add documents to a ChromaDB collection.

        Args:
            documents: List of VectorDocument objects with embeddings.
            collection: Collection name.

        Returns:
            list[str]: List of inserted document IDs.

        Raises:
            ProviderException: If the upsert fails.
        """
        if not documents:
            return []

        try:
            col = await self._get_or_create_collection(collection)

            ids = [doc.id for doc in documents]
            embeddings = [doc.embedding for doc in documents if doc.embedding]
            contents = [doc.content for doc in documents]
            metadatas = [
                {k: str(v) for k, v in doc.metadata.items()}  # ChromaDB requires string values
                for doc in documents
            ]

            await col.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
            )

            logger.debug(
                "Added documents to ChromaDB",
                collection=collection,
                count=len(ids),
            )

            return ids

        except Exception as exc:
            raise ProviderException(
                f"ChromaDB add failed: {exc}",
                provider="chroma",
                operation="add",
            ) from exc

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        collection: str = "default",
        min_score: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorDocument]:
        """Search for similar documents by embedding vector.

        Args:
            query_embedding: Query vector to search against.
            top_k: Maximum results to return.
            collection: Collection name.
            min_score: Minimum cosine similarity threshold.
            filter_metadata: Optional ChromaDB metadata filter dict.

        Returns:
            list[VectorDocument]: Matching documents sorted by score.

        Raises:
            ProviderException: If search fails.
        """
        try:
            col = await self._get_or_create_collection(collection)

            where: dict[str, Any] | None = None
            if filter_metadata:
                where = {k: {"$eq": v} for k, v in filter_metadata.items()}

            results = await col.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances", "embeddings"],
            )

            output: list[VectorDocument] = []

            ids = results["ids"][0] if results["ids"] else []
            docs = results["documents"][0] if results["documents"] else []
            metas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            for i, doc_id in enumerate(ids):
                # Convert cosine distance to similarity score (1 - distance)
                score = 1.0 - distances[i] if i < len(distances) else 0.0

                if score < min_score:
                    continue

                output.append(
                    VectorDocument(
                        id=doc_id,
                        content=docs[i] if i < len(docs) else "",
                        metadata=metas[i] if i < len(metas) else {},
                        score=score,
                    )
                )

            return sorted(output, key=lambda d: d.score or 0.0, reverse=True)

        except Exception as exc:
            raise ProviderException(
                f"ChromaDB search failed: {exc}",
                provider="chroma",
                operation="search",
            ) from exc

    async def delete(
        self,
        document_ids: list[str],
        collection: str = "default",
    ) -> int:
        """Delete documents by ID from a ChromaDB collection.

        Args:
            document_ids: IDs to delete.
            collection: Collection name.

        Returns:
            int: Number of documents deleted.

        Raises:
            ProviderException: If deletion fails.
        """
        if not document_ids:
            return 0

        try:
            col = await self._get_or_create_collection(collection)
            await col.delete(ids=document_ids)
            return len(document_ids)
        except Exception as exc:
            raise ProviderException(
                f"ChromaDB delete failed: {exc}",
                provider="chroma",
                operation="delete",
            ) from exc

    async def clear(self, collection: str = "default") -> None:
        """Delete and recreate a ChromaDB collection.

        Args:
            collection: Collection name to clear.

        Raises:
            ProviderException: If the operation fails.
        """
        try:
            await self._client.delete_collection(collection)
            await self._client.create_collection(
                name=collection,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Cleared ChromaDB collection", collection=collection)
        except Exception as exc:
            raise ProviderException(
                f"ChromaDB clear failed: {exc}",
                provider="chroma",
                operation="clear",
            ) from exc

    async def count(self, collection: str = "default") -> int:
        """Count documents in a ChromaDB collection.

        Args:
            collection: Collection name.

        Returns:
            int: Document count.
        """
        try:
            col = await self._get_or_create_collection(collection)
            return await col.count()
        except Exception:
            return 0
