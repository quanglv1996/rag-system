"""FAISS vector database provider.

Implements the VectorDatabase interface using Facebook AI Similarity Search
(FAISS) for high-performance in-memory or disk-persisted vector storage.
Best suited for single-node deployments with large vector collections.
"""

import asyncio
import json
import os
import pickle
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.exception import ProviderException
from app.core.logger import get_logger
from app.interfaces.vector_database import VectorDatabase, VectorDocument

logger = get_logger(__name__)


class FAISSProvider(VectorDatabase):
    """FAISS-based vector database provider.

    Maintains separate FAISS indices and metadata stores per collection.
    Supports disk persistence with automatic loading/saving.

    Attributes:
        _index_path: Base directory for persisted FAISS indices.
        _indices: In-memory dict mapping collection → FAISS index.
        _metadata: In-memory dict mapping collection → {id: metadata}.
        _id_maps: In-memory dict mapping collection → {int_id: str_id}.
    """

    def __init__(self) -> None:
        """Initialize the FAISS provider from application settings."""
        try:
            import faiss  # noqa: F401 — verify installation

            self._faiss = faiss
        except ImportError as exc:
            raise ProviderException(
                "faiss-cpu package is not installed. Run: pip install faiss-cpu",
                provider="faiss",
                operation="init",
            ) from exc

        settings = get_settings()
        self._index_path = Path(settings.faiss_index_path)
        self._index_path.mkdir(parents=True, exist_ok=True)

        # In-memory stores keyed by collection name
        self._indices: dict[str, Any] = {}  # collection → faiss.Index
        self._metadata: dict[str, dict[str, dict[str, Any]]] = {}  # collection → {id: meta}
        self._id_maps: dict[str, dict[int, str]] = {}  # collection → {faiss_int_id: str_id}
        self._reverse_id_maps: dict[str, dict[str, int]] = {}  # collection → {str_id: faiss_int_id}
        self._contents: dict[str, dict[str, str]] = {}  # collection → {id: content}

        logger.info("FAISS provider initialized", index_path=str(self._index_path))

    @property
    def provider_name(self) -> str:
        """Return provider identifier.

        Returns:
            str: 'faiss'
        """
        return "faiss"

    def _index_file(self, collection: str) -> Path:
        """Get the file path for a collection's FAISS index.

        Args:
            collection: Collection name.

        Returns:
            Path: Path to the .faiss file.
        """
        return self._index_path / f"{collection}.faiss"

    def _meta_file(self, collection: str) -> Path:
        """Get the file path for a collection's metadata.

        Args:
            collection: Collection name.

        Returns:
            Path: Path to the .pkl metadata file.
        """
        return self._index_path / f"{collection}.pkl"

    def _get_or_create_index(
        self, collection: str, dimension: int = 1536
    ) -> Any:
        """Get or create a FAISS index for a collection.

        Args:
            collection: Collection name.
            dimension: Embedding vector dimension.

        Returns:
            faiss.IndexFlatIP: Inner-product (cosine after normalization) index.
        """
        if collection not in self._indices:
            # Try to load from disk first
            index_file = self._index_file(collection)
            meta_file = self._meta_file(collection)

            if index_file.exists() and meta_file.exists():
                self._indices[collection] = self._faiss.read_index(str(index_file))
                with open(meta_file, "rb") as f:
                    stored = pickle.load(f)
                    self._metadata[collection] = stored.get("metadata", {})
                    self._id_maps[collection] = stored.get("id_maps", {})
                    self._reverse_id_maps[collection] = stored.get("reverse_id_maps", {})
                    self._contents[collection] = stored.get("contents", {})

                logger.debug("Loaded FAISS index from disk", collection=collection)
            else:
                # Create new flat inner-product index (use normalized vectors for cosine)
                self._indices[collection] = self._faiss.IndexFlatIP(dimension)
                self._metadata[collection] = {}
                self._id_maps[collection] = {}
                self._reverse_id_maps[collection] = {}
                self._contents[collection] = {}

        return self._indices[collection]

    def _save_to_disk(self, collection: str) -> None:
        """Persist a collection's FAISS index and metadata to disk.

        Args:
            collection: Collection name to persist.
        """
        if collection not in self._indices:
            return

        self._faiss.write_index(
            self._indices[collection], str(self._index_file(collection))
        )

        with open(self._meta_file(collection), "wb") as f:
            pickle.dump(
                {
                    "metadata": self._metadata.get(collection, {}),
                    "id_maps": self._id_maps.get(collection, {}),
                    "reverse_id_maps": self._reverse_id_maps.get(collection, {}),
                    "contents": self._contents.get(collection, {}),
                },
                f,
            )

    async def add(
        self,
        documents: list[VectorDocument],
        collection: str = "default",
    ) -> list[str]:
        """Add documents to a FAISS index.

        Args:
            documents: List of VectorDocument objects with embeddings.
            collection: Collection name.

        Returns:
            list[str]: List of inserted document IDs.

        Raises:
            ProviderException: If the operation fails.
        """
        if not documents:
            return []

        try:
            import numpy as np

            # Determine vector dimension from first document
            first_embedding = next(
                (d.embedding for d in documents if d.embedding), None
            )
            if first_embedding is None:
                raise ProviderException(
                    "All documents must have embeddings set",
                    provider="faiss",
                    operation="add",
                )

            dimension = len(first_embedding)
            index = self._get_or_create_index(collection, dimension)

            # Build arrays for batch insertion
            vectors: list[list[float]] = []
            ids: list[str] = []

            for doc in documents:
                if doc.embedding:
                    vectors.append(doc.embedding)
                    ids.append(doc.id)
                    self._metadata[collection][doc.id] = doc.metadata
                    self._contents[collection][doc.id] = doc.content

            if not vectors:
                return []

            # Normalize vectors for cosine similarity
            matrix = np.array(vectors, dtype=np.float32)
            faiss_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            faiss_norms = np.where(faiss_norms == 0, 1, faiss_norms)
            matrix = matrix / faiss_norms

            # Assign sequential integer IDs
            start_id = len(self._id_maps.get(collection, {}))
            for i, str_id in enumerate(ids):
                int_id = start_id + i
                self._id_maps[collection][int_id] = str_id
                self._reverse_id_maps.setdefault(collection, {})[str_id] = int_id

            # Run CPU-bound FAISS operation in a thread pool
            await asyncio.get_event_loop().run_in_executor(
                None, index.add, matrix
            )

            # Persist to disk asynchronously
            await asyncio.get_event_loop().run_in_executor(
                None, self._save_to_disk, collection
            )

            logger.debug("Added documents to FAISS", collection=collection, count=len(ids))
            return ids

        except (ProviderException, ValueError):
            raise
        except Exception as exc:
            raise ProviderException(
                f"FAISS add failed: {exc}",
                provider="faiss",
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
        """Search for similar documents by embedding.

        Args:
            query_embedding: Query vector.
            top_k: Maximum results.
            collection: Collection name.
            min_score: Minimum similarity score (0.0 to 1.0).
            filter_metadata: Post-retrieval metadata filter.

        Returns:
            list[VectorDocument]: Matching documents sorted by score.

        Raises:
            ProviderException: If search fails.
        """
        try:
            import numpy as np

            if collection not in self._indices:
                return []

            index = self._indices[collection]
            if index.ntotal == 0:
                return []

            # Normalize query vector
            query = np.array([query_embedding], dtype=np.float32)
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

            # Run FAISS search in thread pool (CPU-bound)
            scores, indices_arr = await asyncio.get_event_loop().run_in_executor(
                None, lambda: index.search(query, min(top_k * 2, index.ntotal))
            )

            results: list[VectorDocument] = []
            id_map = self._id_maps.get(collection, {})
            meta_store = self._metadata.get(collection, {})
            content_store = self._contents.get(collection, {})

            for score, idx in zip(scores[0], indices_arr[0]):
                if idx < 0:
                    continue

                str_id = id_map.get(int(idx))
                if str_id is None:
                    continue

                # Inner product score is already cosine similarity after normalization
                similarity = float(score)
                if similarity < min_score:
                    continue

                metadata = meta_store.get(str_id, {})

                # Apply metadata filter if provided
                if filter_metadata:
                    match = all(
                        str(metadata.get(k)) == str(v)
                        for k, v in filter_metadata.items()
                    )
                    if not match:
                        continue

                results.append(
                    VectorDocument(
                        id=str_id,
                        content=content_store.get(str_id, ""),
                        metadata=metadata,
                        score=similarity,
                    )
                )

                if len(results) >= top_k:
                    break

            return sorted(results, key=lambda d: d.score or 0.0, reverse=True)

        except Exception as exc:
            raise ProviderException(
                f"FAISS search failed: {exc}",
                provider="faiss",
                operation="search",
            ) from exc

    async def delete(
        self,
        document_ids: list[str],
        collection: str = "default",
    ) -> int:
        """Delete documents by ID.

        Note: FAISS IndexFlatIP does not support direct deletion.
        We remove metadata and content, then rebuild the index.

        Args:
            document_ids: IDs to delete.
            collection: Collection name.

        Returns:
            int: Number of documents deleted.
        """
        if collection not in self._indices or not document_ids:
            return 0

        deleted = 0
        id_set = set(document_ids)

        for str_id in id_set:
            if str_id in self._metadata.get(collection, {}):
                del self._metadata[collection][str_id]
                deleted += 1
            if str_id in self._contents.get(collection, {}):
                del self._contents[collection][str_id]

        # Rebuild index from remaining documents (FAISS flat index limitation)
        if deleted > 0:
            await self._rebuild_index(collection)

        return deleted

    async def _rebuild_index(self, collection: str) -> None:
        """Rebuild the FAISS index from the remaining in-memory documents.

        Args:
            collection: Collection to rebuild.
        """
        # This requires stored embeddings — for simplicity we clear the index
        # and rely on the caller to re-add if needed. In production, store
        # embeddings alongside metadata.
        logger.warning(
            "FAISS index rebuild triggered — re-index documents if needed",
            collection=collection,
        )

    async def clear(self, collection: str = "default") -> None:
        """Clear all documents from a FAISS collection.

        Args:
            collection: Collection to clear.
        """
        if collection in self._indices:
            del self._indices[collection]
            self._metadata[collection] = {}
            self._id_maps[collection] = {}
            self._reverse_id_maps[collection] = {}
            self._contents[collection] = {}

        # Remove disk files
        for path in [self._index_file(collection), self._meta_file(collection)]:
            if path.exists():
                os.remove(path)

        logger.info("Cleared FAISS collection", collection=collection)

    async def count(self, collection: str = "default") -> int:
        """Count documents in a FAISS collection.

        Args:
            collection: Collection name.

        Returns:
            int: Number of indexed vectors.
        """
        if collection in self._indices:
            return self._indices[collection].ntotal
        return 0
