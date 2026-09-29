"""
Pinecone Vector Database Store Module.
Manages index verification/creation, batched upserts, namespace isolation,
deterministic vector ID management, and semantic retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from app.config import Settings, get_settings
from app.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class PineconeStoreError(Exception):
    """Exception raised for Pinecone vector store operations."""
    pass


class PineconeNotConfiguredError(PineconeStoreError):
    """Raised when PINECONE_API_KEY is missing or invalid."""
    pass


class PineconeStore:
    """Production vector store implementation for Pinecone."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.api_key = self.settings.PINECONE_API_KEY
        self.index_name = self.settings.PINECONE_INDEX_NAME
        self.namespace = self.settings.PINECONE_NAMESPACE
        self.cloud = self.settings.PINECONE_CLOUD
        self.region = self.settings.PINECONE_REGION
        self._client: Optional[Pinecone] = None
        self._index = None

    @property
    def client(self) -> Pinecone:
        """Lazy-initialize and return Pinecone client."""
        if not self.api_key:
            raise PineconeNotConfiguredError(
                "PINECONE_API_KEY is not set. Please add your Pinecone API key to the environment or .env file."
            )
        if self._client is None:
            self._client = Pinecone(api_key=self.api_key)
        return self._client

    def ensure_index(self, dimension: int, metric: str = "cosine") -> None:
        """
        Verify that the Pinecone index exists; create it with serverless spec if missing.
        Does NOT recreate the index if it already exists.
        """
        client = self.client
        existing_indexes = [idx.name for idx in client.list_indexes()]

        if self.index_name not in existing_indexes:
            logger.info(
                "Creating Pinecone serverless index '%s' (dim=%d, metric=%s, cloud=%s, region=%s)...",
                self.index_name, dimension, metric, self.cloud, self.region
            )
            try:
                client.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                )
                logger.info("Successfully initiated creation of index '%s'.", self.index_name)
            except Exception as e:
                raise PineconeStoreError(f"Failed to create Pinecone index '{self.index_name}': {e}") from e
        else:
            logger.info("Pinecone index '%s' already exists. Skipping creation.", self.index_name)

        self._index = client.Index(self.index_name)

    @property
    def index(self):
        """Return the initialized index object."""
        if self._index is None:
            self._index = self.client.Index(self.index_name)
        return self._index

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """
        Batch upsert chunks with rich metadata and deterministic IDs into Pinecone.

        Args:
            chunks: List of DocumentChunk instances.
            embeddings: List of embedding vectors matching chunks length.
            batch_size: Number of vectors per upsert network request.

        Returns:
            Total count of upserted vectors.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)}).")

        total_upserted = 0
        vectors: List[Dict[str, Any]] = []

        for chunk, embedding in zip(chunks, embeddings):
            vector_item = {
                "id": chunk.metadata.chunk_id,
                "values": embedding,
                "metadata": {
                    "text": chunk.text,
                    "source": chunk.metadata.source,
                    "document_name": chunk.metadata.document_name,
                    "page": chunk.metadata.page,
                    "chunk_id": chunk.metadata.chunk_id,
                    "char_count": chunk.metadata.char_count,
                },
            }
            vectors.append(vector_item)

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            try:
                self.index.upsert(vectors=batch, namespace=self.namespace)
                total_upserted += len(batch)
                logger.info("Upserted batch %d/%d (%d vectors)", i // batch_size + 1, (len(vectors) + batch_size - 1) // batch_size, len(batch))
            except Exception as e:
                raise PineconeStoreError(f"Error during batch upsert to Pinecone: {e}") from e

        logger.info("Successfully completed upsert of %d vectors to namespace '%s'", total_upserted, self.namespace)
        return total_upserted

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone index for top-k semantically similar chunks.

        Args:
            query_embedding: Dense vector for query.
            top_k: Number of matches to retrieve.
            namespace: Namespace to search (defaults to settings.PINECONE_NAMESPACE).

        Returns:
            List of standardized result dicts.
        """
        target_namespace = namespace or self.namespace
        try:
            res = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=target_namespace,
                include_metadata=True,
            )
        except Exception as e:
            raise PineconeStoreError(f"Error executing Pinecone query: {e}") from e

        results = []
        for match in res.get("matches", []):
            meta = match.get("metadata", {})
            results.append({
                "chunk_id": match.get("id"),
                "score": float(match.get("score", 0.0)),
                "text": meta.get("text", ""),
                "source": meta.get("source", self.settings.PDF_URL.split("/")[-1]),
                "page": int(meta.get("page", 0)),
                "document_name": meta.get("document_name", "Agentic AI eBook"),
            })

        return results
