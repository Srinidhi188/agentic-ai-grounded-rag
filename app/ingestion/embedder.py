"""
Modular Text Embedding Module.
Provides uniform embedding interfaces for OpenAI, Google Gemini, and Local/Mock
providers with programmatic dimension verification and batching.
"""

import abc
import hashlib
import logging
import math
import os
from typing import List, Optional
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class BaseEmbedder(abc.ABC):
    """Abstract Base Class for text embedding generation."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality of this embedding model."""
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Return the identifier of the embedding model."""
        pass

    @abc.abstractmethod
    def embed_documents(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Generate dense vector embeddings for a list of texts."""
        pass

    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate a dense vector embedding for a single search query."""
        pass


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI Embedding Provider using text-embedding-3 models."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        from openai import OpenAI
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbedder.")
        self._client = OpenAI(api_key=self._api_key)
        self._dimension = 1536 if "small" in model else (3072 if "large" in model else 1536)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    def embed_documents(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            res = self._client.embeddings.create(input=batch, model=self._model)
            embeddings.extend([item.embedding for item in res.data])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        res = self._client.embeddings.create(input=[text], model=self._model)
        return res.data[0].embedding


class GeminiEmbedder(BaseEmbedder):
    """Google Gemini Embedding Provider using gemini-embedding-001 or text-embedding-004."""

    def __init__(self, model: str = "gemini-embedding-001", api_key: Optional[str] = None):
        from google import genai
        self._model = model
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for GeminiEmbedder.")
        self._client = genai.Client(api_key=self._api_key)
        # Verify dimension on initialization
        test_res = self._client.models.embed_content(model=self._model, contents="init")
        raw_vals = test_res.embedding.values if hasattr(test_res, "embedding") else test_res.embeddings[0].values
        self._dimension = len(raw_vals)
        logger.info("Initialized GeminiEmbedder with model %s (dimension=%d)", self._model, self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    def embed_documents(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                res = self._client.models.embed_content(model=self._model, contents=text)
                vals = res.embedding.values if hasattr(res, "embedding") else res.embeddings[0].values
                embeddings.append(list(vals))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        res = self._client.models.embed_content(model=self._model, contents=text)
        vals = res.embedding.values if hasattr(res, "embedding") else res.embeddings[0].values
        return list(vals)


class MockEmbedder(BaseEmbedder):
    """
    Deterministic pseudo-vector embedder for testing and offline execution.
    Generates normalized unit vectors derived from SHA-256 hashes of the text.
    Guarantees reproducible cosine similarities without external API dependency.
    """

    def __init__(self, dimension: int = 1536, model: str = "mock-deterministic-embedder"):
        self._dimension = dimension
        self._model = model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    def _generate_vector(self, text: str) -> List[float]:
        # Hash text deterministically
        raw_hash = hashlib.sha256(text.encode("utf-8")).digest()
        # Seed pseudo vector
        vector = []
        for i in range(self._dimension):
            byte_val = raw_hash[i % len(raw_hash)]
            val = (byte_val / 127.5) - 1.0 + (math.sin(i + len(text)) * 0.1)
            vector.append(val)

        # Normalize to unit length for cosine similarity
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._generate_vector(text)


def get_embedder(settings: Optional[Settings] = None) -> BaseEmbedder:
    """
    Factory function returning the configured embedder instance.
    Falls back gracefully if specific providers are chosen.
    """
    cfg = settings or get_settings()
    provider = cfg.EMBEDDING_PROVIDER.lower()

    if provider == "openai":
        try:
            return OpenAIEmbedder(model=cfg.EMBEDDING_MODEL, api_key=cfg.OPENAI_API_KEY)
        except Exception as e:
            logger.warning("Failed to initialize OpenAIEmbedder (%s). Checking alternatives...", e)
            # If Gemini is available, fallback to Gemini, else Mock
            if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                logger.info("Falling back to GeminiEmbedder.")
                return GeminiEmbedder(model="gemini-embedding-001")
            logger.info("Falling back to MockEmbedder.")
            return MockEmbedder(dimension=cfg.EMBEDDING_DIMENSION)

    elif provider == "gemini":
        try:
            return GeminiEmbedder(model=cfg.EMBEDDING_MODEL or "gemini-embedding-001", api_key=cfg.GEMINI_API_KEY)
        except Exception as e:
            logger.warning("Failed to initialize GeminiEmbedder (%s). Falling back to MockEmbedder.", e)
            return MockEmbedder(dimension=cfg.EMBEDDING_DIMENSION)

    else:
        return MockEmbedder(dimension=cfg.EMBEDDING_DIMENSION)
