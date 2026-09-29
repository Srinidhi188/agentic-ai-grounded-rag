"""
Tests for Document Ingestion, PDF Extraction, Semantic Chunking,
Embeddings Interface, and Vector Store Integration.
"""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from pypdf import PdfWriter

from app.config import Settings
from app.ingestion.loader import PDFLoader, PDFPage, PDFLoaderError
from app.ingestion.chunker import TextChunker, ChunkMetadata, DocumentChunk
from app.ingestion.embedder import MockEmbedder, get_embedder
from app.vectorstore.pinecone_store import (
    PineconeStore,
    PineconeNotConfiguredError,
    PineconeStoreError,
)


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a minimal real 2-page PDF file with pypdf."""
    pdf_path = tmp_path / "test_sample.pdf"
    writer = PdfWriter()
    # Add page 1 with text
    writer.add_blank_page(width=200, height=200)
    # Add page 2 (blank)
    writer.add_blank_page(width=200, height=200)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


def test_pdf_loader_invalid_file(tmp_path):
    """Verify loader raises error for missing or empty files."""
    loader = PDFLoader()
    with pytest.raises(PDFLoaderError, match="not found"):
        loader.extract_pages(tmp_path / "non_existent.pdf")

    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")
    with pytest.raises(PDFLoaderError, match="empty"):
        loader.extract_pages(empty_file)


def test_pdf_loader_invalid_header(tmp_path):
    """Verify loader checks for PDF magic header."""
    loader = PDFLoader()
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"NOT_A_PDF_DOCUMENT_CONTENT")
    with pytest.raises(PDFLoaderError, match="Invalid PDF header"):
        loader.extract_pages(corrupt_file)


def test_pdf_loader_extract_pages(sample_pdf):
    """Verify loader extracts pages with 1-indexed page numbering."""
    loader = PDFLoader()
    pages = loader.extract_pages(sample_pdf)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2


def test_clean_text():
    """Verify conservative text cleaning preserves semantics while cleaning artifacts."""
    raw = "  Agentic AI \r\n\r\n is  transformative. \x00 Here’s a “quote” — dash. \n\n\n\n Next section.  "
    cleaned = TextChunker.clean_text(raw)
    assert "\r" not in cleaned
    assert "\x00" not in cleaned
    assert "Here's" in cleaned or "Here’s" not in cleaned
    assert '"quote"' in cleaned
    assert "- dash." in cleaned
    # Check that excessive newlines are collapsed
    assert "\n\n\n" not in cleaned


def test_chunker_empty_page():
    """Verify empty pages produce no chunks."""
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    empty_page = PDFPage(page_number=1, text="   ", char_count=3)
    chunks = chunker.chunk_page(empty_page)
    assert len(chunks) == 0


def test_chunker_overlap_validation():
    """Verify chunk_overlap must be strictly less than chunk_size."""
    with pytest.raises(ValueError, match="strictly less than"):
        TextChunker(chunk_size=200, chunk_overlap=200)

    with pytest.raises(ValueError, match="strictly less than"):
        TextChunker(chunk_size=200, chunk_overlap=250)


def test_chunker_deterministic_metadata_and_ids():
    """Verify chunking assigns correct deterministic chunk_ids and metadata."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20, doc_prefix="test-doc")
    long_text = "Sentence one is clear. " * 10
    page = PDFPage(page_number=3, text=long_text, char_count=len(long_text))

    chunks = chunker.chunk_page(page, source="test.pdf", document_name="Test Doc")
    assert len(chunks) > 1

    # Check deterministic naming format
    assert chunks[0].metadata.chunk_id == "test-doc-p03-c01"
    assert chunks[1].metadata.chunk_id == "test-doc-p03-c02"
    assert chunks[0].metadata.page == 3
    assert chunks[0].metadata.source == "test.pdf"
    assert chunks[0].metadata.document_name == "Test Doc"
    assert chunks[0].metadata.char_count == len(chunks[0].text)


def test_mock_embedder_dimension_and_unit_norm():
    """Verify MockEmbedder outputs vectors with exact requested dimension and unit length."""
    embedder = MockEmbedder(dimension=512)
    assert embedder.dimension == 512
    assert embedder.model_name == "mock-deterministic-embedder"

    vec = embedder.embed_query("What is Agentic AI?")
    assert len(vec) == 512

    # Check unit norm: sqrt(sum(x^2)) ≈ 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    assert pytest.approx(norm, rel=1e-4) == 1.0

    # Verify batch embedding
    batch_vecs = embedder.embed_documents(["First doc", "Second doc"], batch_size=2)
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 512


def test_embedder_factory():
    """Verify embedder factory returns MockEmbedder for 'local' setting."""
    settings = Settings(EMBEDDING_PROVIDER="local", EMBEDDING_DIMENSION=256)
    embedder = get_embedder(settings)
    assert isinstance(embedder, MockEmbedder)
    assert embedder.dimension == 256


def test_pinecone_store_missing_api_key():
    """Verify PineconeStore raises PineconeNotConfiguredError when API key is missing."""
    settings = Settings(PINECONE_API_KEY=None)
    store = PineconeStore(settings)
    with pytest.raises(PineconeNotConfiguredError, match="PINECONE_API_KEY is not set"):
        _ = store.client


def test_pinecone_store_mocked_upsert_and_query():
    """Verify PineconeStore batch upsert and query mapping using mocked Pinecone client."""
    settings = Settings(
        PINECONE_API_KEY="mock-pinecone-key",
        PINECONE_INDEX_NAME="test-index",
        PINECONE_NAMESPACE="test-namespace",
    )
    store = PineconeStore(settings)

    mock_client = MagicMock()
    mock_index = MagicMock()
    mock_idx_model = MagicMock()
    mock_idx_model.name = "test-index"
    mock_client.list_indexes.return_value = [mock_idx_model]
    mock_client.Index.return_value = mock_index
    store._client = mock_client
    store._index = mock_index

    # Test ensure_index does not recreate existing index
    store.ensure_index(dimension=1536)
    mock_client.create_index.assert_not_called()

    # Test upsert_chunks
    meta = ChunkMetadata(
        source="doc.pdf",
        document_name="Doc",
        page=1,
        chunk_id="doc-p01-c01",
        char_count=50,
    )
    chunks = [DocumentChunk(text="Sample chunk text", metadata=meta)]
    embeddings = [[0.1] * 1536]

    count = store.upsert_chunks(chunks, embeddings, batch_size=50)
    assert count == 1
    mock_index.upsert.assert_called_once()
    upsert_args = mock_index.upsert.call_args[1]
    assert upsert_args["namespace"] == "test-namespace"
    assert len(upsert_args["vectors"]) == 1
    assert upsert_args["vectors"][0]["id"] == "doc-p01-c01"

    # Test query
    mock_index.query.return_value = {
        "matches": [
            {
                "id": "doc-p01-c01",
                "score": 0.88,
                "metadata": {
                    "text": "Sample chunk text",
                    "source": "doc.pdf",
                    "page": 1,
                    "document_name": "Doc",
                },
            }
        ]
    }
    results = store.query(query_embedding=[0.1] * 1536, top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "doc-p01-c01"
    assert results[0]["score"] == 0.88
    assert results[0]["page"] == 1
    assert results[0]["text"] == "Sample chunk text"
