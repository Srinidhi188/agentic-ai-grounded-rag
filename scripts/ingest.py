"""
Knowledge Base PDF Ingestion CLI Script.
Executes the full pipeline:
PDF Acquisition -> Page Extraction -> Text Cleaning -> Semantic Chunking ->
Embedding Generation -> Pinecone Vector Upsert.

Usage:
    python -m scripts.ingest [--dry-run] [--pdf-path PATH] [--batch-size N]
"""

import argparse
import logging
import sys
from pathlib import Path
from app.config import get_settings
from app.ingestion.loader import PDFLoader, PDFLoaderError
from app.ingestion.chunker import TextChunker
from app.ingestion.embedder import get_embedder
from app.vectorstore.pinecone_store import PineconeStore, PineconeNotConfiguredError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest Agentic AI eBook into Pinecone vector store.")
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="Optional local path to the eBook PDF file (bypasses download).",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional URL to download PDF from (defaults to configured PDF_URL).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform download, extraction, chunking, and embedding generation without upserting to Pinecone.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for embeddings and vector database upsert (default: 50).",
    )
    return parser.parse_args()


def run_ingestion(pdf_path: str = None, url: str = None, dry_run: bool = False, batch_size: int = 50):
    settings = get_settings()
    pdf_source_url = url or settings.PDF_URL

    logger.info("=== Starting Knowledge Base Ingestion Pipeline ===")
    logger.info("Configured Target PDF: %s", pdf_source_url)
    logger.info("Embedding Provider: %s (Model: %s)", settings.EMBEDDING_PROVIDER, settings.EMBEDDING_MODEL)
    logger.info("Dry run mode: %s", dry_run)

    # Step 1: PDF Acquisition
    loader = PDFLoader()
    if pdf_path and Path(pdf_path).exists():
        target_file = Path(pdf_path)
        logger.info("Using provided local PDF path: %s", target_file)
    else:
        try:
            target_file = loader.download_pdf(pdf_source_url, target_dir=settings.DATA_DIR)
        except PDFLoaderError as e:
            logger.error("Ingestion aborted during PDF download: %s", e)
            sys.exit(1)

    # Step 2: Page-by-Page Extraction
    try:
        pages = loader.extract_pages(target_file)
    except PDFLoaderError as e:
        logger.error("Ingestion aborted during PDF extraction: %s", e)
        sys.exit(1)

    non_empty_pages = [p for p in pages if p.text]
    total_chars = sum(p.char_count for p in non_empty_pages)
    logger.info(
        "Extracted %d total pages (%d pages contain extractable text, ~%d total characters)",
        len(pages),
        len(non_empty_pages),
        total_chars,
    )

    # Step 3 & 4: Conservative Cleaning and Semantic Chunking
    chunker = TextChunker(chunk_size=800, chunk_overlap=150)
    source_name = target_file.name
    chunks = chunker.chunk_document(pages, source=source_name, document_name="Agentic AI eBook")

    if not chunks:
        logger.error("No valid text chunks generated from document. Aborting.")
        sys.exit(1)

    logger.info(
        "Generated %d chunks. Average chunk length: %.1f chars.",
        len(chunks),
        sum(c.metadata.char_count for c in chunks) / len(chunks),
    )

    # Step 5: Embedding Model Initialization & Generation
    try:
        embedder = get_embedder(settings)
        logger.info(
            "Initialized embedder '%s' (vector dimension=%d)",
            embedder.model_name,
            embedder.dimension,
        )
    except Exception as e:
        logger.error("Failed to initialize embedding model: %s", e)
        sys.exit(1)

    texts = [chunk.text for chunk in chunks]
    logger.info("Generating embeddings for %d chunks in batches of %d...", len(texts), batch_size)
    try:
        embeddings = embedder.embed_documents(texts, batch_size=batch_size)
        logger.info("Successfully generated %d embeddings.", len(embeddings))
    except Exception as e:
        logger.error("Failed during embedding computation: %s", e)
        sys.exit(1)

    # Programmatic dimension validation
    if embeddings and len(embeddings[0]) != embedder.dimension:
        logger.warning(
            "Dimension mismatch: expected %d, got %d. Updating dimension setting.",
            embedder.dimension,
            len(embeddings[0]),
        )

    # Step 6: Vector Database Upsert
    if dry_run:
        logger.info("--- DRY-RUN SUMMARY ---")
        logger.info("Total Pages: %d", len(pages))
        logger.info("Total Chunks: %d", len(chunks))
        logger.info("Embedding Dimension: %d", len(embeddings[0]) if embeddings else 0)
        logger.info("Sample Chunk ID: %s (Page %d)", chunks[0].metadata.chunk_id, chunks[0].metadata.page)
        logger.info("Dry-run complete. No vectors upserted to Pinecone.")
        return

    # Upsert to Pinecone
    try:
        store = PineconeStore(settings)
        store.ensure_index(dimension=embedder.dimension, metric="cosine")
        upserted_count = store.upsert_chunks(chunks, embeddings, batch_size=batch_size)
        logger.info("=== Ingestion Successfully Finished (%d vectors stored in Pinecone) ===", upserted_count)
    except PineconeNotConfiguredError as e:
        logger.warning("Pinecone integration note: %s", e)
        logger.warning("To upsert vectors to Pinecone, provide PINECONE_API_KEY in .env and re-run without --dry-run.")
    except Exception as e:
        logger.error("Error upserting vectors to Pinecone: %s", e)
        sys.exit(1)


def main():
    args = parse_args()
    run_ingestion(
        pdf_path=args.pdf_path,
        url=args.url,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
