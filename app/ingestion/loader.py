"""
PDF Acquisition and Page Extraction Module.
Downloads and extracts text page-by-page from the Agentic AI eBook,
preserving page metadata and handling corrupted/empty pages gracefully.
"""

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


@dataclass
class PDFPage:
    """Represents a single extracted page with metadata."""
    page_number: int  # 1-indexed page number
    text: str
    char_count: int


class PDFLoaderError(Exception):
    """Base exception for PDF loading and extraction errors."""
    pass


class PDFLoader:
    """Downloads and extracts text page-by-page from PDF documents."""

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def download_pdf(self, url: str, target_dir: str = "./data") -> Path:
        """
        Download PDF from URL to local directory if not already cached.
        
        Args:
            url: HTTP/HTTPS URL of the PDF document.
            target_dir: Local directory to store cached PDF.

        Returns:
            Path to the downloaded PDF file.
        """
        dest_dir = Path(target_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = url.split("/")[-1] or "document.pdf"
        dest_path = dest_dir / filename

        if dest_path.exists() and dest_path.stat().st_size > 0:
            logger.info("Using cached PDF file: %s (%d bytes)", dest_path, dest_path.stat().st_size)
            return dest_path

        logger.info("Downloading PDF from %s to %s...", url, dest_path)
        try:
            headers = {"User-Agent": "AgenticAI-RAG-Ingest/1.0"}
            response = requests.get(url, headers=headers, stream=True, timeout=self.timeout)
            response.raise_for_status()

            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)

            logger.info("Successfully downloaded PDF (%d bytes)", dest_path.stat().st_size)
            return dest_path

        except requests.RequestException as e:
            logger.error("Failed to download PDF from %s: %s", url, str(e))
            raise PDFLoaderError(f"Network error downloading PDF from {url}: {e}") from e

    def extract_pages(self, pdf_path: Path) -> List[PDFPage]:
        """
        Extract text page-by-page from a local PDF file.

        Args:
            pdf_path: Local filesystem path to the PDF.

        Returns:
            List of PDFPage objects with 1-indexed page numbers.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise PDFLoaderError(f"PDF file not found: {path}")

        if path.stat().st_size == 0:
            raise PDFLoaderError(f"PDF file is empty (0 bytes): {path}")

        # Verify PDF header magic bytes
        with open(path, "rb") as f:
            header = f.read(5)
            if not header.startswith(b"%PDF-"):
                raise PDFLoaderError(f"Invalid PDF header format in file: {path}")

        pages: List[PDFPage] = []
        try:
            reader = PdfReader(str(path))
            total_pages = len(reader.pages)
            logger.info("Extracting %d pages from %s", total_pages, path.name)

            if total_pages == 0:
                raise PDFLoaderError(f"PDF file contains 0 pages: {path}")

            for idx, page in enumerate(reader.pages):
                page_num = idx + 1
                try:
                    text = page.extract_text() or ""
                except Exception as extract_err:
                    logger.warning(
                        "Failed to extract text from page %d of %s: %s",
                        page_num, path.name, str(extract_err)
                    )
                    text = ""

                clean_raw = text.strip()
                if not clean_raw:
                    logger.warning("Page %d has empty extracted text (may be image or blank)", page_num)

                pages.append(
                    PDFPage(
                        page_number=page_num,
                        text=clean_raw,
                        char_count=len(clean_raw),
                    )
                )

        except PdfReadError as e:
            logger.error("Corrupted or unreadable PDF (%s): %s", path, str(e))
            raise PDFLoaderError(f"Corrupted or invalid PDF format: {e}") from e
        except Exception as e:
            if isinstance(e, PDFLoaderError):
                raise
            logger.error("Unexpected error during PDF extraction (%s): %s", path, str(e))
            raise PDFLoaderError(f"Failed to extract PDF pages: {e}") from e

        return pages
