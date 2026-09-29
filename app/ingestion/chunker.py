"""
Document Chunking and Metadata Tagging Module.
Applies conservative text cleaning and recursive semantic chunking while
attaching strict deterministic metadata for citations and hallucination prevention.
"""

import re
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional
from app.ingestion.loader import PDFPage

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata attached to every individual document chunk."""
    source: str          # e.g., 'Ebook-Agentic-AI.pdf'
    document_name: str   # Clean document title
    page: int            # 1-indexed page number from PDF
    chunk_id: str        # Deterministic identifier e.g., 'agentic-ai-p04-c01'
    char_count: int      # Character length of the chunk text


@dataclass
class DocumentChunk:
    """Represents a text chunk paired with rich retrieval metadata."""
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict:
        """Convert chunk to dictionary for serialization / vector store storage."""
        return {
            "text": self.text,
            "metadata": asdict(self.metadata),
        }


class TextChunker:
    """
    Performs conservative text normalization and recursive character chunking.
    
    Chunking Rationale:
    - chunk_size = 800 characters (~130–160 words): Captures cohesive conceptual statements,
      definitions, or architectural components from technical eBook layouts without diluting
      embedding density.
    - chunk_overlap = 150 characters (~20–25 words): Prevents boundary fragmentation where
      a definition or key point might otherwise be severed across neighboring chunks.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        doc_prefix: str = "agentic-ai",
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.doc_prefix = doc_prefix

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """
        Conservatively clean extracted PDF text:
        - Normalize carriage returns to standard newlines.
        - Collapse multiple inline horizontal spaces (tabs, consecutive spaces) into single spaces.
        - Preserve semantic paragraph breaks (double newlines).
        - Strip non-printable control characters except standard whitespace.
        """
        if not raw_text:
            return ""

        # Normalize line endings
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize common typographic and encoding anomalies
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        # Replace Unicode replacement character (often converted apostrophe in PDFs)
        text = text.replace("\ufffd's", "'s").replace("\ufffds", "'s").replace("\ufffd", " ")

        # Remove non-printable control characters (keeping \n and \t)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Normalize multiple spaces and tabs within the same line
        text = re.sub(r"[ \t]+", " ", text)

        # Collapse more than two consecutive newlines into two
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip leading and trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def _recursive_split(self, text: str) -> List[str]:
        """
        Recursively split text along natural semantic boundaries:
        1. Paragraph breaks (\n\n)
        2. Line breaks (\n)
        3. Sentence boundaries (. )
        4. Word boundaries ( )
        """
        if len(text) <= self.chunk_size:
            return [text] if text else []

        separators = ["\n\n", "\n", ". ", " "]
        chosen_sep = ""
        for sep in separators:
            if sep in text:
                chosen_sep = sep
                break

        if not chosen_sep:
            # Fallback to hard character slicing if no separator exists
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

        splits = text.split(chosen_sep)
        chunks: List[str] = []
        current_chunk = ""

        for part in splits:
            candidate = f"{current_chunk}{chosen_sep}{part}".strip() if current_chunk else part.strip()
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(part) > self.chunk_size:
                    # Sub-split long individual segments recursively
                    sub_chunks = self._recursive_split(part)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    # Apply overlap from the tail of the previous chunk if possible
                    if chunks and self.chunk_overlap > 0:
                        tail = chunks[-1][-self.chunk_overlap :]
                        current_chunk = f"{tail} {part}".strip()
                    else:
                        current_chunk = part.strip()

        if current_chunk:
            chunks.append(current_chunk)

        return [c for c in chunks if c.strip()]

    def chunk_page(
        self,
        page: PDFPage,
        source: str = "Ebook-Agentic-AI.pdf",
        document_name: str = "Agentic AI eBook",
    ) -> List[DocumentChunk]:
        """
        Clean and split a single extracted PDF page into metadata-tagged DocumentChunks.
        """
        cleaned_text = self.clean_text(page.text)
        if not cleaned_text:
            return []

        text_pieces = self._recursive_split(cleaned_text)
        chunks: List[DocumentChunk] = []

        for idx, piece in enumerate(text_pieces):
            chunk_id = f"{self.doc_prefix}-p{page.page_number:02d}-c{idx + 1:02d}"
            metadata = ChunkMetadata(
                source=source,
                document_name=document_name,
                page=page.page_number,
                chunk_id=chunk_id,
                char_count=len(piece),
            )
            chunks.append(DocumentChunk(text=piece, metadata=metadata))

        return chunks

    def chunk_document(
        self,
        pages: List[PDFPage],
        source: str = "Ebook-Agentic-AI.pdf",
        document_name: str = "Agentic AI eBook",
    ) -> List[DocumentChunk]:
        """
        Clean and chunk all pages of a document sequentially.
        """
        all_chunks: List[DocumentChunk] = []
        for page in pages:
            page_chunks = self.chunk_page(page, source=source, document_name=document_name)
            all_chunks.extend(page_chunks)

        logger.info(
            "Chunked document into %d total chunks across %d pages (avg %.1f chunks/page)",
            len(all_chunks),
            len(pages),
            len(all_chunks) / max(len(pages), 1),
        )
        return all_chunks
