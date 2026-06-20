"""
PDF parser using unstructured.partition.pdf with hi-res strategy.

Interview note: We use unstructured's hi-res strategy (backed by
detectron2 layout detection) rather than naive text extraction because:
  - Hi-res preserves table structure (row/column alignment)
  - Images are extracted as separate elements with bounding boxes
  - Section headers are detected, enabling hierarchical chunking
  - Naive PyPDF2/pdfminer extraction loses all table formatting

Three element types are returned as separate chunks:
  - text   : paragraphs, headers, list items
  - table  : HTML table string + surrounding context
  - image  : base64-encoded image bytes for Vision summarisation

Each chunk carries page_number for source traceability.
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog
from PIL import Image

logger = structlog.get_logger(__name__)


@dataclass
class ParsedChunk:
    """
    A single chunk extracted from a PDF.

    chunk_id is a SHA-256 hash of (document_name + page + index + content[:100])
    to ensure deterministic deduplication across re-ingests.
    """

    chunk_id: str
    chunk_type: str          # "text" | "table" | "image"
    content: str             # Text content or summary (filled by summariser for table/image)
    page_number: int
    chunk_index: int
    document_name: str
    # Raw data for Vision summarisation — not stored in MongoDB after processing
    raw_table_html: Optional[str] = None
    raw_image_b64: Optional[str] = None
    # Additional metadata
    section: Optional[str] = None


def _make_chunk_id(document_name: str, page_number: int, chunk_index: int, content_prefix: str) -> str:
    """Generate a deterministic SHA-256 chunk ID."""
    raw = f"{document_name}:p{page_number}:i{chunk_index}:{content_prefix[:100]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_pdf(file_path: Path, document_name: str) -> list[ParsedChunk]:
    """
    Parse a PDF file using unstructured hi-res strategy.

    Returns a list of ParsedChunk objects with type=text|table|image.
    Each chunk has a page_number and deterministic chunk_id.

    The hi-res strategy is slower than "fast" but preserves:
      - Table HTML structure via table_structure detection
      - Images as CompositeElement or Image elements
      - Section hierarchy via Title elements

    Falls back to "fast" strategy if hi-res dependencies (detectron2)
    are not installed — logs a warning so the operator knows.
    """
    # Import inside function to avoid slow module-level import
    try:
        from unstructured.partition.pdf import partition_pdf
        strategy = "hi_res"
    except ImportError:
        logger.error("parser.unstructured_not_installed")
        raise

    logger.info("parser.starting", file_path=str(file_path), strategy=strategy)

    try:
        elements = partition_pdf(
            filename=str(file_path),
            strategy=strategy,
            infer_table_structure=True,        # Extracts HTML table markup
            extract_images_in_pdf=True,        # Extracts embedded images
            extract_image_block_types=["Image", "Table"],
            include_page_breaks=True,
        )
    except Exception as exc:
        logger.warning(
            "parser.hi_res_failed_falling_back",
            error=str(exc),
            strategy="fast",
        )
        from unstructured.partition.pdf import partition_pdf as partition_fast
        elements = partition_fast(
            filename=str(file_path),
            strategy="fast",
            infer_table_structure=True,
            include_page_breaks=True,
        )

    chunks: list[ParsedChunk] = []
    chunk_index = 0
    current_section: Optional[str] = None

    for element in elements:
        elem_type = type(element).__name__
        page_num = _get_page_number(element)

        # Track current section heading for context
        if elem_type == "Title":
            current_section = str(element).strip()

        # ── Table element ──────────────────────────────────────
        if elem_type == "Table":
            table_html = _get_table_html(element)
            # Content is placeholder — summariser will fill it with Gemini
            content_preview = str(element)[:200]
            chunk = ParsedChunk(
                chunk_id=_make_chunk_id(document_name, page_num, chunk_index, content_preview),
                chunk_type="table",
                content=content_preview,       # Overwritten by summariser
                page_number=page_num,
                chunk_index=chunk_index,
                document_name=document_name,
                raw_table_html=table_html,
                section=current_section,
            )
            chunks.append(chunk)
            chunk_index += 1

        # ── Image element ──────────────────────────────────────
        elif elem_type in ("Image", "FigureCaption"):
            image_b64 = _get_image_b64(element)
            if image_b64:
                chunk = ParsedChunk(
                    chunk_id=_make_chunk_id(document_name, page_num, chunk_index, f"image_{chunk_index}"),
                    chunk_type="image",
                    content=f"[Image on page {page_num}]",   # Overwritten by summariser
                    page_number=page_num,
                    chunk_index=chunk_index,
                    document_name=document_name,
                    raw_image_b64=image_b64,
                    section=current_section,
                )
                chunks.append(chunk)
                chunk_index += 1

        # ── Text element ───────────────────────────────────────
        elif elem_type not in ("PageBreak",):
            text = str(element).strip()
            if len(text) < 20:     # Skip very short fragments (headers, footers)
                continue
            chunk = ParsedChunk(
                chunk_id=_make_chunk_id(document_name, page_num, chunk_index, text),
                chunk_type="text",
                content=text,
                page_number=page_num,
                chunk_index=chunk_index,
                document_name=document_name,
                section=current_section,
            )
            chunks.append(chunk)
            chunk_index += 1

    logger.info(
        "parser.complete",
        document_name=document_name,
        total_chunks=len(chunks),
        text_chunks=sum(1 for c in chunks if c.chunk_type == "text"),
        table_chunks=sum(1 for c in chunks if c.chunk_type == "table"),
        image_chunks=sum(1 for c in chunks if c.chunk_type == "image"),
    )
    return chunks


def _get_page_number(element: object) -> int:
    """Extract page number from an unstructured element. Defaults to 1."""
    try:
        metadata = getattr(element, "metadata", None)
        if metadata:
            return int(getattr(metadata, "page_number", 1) or 1)
    except (AttributeError, TypeError, ValueError):
        pass
    return 1


def _get_table_html(element: object) -> Optional[str]:
    """Extract HTML table markup from a Table element."""
    try:
        metadata = getattr(element, "metadata", None)
        if metadata:
            return getattr(metadata, "text_as_html", None)
    except AttributeError:
        pass
    return str(element)


def _get_image_b64(element: object) -> Optional[str]:
    """
    Extract base64-encoded image from an Image element.
    Returns None if image data is not available.
    """
    try:
        # unstructured stores image bytes in metadata.image_base64
        metadata = getattr(element, "metadata", None)
        if metadata:
            img_b64 = getattr(metadata, "image_base64", None)
            if img_b64:
                return img_b64

        # Fallback: try to get image from coordinates/crop (unstructured >= 0.10)
        image_path = getattr(getattr(element, "metadata", None), "image_path", None)
        if image_path and Path(image_path).exists():
            with Image.open(image_path) as img:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as exc:
        logger.debug("parser.image_extract_failed", error=str(exc))
    return None


def count_pages(file_path: Path) -> int:
    """Count the number of pages in a PDF without full parsing."""
    try:
        import pdfminer.high_level
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfparser import PDFParser

        with open(file_path, "rb") as f:
            parser = PDFParser(f)
            doc = PDFDocument(parser)
            return sum(1 for _ in pdfminer.high_level.extract_pages(str(file_path)))
    except Exception:
        return 0
