"""
POST /api/v1/ingest — Accept a PDF and run the full ingestion pipeline.

Accepts multipart/form-data with a PDF file field named "file".
Validates content type, runs the pipeline, returns ingestion summary.
"""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, HTTPException, UploadFile, status

from app.core.metrics_store import get_metrics
from app.core.sanitiser import sanitise_document_name
from app.dependencies import AuthDep
from app.ingestion.pipeline import run_ingestion_pipeline

logger = structlog.get_logger(__name__)
router = APIRouter()

MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB hard limit


@router.post(
    "/ingest",
    summary="Ingest a PDF document",
    description=(
        "Upload a PDF file for ingestion. The pipeline will parse text, tables, "
        "and images separately, summarise non-text elements using Gemini Vision, "
        "embed all chunks, and store them in MongoDB for hybrid search. "
        "Re-ingesting the same filename creates a new version."
    ),
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_document(
    file: UploadFile,
    api_key: AuthDep,
) -> dict:
    """
    Ingest a PDF document.

    - **file**: PDF file (multipart/form-data, max 50 MB)

    Returns ingestion summary with document_id, version, chunk count breakdown.
    """
    metrics = get_metrics()
    metrics.record_request("ingest")
    start_time = time.monotonic()

    # ── Validate content type ──────────────────────────────────
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only PDF files are accepted. Got: {file.content_type!r}",
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File must have a filename.",
        )

    # ── Read and size-check ────────────────────────────────────
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )
    if len(file_bytes) > MAX_PDF_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_PDF_SIZE_BYTES // (1024*1024)} MB.",
        )

    # ── Validate PDF magic bytes ───────────────────────────────
    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File does not appear to be a valid PDF (missing %PDF header).",
        )

    # ── Sanitise filename ──────────────────────────────────────
    safe_filename = sanitise_document_name(file.filename)

    logger.info(
        "ingest.request",
        filename=safe_filename,
        size_kb=round(len(file_bytes) / 1024, 1),
        api_key_name=api_key.name,
    )

    # ── Run pipeline ───────────────────────────────────────────
    result = await run_ingestion_pipeline(
        filename=safe_filename,
        file_bytes=file_bytes,
    )

    total_latency_ms = int((time.monotonic() - start_time) * 1000)
    result["latency_ms"] = total_latency_ms

    if result.get("status") == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {result.get('error', 'Unknown error')}",
        )

    logger.info(
        "ingest.complete",
        filename=safe_filename,
        document_id=result.get("document_id"),
        version=result.get("version"),
        chunk_count=result.get("chunk_count"),
        latency_ms=total_latency_ms,
    )

    return result
