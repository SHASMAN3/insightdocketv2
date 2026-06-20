"""
POST /api/v1/query — Ask a question against ingested documents.

Full pipeline:
  1. Input sanitisation + injection detection
  2. Embed query → $vectorSearch + $text BM25
  3. RRF fusion → cross-encoder reranking
  4. Confidence scoring → fallback if below threshold
  5. Gemini generation with grounded prompt
  6. Hallucination filter
  7. Audit log write
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.config import get_settings
from app.core.metrics_store import get_metrics
from app.core.sanitiser import sanitise_input
from app.dependencies import AuthDep
from app.generation.generator import generate_answer
from app.ingestion.embedder import embed_query
from app.observability.audit_logger import write_audit_log
from app.observability.langsmith_tracer import trace_query_event
from app.retrieval.confidence import build_fallback_response, compute_confidence
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.retrieval.text_search import text_search
from app.retrieval.vector_search import vector_search

logger = structlog.get_logger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    """Request body for /query."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language question to ask against ingested documents.",
        examples=["What is the total revenue reported in Q3?"],
    )
    document_id: Optional[int] = Field(
        default=None,
        description="Optional: restrict search to a specific document ID.",
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override default confidence threshold (0.0–1.0).",
    )

    @field_validator("question")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class QueryResponse(BaseModel):
    """Response body for /query."""

    request_id: str
    answer: str
    response_type: str
    confidence_score: float
    sources: list[dict]
    hallucination_check: Optional[dict] = None
    retrieval_latency_ms: int
    generation_latency_ms: Optional[int] = None
    total_latency_ms: int
    injection_detected: bool
    query: str


@router.post(
    "/query",
    summary="Ask a question against ingested documents",
    response_model=QueryResponse,
)
async def query_documents(
    body: QueryRequest,
    api_key: AuthDep,
) -> QueryResponse:
    """
    Run a RAG query against all ingested documents (or a specific document).

    Returns a grounded answer with source citations and confidence score.
    If confidence is below threshold, returns a structured fallback response.
    """
    settings = get_settings()
    metrics = get_metrics()
    metrics.record_request("query")

    request_id = str(uuid.uuid4())
    total_start = time.monotonic()

    logger.info(
        "query.request",
        request_id=request_id,
        question_preview=body.question[:80],
        document_id=body.document_id,
        api_key_name=api_key.name,
    )

    # ── Step 1: Input sanitisation + injection detection ───────
    san_result = sanitise_input(body.question)
    clean_question = san_result.sanitised_text

    if san_result.injection_detected:
        logger.warning(
            "query.injection_blocked",
            request_id=request_id,
            pattern=san_result.injection_pattern,
        )
        response_data = {
            "answer": "Your query contains patterns associated with prompt injection attacks and cannot be processed.",
            "response_type": "injection_blocked",
            "confidence_score": 0.0,
            "sources": [],
            "retrieval_latency_ms": 0,
            "generation_latency_ms": None,
            "total_latency_ms": int((time.monotonic() - total_start) * 1000),
            "injection_detected": True,
            "query": body.question[:200],
        }
        await _write_audit(request_id, body.question, response_data, None)
        metrics.record_response(
            "injection_blocked",
            total_latency_ms=response_data["total_latency_ms"],
        )
        return QueryResponse(request_id=request_id, **response_data)

    # ── Step 2: Embed query ────────────────────────────────────
    retrieval_start = time.monotonic()
    query_embedding = await embed_query(clean_question)

    # ── Step 3: Hybrid retrieval ───────────────────────────────
    vector_results, text_results = await _run_hybrid_search(
        query_embedding=query_embedding,
        query_text=clean_question,
        document_id=body.document_id,
        settings=settings,
    )

    # ── Step 4: RRF fusion ─────────────────────────────────────
    fused = reciprocal_rank_fusion(
        vector_results=vector_results,
        text_results=text_results,
        top_k=settings.vector_top_k + settings.text_top_k,
    )

    # ── Step 5: Reranking ──────────────────────────────────────
    reranked = await rerank(
        query=clean_question,
        candidates=fused,
        top_k=settings.final_top_k,
    )

    retrieval_latency_ms = int((time.monotonic() - retrieval_start) * 1000)

    # ── Step 6: Confidence scoring ─────────────────────────────
    confidence_result = compute_confidence(
        reranked_chunks=reranked,
        threshold=body.confidence_threshold,
    )

    if not confidence_result.above_threshold:
        fallback = build_fallback_response(clean_question, confidence_result)
        total_latency_ms = int((time.monotonic() - total_start) * 1000)
        response_data = {
            **fallback,
            "request_id": request_id,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": None,
            "total_latency_ms": total_latency_ms,
            "injection_detected": False,
            "query": clean_question,
        }
        await _write_audit(request_id, clean_question, response_data, reranked)
        metrics.record_response(
            "fallback",
            total_latency_ms=total_latency_ms,
            retrieval_latency_ms=float(retrieval_latency_ms),
            confidence_score=confidence_result.score,
        )
        return QueryResponse(**response_data)

    # ── Step 7: Generation ─────────────────────────────────────
    gen_result = await generate_answer(
        query=clean_question,
        chunks=reranked,
        request_id=request_id,
    )

    total_latency_ms = int((time.monotonic() - total_start) * 1000)

    response_data = {
        "request_id": request_id,
        "answer": gen_result["answer"],
        "response_type": "grounded",
        "confidence_score": confidence_result.score,
        "sources": gen_result["sources"],
        "hallucination_check": gen_result["hallucination_check"],
        "retrieval_latency_ms": retrieval_latency_ms,
        "generation_latency_ms": gen_result["generation_latency_ms"],
        "total_latency_ms": total_latency_ms,
        "injection_detected": False,
        "query": clean_question,
    }

    # ── Step 8: Audit log ──────────────────────────────────────
    await _write_audit(request_id, clean_question, response_data, reranked)

    # ── Step 9: LangSmith trace ────────────────────────────────
    await trace_query_event(
        request_id=request_id,
        query=clean_question,
        answer=gen_result["answer"],
        latency_ms=total_latency_ms,
        confidence_score=confidence_result.score,
        response_type="grounded",
    )

    metrics.record_response(
        "grounded",
        total_latency_ms=float(total_latency_ms),
        retrieval_latency_ms=float(retrieval_latency_ms),
        generation_latency_ms=float(gen_result["generation_latency_ms"] or 0),
        confidence_score=confidence_result.score,
    )

    logger.info(
        "query.complete",
        request_id=request_id,
        response_type="grounded",
        confidence=confidence_result.score,
        total_latency_ms=total_latency_ms,
    )

    return QueryResponse(**response_data)


async def _run_hybrid_search(
    query_embedding: list[float],
    query_text: str,
    document_id: int | None,
    settings: object,
) -> tuple:
    """Run vector and text searches concurrently."""
    import asyncio
    from app.config import Settings
    assert isinstance(settings, Settings)

    vector_task = asyncio.create_task(
        vector_search(
            query_embedding=query_embedding,
            top_k=settings.vector_top_k,
            document_id=document_id,
        )
    )
    text_task = asyncio.create_task(
        text_search(
            query=query_text,
            top_k=settings.text_top_k,
            document_id=document_id,
        )
    )
    vector_results, text_results = await asyncio.gather(vector_task, text_task)
    return vector_results, text_results


async def _write_audit(
    request_id: str,
    question: str,
    response_data: dict,
    chunks: list | None,
) -> None:
    """Write audit log entry — swallows exceptions."""
    try:
        chunk_ids = [c.chunk_id for c in chunks] if chunks else []
        page_numbers = list({c.page_number for c in chunks}) if chunks else []
        document_id = chunks[0].document_id if chunks else None

        await write_audit_log(
            request_id=request_id,
            question=question,
            answer=response_data.get("answer"),
            response_type=response_data.get("response_type", "unknown"),
            confidence_score=response_data.get("confidence_score"),
            retrieval_latency_ms=response_data.get("retrieval_latency_ms"),
            generation_latency_ms=response_data.get("generation_latency_ms"),
            total_latency_ms=response_data.get("total_latency_ms", 0),
            chunk_ids=chunk_ids,
            page_numbers=page_numbers,
            injection_detected=response_data.get("injection_detected", False),
            document_id=document_id,
        )
    except Exception as exc:
        logger.error("query.audit_write_failed", request_id=request_id, error=str(exc))
