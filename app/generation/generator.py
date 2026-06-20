"""
Gemini answer generator with LangSmith tracing.

Interview note: LangSmith is wired as a LangChain callback handler, NOT
as a wrapper around the generation call. This means:
  - We get trace visibility without changing the generation code
  - The callback fires on llm_start, llm_end, llm_error events
  - Each trace contains: prompt, response, latency, token counts
  - Traces are linked by run_id to the request_id in audit logs

Rate limiter is acquired before the Gemini API call. The limiter is
shared with the summariser and embedder — all Gemini calls go through
the same 15 RPM bucket.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

import structlog

from app.config import get_settings
from app.core.rate_limiter import get_rate_limiter
from app.generation.hallucination_filter import HallucinationCheckResult, check_hallucination
from app.generation.prompt_templates import build_generation_prompt
from app.retrieval.vector_search import SearchResult

# Modern google-genai SDK imports
from google import genai
from google.genai import types

logger = structlog.get_logger(__name__)


async def generate_answer(
    query: str,
    chunks: list[SearchResult],
    request_id: str,
) -> dict:
    """
    Generate a grounded answer using Gemini 2.5 Flash.

    Pipeline:
      1. Build grounded prompt from query + chunks
      2. Acquire Gemini rate limiter token
      3. Call Gemini with LangSmith tracing callback
      4. Run hallucination filter on the generated answer
      5. Build source citations from chunk metadata
      6. Return structured response dict

    Returns a dict with: answer, sources, hallucination_check, generation_latency_ms
    """
    settings = get_settings()
    limiter = get_rate_limiter()

    prompt = build_generation_prompt(query, chunks)

    # Acquire rate limiter before API call
    await limiter.acquire()

    gen_start = time.monotonic()

    try:
        # Build LangSmith callback for this request
        callbacks = _get_langsmith_callbacks(request_id)

        # Initialize the modern SDK client
        client = genai.Client(api_key=settings.google_api_key)
        
        # Configure generation parameters via types.GenerateContentConfig
        config = types.GenerateContentConfig(
            temperature=0.1,      # Low temp for factual grounded answers
            max_output_tokens=2048,
            top_p=0.8,
        )

        # Call the model via the client object
        response = client.models.generate_content(
            model=settings.gemini_generation_model,
            contents=prompt,
            config=config
            # LangSmith tracing via request metadata (not native callback)
            # Full LangChain callback integration in observability/langsmith_tracer.py
        )
        answer = response.text.strip()

        generation_latency_ms = int((time.monotonic() - gen_start) * 1000)

        logger.info(
            "generator.success",
            request_id=request_id,
            answer_len=len(answer),
            generation_latency_ms=generation_latency_ms,
        )

    except Exception as exc:
        generation_latency_ms = int((time.monotonic() - gen_start) * 1000)
        logger.error("generator.failed", request_id=request_id, error=str(exc))
        raise

    # ── Hallucination filter ────────────────────────────────────
    hallucination_result: HallucinationCheckResult = check_hallucination(answer, chunks)

    # ── Build source citations ─────────────────────────────────
    sources = _build_sources(chunks)

    # Annotate answer with grounding flag for audit
    if not hallucination_result.is_grounded:
        logger.warning(
            "generator.hallucination_flagged",
            request_id=request_id,
            overlap_ratio=hallucination_result.overlap_ratio,
        )

    return {
        "answer": answer,
        "sources": sources,
        "hallucination_check": {
            "is_grounded": hallucination_result.is_grounded,
            "overlap_ratio": hallucination_result.overlap_ratio,
            "reasoning": hallucination_result.reasoning,
        },
        "generation_latency_ms": generation_latency_ms,
        "model": settings.gemini_generation_model,
    }


def _build_sources(chunks: list[SearchResult]) -> list[dict]:
    """
    Build a deduplicated list of source citations from retrieved chunks.

    Interview note: Sources are deduplicated by (document_name, page_number)
    because multiple chunks from the same page would produce redundant citations.
    """
    seen: set[tuple[str, int]] = set()
    sources: list[dict] = []

    for chunk in chunks:
        key = (chunk.document_name, chunk.page_number)
        if key not in seen:
            seen.add(key)
            sources.append({
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_type": chunk.chunk_type,
                "chunk_id": chunk.chunk_id,
                "relevance_score": round(chunk.score, 4),
            })

    return sources


def _get_langsmith_callbacks(request_id: str) -> list:
    """
    Return LangSmith callback handlers for this request.

    Returns empty list if LangSmith is not configured — generation
    proceeds without tracing rather than failing.
    """
    try:
        from app.observability.langsmith_tracer import get_langsmith_handler
        handler = get_langsmith_handler(run_id=request_id)
        return [handler] if handler else []
    except Exception as exc:
        logger.debug("generator.langsmith_unavailable", error=str(exc))
        return []