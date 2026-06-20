"""
Cross-encoder reranker for post-RRF result refinement.

Interview note: Bi-encoder embeddings (used for ANN retrieval) optimise
for speed by encoding query and document independently. Cross-encoders
attend jointly to query+document, capturing fine-grained relevance signals
that bi-encoders miss — at the cost of O(n) inference per query.

Architecture:
  1. $vectorSearch + $text → 40 candidates (RRF fused)
  2. Cross-encoder reranks 40 → 5 final chunks
  3. Confidence scoring on the 5 final chunks

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M parameters — fast enough for real-time reranking
  - Trained on MS MARCO passage ranking (∼500k queries)
  - Runs entirely locally — zero API cost, zero latency to external service
  - Downloaded once, cached at ~/.cache/huggingface/hub/

The model is loaded lazily and cached at module level to avoid
reloading on every request (transformer loading takes ~2s).
"""

from __future__ import annotations

import structlog

from app.config import get_settings
from app.retrieval.vector_search import SearchResult

logger = structlog.get_logger(__name__)

# Module-level cache — model loaded once, reused across requests
_cross_encoder = None


def _get_cross_encoder():  # type: ignore[return]
    """
    Lazily load and cache the cross-encoder model.
    Thread-safe in CPython due to GIL; asyncio ensures single-threaded access.
    """
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        settings = get_settings()
        logger.info("reranker.loading_model", model=settings.reranker_model)
        _cross_encoder = CrossEncoder(
            settings.reranker_model,
            max_length=512,     # MiniLM context window
        )
        logger.info("reranker.model_loaded", model=settings.reranker_model)
    return _cross_encoder


async def rerank(
    query: str,
    candidates: list[SearchResult],
    top_k: int | None = None,
) -> list[SearchResult]:
    """
    Rerank candidate chunks using a cross-encoder model.

    The cross-encoder scores each (query, chunk_content) pair jointly,
    producing a relevance score in (-∞, +∞) that we normalise to [0, 1]
    via sigmoid for downstream confidence scoring.

    Args:
        query: original user question
        candidates: RRF-fused candidates to rerank
        top_k: number of top results to return (defaults to settings.final_top_k)

    Returns list[SearchResult] with .score updated to sigmoid(cross_encoder_score),
    sorted descending by the new score.
    """
    settings = get_settings()
    k = top_k or settings.final_top_k

    if not candidates:
        return []

    cross_encoder = _get_cross_encoder()

    # Build (query, passage) pairs for batch inference
    pairs = [(query, result.content) for result in candidates]

    # Cross-encoder inference — blocking CPU-bound call
    # For production with high concurrency, run in ThreadPoolExecutor
    import asyncio
    scores: list[float] = await asyncio.get_event_loop().run_in_executor(
        None,  # Default executor (ThreadPoolExecutor)
        lambda: cross_encoder.predict(pairs).tolist(),
    )

    # Sigmoid normalisation: score ∈ (-∞, +∞) → (0, 1)
    import math
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    normalised_scores = [sigmoid(s) for s in scores]

    # Attach normalised scores and sort
    scored_results: list[tuple[float, SearchResult]] = [
        (normalised_scores[i], candidates[i])
        for i in range(len(candidates))
    ]
    scored_results.sort(key=lambda x: x[0], reverse=True)

    # Rebuild SearchResult objects with updated scores
    reranked: list[SearchResult] = []
    for score, result in scored_results[:k]:
        reranked.append(SearchResult(
            chunk_id=result.chunk_id,
            content=result.content,
            chunk_type=result.chunk_type,
            document_id=result.document_id,
            document_name=result.document_name,
            page_number=result.page_number,
            chunk_index=result.chunk_index,
            version=result.version,
            score=round(score, 6),
            metadata=result.metadata,
        ))

    logger.info(
        "reranker.complete",
        input_candidates=len(candidates),
        output_top_k=len(reranked),
        top_score=round(reranked[0].score, 6) if reranked else 0.0,
        bottom_score=round(reranked[-1].score, 6) if reranked else 0.0,
    )

    return reranked
