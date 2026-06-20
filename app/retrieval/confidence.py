"""
Confidence scoring and fallback threshold logic.

Interview note: Confidence scoring addresses a fundamental RAG failure mode —
the system answering confidently from weakly-relevant chunks.

Two-stage approach:
  1. Confidence score = weighted average of top-k reranker scores
     (top chunk weighted more heavily — 40% of final score)
  2. If confidence < threshold → structured "insufficient information" response
     rather than a potentially hallucinated answer

This is preferable to always generating an answer because:
  - Low-confidence answers erode user trust faster than honest fallbacks
  - Audit logs capture fallback rate — a KPI for retrieval quality
  - Operators can tune the threshold based on their risk tolerance

Threshold = 0.35 by default. Lower = more permissive (higher hallucination risk).
Higher = more conservative (higher fallback rate).
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from app.config import get_settings
from app.retrieval.vector_search import SearchResult

logger = structlog.get_logger(__name__)


@dataclass
class ConfidenceResult:
    """Output of confidence scoring."""
    score: float                    # 0.0 – 1.0
    above_threshold: bool           # True → proceed to generation
    top_chunks: list[SearchResult]  # Final chunks to pass to LLM
    reasoning: str                  # Human-readable explanation


def compute_confidence(
    reranked_chunks: list[SearchResult],
    threshold: float | None = None,
) -> ConfidenceResult:
    """
    Compute a confidence score from reranked chunk scores.

    Scoring formula:
      - Top chunk contributes 40% of the final score
      - Remaining chunks contribute 60% equally weighted
      - If fewer than 3 chunks available, score is penalised by 0.1 per missing chunk

    The formula is intentionally simple so it's explainable in interviews:
    "The top chunk is most predictive of answer quality, so we weight it higher."

    Args:
        reranked_chunks: output from reranker.rerank(), sorted by score desc
        threshold: override for settings.confidence_threshold

    Returns ConfidenceResult with the aggregate score and above_threshold flag.
    """
    settings = get_settings()
    effective_threshold = threshold if threshold is not None else settings.confidence_threshold

    if not reranked_chunks:
        return ConfidenceResult(
            score=0.0,
            above_threshold=False,
            top_chunks=[],
            reasoning="No chunks retrieved — collection may be empty or query has no matches.",
        )

    scores = [c.score for c in reranked_chunks]

    # Weighted average: top chunk = 40%, rest = 60% equally divided
    top_weight = 0.40
    rest_weight = 0.60

    if len(scores) == 1:
        raw_score = scores[0]
    else:
        top_score = scores[0] * top_weight
        rest_avg = sum(scores[1:]) / len(scores[1:])
        raw_score = top_score + rest_avg * rest_weight

    # Penalise if fewer than 3 chunks (suggests sparse retrieval)
    min_expected = 3
    if len(reranked_chunks) < min_expected:
        penalty = 0.1 * (min_expected - len(reranked_chunks))
        raw_score = max(0.0, raw_score - penalty)

    confidence = round(min(1.0, max(0.0, raw_score)), 4)
    above_threshold = confidence >= effective_threshold

    reasoning = (
        f"Confidence {confidence:.2%} from {len(reranked_chunks)} chunks "
        f"(top score: {scores[0]:.4f}). "
        f"Threshold: {effective_threshold:.2%}. "
        f"{'Proceeding to generation.' if above_threshold else 'Falling back — confidence below threshold.'}"
    )

    logger.info(
        "confidence.computed",
        score=confidence,
        threshold=effective_threshold,
        above_threshold=above_threshold,
        chunk_count=len(reranked_chunks),
        top_chunk_score=round(scores[0], 4),
    )

    return ConfidenceResult(
        score=confidence,
        above_threshold=above_threshold,
        top_chunks=reranked_chunks,
        reasoning=reasoning,
    )


def build_fallback_response(query: str, confidence_result: ConfidenceResult) -> dict:
    """
    Build a structured fallback response when confidence is below threshold.

    Returns a dict that the /query endpoint returns directly to the caller.
    The response_type="fallback" is logged in audit_logs for KPI tracking.
    """
    return {
        "answer": (
            "I was unable to find sufficiently relevant information in the ingested documents "
            "to answer your question confidently. Please ensure the relevant document has been "
            "ingested, or try rephrasing your question."
        ),
        "response_type": "fallback",
        "confidence_score": confidence_result.score,
        "confidence_reasoning": confidence_result.reasoning,
        "sources": [],
        "query": query,
    }
