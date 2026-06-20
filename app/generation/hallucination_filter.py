"""
Hallucination filter — verifies answer tokens appear in retrieved context.

Interview note: Even with a strict grounded prompt, LLMs occasionally
"slip" by drawing on parametric memory for specific numbers, names, or dates.
This filter provides a second check after generation.

Approach: token-overlap (Jaccard-inspired)
  1. Tokenise the generated answer into meaningful terms (stop words removed)
  2. Tokenise the combined retrieved context
  3. Compute overlap ratio = |answer_terms ∩ context_terms| / |answer_terms|
  4. If overlap < OVERLAP_THRESHOLD → flag as potentially hallucinated

This is intentionally conservative — false positives (over-flagging) are
preferable to false negatives (missing hallucinations). A flagged answer
is still returned but tagged with hallucination_detected=True in the audit log.

Limitations acknowledged in interviews:
  - Token overlap can't catch paraphrasing hallucinations
  - LLM-as-judge (using a second Gemini call) would be more accurate
    but doubles cost/latency — a reasonable follow-up improvement
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

import structlog

from app.retrieval.vector_search import SearchResult

logger = structlog.get_logger(__name__)

# Tokens below this frequency in English corpora are considered meaningful
# Common stop words — conservative list to avoid over-filtering
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "this",
    "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "when", "where", "why", "how", "all", "each",
    "not", "no", "so", "if", "then", "than", "its", "their", "our", "your",
    "also", "into", "about", "there", "use", "used", "using",
})

# Overlap below this threshold triggers hallucination flag
OVERLAP_THRESHOLD = 0.40


@dataclass
class HallucinationCheckResult:
    """Result of hallucination check."""
    is_grounded: bool           # True = likely grounded, False = potentially hallucinated
    overlap_ratio: float        # |answer ∩ context| / |answer_terms|
    answer_term_count: int
    matched_term_count: int
    reasoning: str


def _tokenise(text: str) -> set[str]:
    """
    Tokenise text into meaningful lowercase terms, removing stop words.

    Returns a set (not list) — we care about presence, not frequency.
    """
    # Lowercase and remove punctuation
    cleaned = text.lower()
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    # Split on whitespace, filter short tokens and stop words
    tokens = {
        word for word in cleaned.split()
        if len(word) >= 3 and word not in _STOP_WORDS
    }
    return tokens


def check_hallucination(
    answer: str,
    chunks: list[SearchResult],
    threshold: float = OVERLAP_THRESHOLD,
) -> HallucinationCheckResult:
    """
    Check whether the generated answer is grounded in the retrieved context.

    Args:
        answer: LLM-generated answer string
        chunks: retrieved chunks used as context for generation
        threshold: minimum overlap ratio to consider answer grounded

    Returns HallucinationCheckResult with grounding verdict and diagnostics.
    """
    # Skip check for special responses
    if not answer or len(answer) < 20:
        return HallucinationCheckResult(
            is_grounded=True,
            overlap_ratio=1.0,
            answer_term_count=0,
            matched_term_count=0,
            reasoning="Answer too short to meaningfully check — skipping.",
        )

    if "do not contain sufficient information" in answer.lower():
        return HallucinationCheckResult(
            is_grounded=True,
            overlap_ratio=1.0,
            answer_term_count=0,
            matched_term_count=0,
            reasoning="Fallback response — no grounding check needed.",
        )

    # Build context vocabulary from all retrieved chunks
    context_text = " ".join(chunk.content for chunk in chunks)
    context_terms = _tokenise(context_text)

    # Tokenise the answer
    answer_terms = _tokenise(answer)

    if not answer_terms:
        return HallucinationCheckResult(
            is_grounded=True,
            overlap_ratio=1.0,
            answer_term_count=0,
            matched_term_count=0,
            reasoning="Answer produced no meaningful tokens after stop-word removal.",
        )

    # Compute overlap
    matched_terms = answer_terms & context_terms
    overlap_ratio = len(matched_terms) / len(answer_terms)

    is_grounded = overlap_ratio >= threshold

    reasoning = (
        f"Overlap ratio: {overlap_ratio:.2%} "
        f"({len(matched_terms)}/{len(answer_terms)} terms matched in context). "
        f"Threshold: {threshold:.2%}. "
        f"{'Grounded.' if is_grounded else 'POTENTIAL HALLUCINATION DETECTED.'}"
    )

    if not is_grounded:
        # Log unmatched terms for debugging
        unmatched = sorted(answer_terms - context_terms)[:10]
        logger.warning(
            "hallucination.detected",
            overlap_ratio=round(overlap_ratio, 4),
            threshold=threshold,
            sample_unmatched_terms=unmatched,
        )
    else:
        logger.debug(
            "hallucination.grounded",
            overlap_ratio=round(overlap_ratio, 4),
            matched=len(matched_terms),
            total=len(answer_terms),
        )

    return HallucinationCheckResult(
        is_grounded=is_grounded,
        overlap_ratio=round(overlap_ratio, 4),
        answer_term_count=len(answer_terms),
        matched_term_count=len(matched_terms),
        reasoning=reasoning,
    )
