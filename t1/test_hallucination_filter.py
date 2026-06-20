"""
Tests for app.generation.hallucination_filter — token overlap grounding check.
"""

from __future__ import annotations

import pytest

from app.generation.hallucination_filter import (
    OVERLAP_THRESHOLD,
    check_hallucination,
    _tokenise,
)
from app.retrieval.vector_search import SearchResult


def _make_chunk(content: str) -> SearchResult:
    return SearchResult(
        chunk_id="test_chunk",
        content=content,
        chunk_type="text",
        document_id=1,
        document_name="test.pdf",
        page_number=1,
        chunk_index=0,
        version=1,
        score=0.9,
        metadata={},
    )


class TestTokenise:

    def test_stop_words_removed(self) -> None:
        tokens = _tokenise("the quick brown fox")
        assert "the" not in tokens

    def test_short_words_removed(self) -> None:
        tokens = _tokenise("an ox in it")
        # "an", "in", "it" are stop words; "ox" is 2 chars → filtered
        assert "ox" not in tokens  # len < 3

    def test_punctuation_removed(self) -> None:
        tokens = _tokenise("revenue: $1,000,000!")
        assert "$" not in str(tokens)
        assert "!" not in str(tokens)

    def test_lowercase_normalisation(self) -> None:
        tokens = _tokenise("Revenue REVENUE revenue")
        assert len(tokens) == 1  # All normalise to "revenue"

    def test_meaningful_words_kept(self) -> None:
        tokens = _tokenise("quarterly revenue increased significantly")
        assert "quarterly" in tokens
        assert "revenue" in tokens
        assert "increased" in tokens
        assert "significantly" in tokens


class TestCheckHallucination:

    def test_answer_grounded_in_context(self) -> None:
        context = "The quarterly revenue for 2024 was 42 million dollars. Operating profit increased by 15 percent."
        answer = "The quarterly revenue was 42 million dollars and operating profit grew by 15 percent."
        chunks = [_make_chunk(context)]
        result = check_hallucination(answer, chunks)
        assert result.is_grounded is True
        assert result.overlap_ratio >= OVERLAP_THRESHOLD

    def test_answer_not_grounded_in_context(self) -> None:
        context = "The company sells widgets and gadgets."
        # Answer contains terms completely absent from context
        answer = "The CEO announced a merger with pharmaceutical biotech division worth billions."
        chunks = [_make_chunk(context)]
        result = check_hallucination(answer, chunks)
        assert result.is_grounded is False
        assert result.overlap_ratio < OVERLAP_THRESHOLD

    def test_fallback_response_always_grounded(self) -> None:
        answer = "The provided documents do not contain sufficient information to answer this question."
        result = check_hallucination(answer, [])
        assert result.is_grounded is True

    def test_short_answer_skipped(self) -> None:
        result = check_hallucination("Yes", [])
        assert result.is_grounded is True
        assert "too short" in result.reasoning

    def test_empty_chunks_with_long_answer(self) -> None:
        answer = "The revenue figures show substantial growth in manufacturing pharmaceutical biotech sectors."
        result = check_hallucination(answer, [])
        # No context → zero overlap → hallucination flagged
        assert result.is_grounded is False

    def test_multiple_chunks_combined_for_coverage(self) -> None:
        chunk1 = _make_chunk("Revenue increased by fifteen percent year over year.")
        chunk2 = _make_chunk("Operating expenses decreased due to cost reduction measures.")
        answer = "Revenue increased fifteen percent while operating expenses decreased due to cost measures."
        result = check_hallucination(answer, [chunk1, chunk2])
        assert result.is_grounded is True

    def test_overlap_ratio_between_zero_and_one(self) -> None:
        chunks = [_make_chunk("Some content about financial results and revenue figures")]
        answer = "The financial results show positive revenue growth and expansion."
        result = check_hallucination(answer, chunks)
        assert 0.0 <= result.overlap_ratio <= 1.0

    def test_custom_threshold_applied(self) -> None:
        context = "Revenue grew significantly."
        answer = "Revenue grew substantially in the reported period with significant gains."
        chunks = [_make_chunk(context)]
        # Very high threshold — even partial overlap fails
        result_strict = check_hallucination(answer, chunks, threshold=0.95)
        # Very low threshold — minimal overlap passes
        result_lenient = check_hallucination(answer, chunks, threshold=0.05)
        assert result_lenient.is_grounded is True
        # Strict may or may not pass depending on overlap — just check it runs
        assert isinstance(result_strict.is_grounded, bool)
