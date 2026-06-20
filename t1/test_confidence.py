"""
Tests for app.retrieval.confidence — confidence scoring and fallback logic.
"""

from __future__ import annotations

import pytest

from app.retrieval.confidence import (
    build_fallback_response,
    compute_confidence,
)
from app.retrieval.vector_search import SearchResult


def _make_chunk(chunk_id: str, score: float, page: int = 1) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        content="Sample content for testing",
        chunk_type="text",
        document_id=1,
        document_name="test.pdf",
        page_number=page,
        chunk_index=0,
        version=1,
        score=score,
        metadata={},
    )


class TestComputeConfidence:

    def test_empty_chunks_returns_zero(self) -> None:
        result = compute_confidence([])
        assert result.score == 0.0
        assert result.above_threshold is False
        assert "No chunks" in result.reasoning

    def test_single_high_score_chunk_above_threshold(self) -> None:
        chunks = [_make_chunk("c1", score=0.9)]
        result = compute_confidence(chunks, threshold=0.35)
        # Single chunk — penalised for < 3 chunks: 0.9 - 0.2 = 0.7
        assert result.score >= 0.35
        assert result.above_threshold is True

    def test_single_low_score_chunk_below_threshold(self) -> None:
        chunks = [_make_chunk("c1", score=0.1)]
        result = compute_confidence(chunks, threshold=0.35)
        assert result.above_threshold is False

    def test_three_high_score_chunks_above_threshold(self) -> None:
        chunks = [
            _make_chunk("c1", score=0.85),
            _make_chunk("c2", score=0.80),
            _make_chunk("c3", score=0.75),
        ]
        result = compute_confidence(chunks, threshold=0.35)
        assert result.score > 0.35
        assert result.above_threshold is True

    def test_three_low_score_chunks_below_threshold(self) -> None:
        chunks = [
            _make_chunk("c1", score=0.2),
            _make_chunk("c2", score=0.15),
            _make_chunk("c3", score=0.1),
        ]
        result = compute_confidence(chunks, threshold=0.35)
        assert result.above_threshold is False

    def test_score_bounded_zero_to_one(self) -> None:
        chunks = [_make_chunk("c1", score=0.99) for _ in range(5)]
        result = compute_confidence(chunks, threshold=0.0)
        assert 0.0 <= result.score <= 1.0

    def test_penalty_applied_for_fewer_than_three_chunks(self) -> None:
        chunks_3 = [_make_chunk(f"c{i}", score=0.8) for i in range(3)]
        chunks_1 = [_make_chunk("c0", score=0.8)]
        score_3 = compute_confidence(chunks_3).score
        score_1 = compute_confidence(chunks_1).score
        # 1 chunk should be penalised more than 3 chunks
        assert score_1 < score_3

    def test_custom_threshold_respected(self) -> None:
        chunks = [_make_chunk("c1", score=0.5) for _ in range(3)]
        # Above threshold=0.3
        result_low = compute_confidence(chunks, threshold=0.3)
        assert result_low.above_threshold is True
        # Below threshold=0.9
        result_high = compute_confidence(chunks, threshold=0.9)
        assert result_high.above_threshold is False

    def test_reasoning_string_populated(self) -> None:
        chunks = [_make_chunk("c1", score=0.7) for _ in range(3)]
        result = compute_confidence(chunks)
        assert len(result.reasoning) > 20
        assert "%" in result.reasoning

    def test_top_chunks_returned(self) -> None:
        chunks = [_make_chunk(f"c{i}", score=0.8 - i * 0.1) for i in range(5)]
        result = compute_confidence(chunks)
        assert result.top_chunks == chunks


class TestBuildFallbackResponse:

    def test_fallback_response_structure(self) -> None:
        from app.retrieval.confidence import ConfidenceResult
        cr = ConfidenceResult(
            score=0.2,
            above_threshold=False,
            top_chunks=[],
            reasoning="Low confidence.",
        )
        response = build_fallback_response("What is the revenue?", cr)
        assert response["response_type"] == "fallback"
        assert response["confidence_score"] == 0.2
        assert isinstance(response["answer"], str)
        assert len(response["answer"]) > 20
        assert response["sources"] == []

    def test_fallback_answer_is_informative(self) -> None:
        from app.retrieval.confidence import ConfidenceResult
        cr = ConfidenceResult(score=0.1, above_threshold=False, top_chunks=[], reasoning="")
        response = build_fallback_response("test query", cr)
        # Should NOT be an empty string
        assert response["answer"].strip() != ""
        # Should guide the user
        assert any(word in response["answer"].lower() for word in ["information", "document", "rephrase"])
