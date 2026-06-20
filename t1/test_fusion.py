"""
Tests for app.retrieval.fusion — Reciprocal Rank Fusion correctness.
"""

from __future__ import annotations

import pytest

from app.retrieval.fusion import RRF_K, reciprocal_rank_fusion
from app.retrieval.vector_search import SearchResult


def _make_result(chunk_id: str, score: float = 1.0) -> SearchResult:
    """Helper to create a minimal SearchResult for testing."""
    return SearchResult(
        chunk_id=chunk_id,
        content=f"Content for {chunk_id}",
        chunk_type="text",
        document_id=1,
        document_name="test.pdf",
        page_number=1,
        chunk_index=0,
        version=1,
        score=score,
        metadata={},
    )


class TestReciprocalRankFusion:
    """Verify RRF score calculation and result ordering."""

    def test_rrf_formula_correctness(self) -> None:
        """
        Manual calculation: with 2 results in each list:
          chunk_A at rank 1 in vector → 1/(60+1) ≈ 0.016393
          chunk_B at rank 1 in text   → 1/(60+1) ≈ 0.016393
          chunk_A at rank 2 in text   → 1/(60+2) ≈ 0.016129
          chunk_A total = 0.016393 + 0.016129 = 0.032522
          chunk_B total = 0.016393
        So chunk_A should rank above chunk_B.
        """
        vector_results = [_make_result("chunk_A"), _make_result("chunk_C")]
        text_results = [_make_result("chunk_B"), _make_result("chunk_A")]

        fused = reciprocal_rank_fusion(vector_results, text_results, top_k=10)

        chunk_ids = [r.chunk_id for r in fused]
        assert chunk_ids[0] == "chunk_A", "chunk_A should rank first (appears in both lists)"
        assert "chunk_B" in chunk_ids
        assert "chunk_C" in chunk_ids

    def test_chunk_appearing_in_both_lists_ranked_higher(self) -> None:
        """A chunk in both vector and text results should outrank chunks in only one."""
        overlap_id = "overlap_chunk"
        vector_only_id = "vector_only"
        text_only_id = "text_only"

        vector_results = [
            _make_result(overlap_id),
            _make_result(vector_only_id),
        ]
        text_results = [
            _make_result(overlap_id),
            _make_result(text_only_id),
        ]

        fused = reciprocal_rank_fusion(vector_results, text_results, top_k=10)
        assert fused[0].chunk_id == overlap_id

    def test_top_k_limits_results(self) -> None:
        vector_results = [_make_result(f"v{i}") for i in range(10)]
        text_results = [_make_result(f"t{i}") for i in range(10)]

        fused = reciprocal_rank_fusion(vector_results, text_results, top_k=5)
        assert len(fused) == 5

    def test_empty_vector_results_uses_text_only(self) -> None:
        text_results = [_make_result("text_a"), _make_result("text_b")]
        fused = reciprocal_rank_fusion([], text_results, top_k=10)
        assert len(fused) == 2
        assert fused[0].chunk_id == "text_a"  # Rank 1 in text → highest score

    def test_empty_text_results_uses_vector_only(self) -> None:
        vector_results = [_make_result("vec_a"), _make_result("vec_b")]
        fused = reciprocal_rank_fusion(vector_results, [], top_k=10)
        assert len(fused) == 2
        assert fused[0].chunk_id == "vec_a"

    def test_both_empty_returns_empty(self) -> None:
        fused = reciprocal_rank_fusion([], [], top_k=10)
        assert fused == []

    def test_scores_are_positive_and_bounded(self) -> None:
        vector_results = [_make_result(f"c{i}") for i in range(5)]
        text_results = [_make_result(f"c{i}") for i in range(5)]
        fused = reciprocal_rank_fusion(vector_results, text_results, top_k=10)
        for result in fused:
            assert result.score > 0.0
            # Max possible RRF score = 2 * 1/(60+1) ≈ 0.033 for rank-1 in both
            assert result.score <= 2.0 / (RRF_K + 1) + 0.001  # small float tolerance

    def test_score_strictly_descending(self) -> None:
        vector_results = [_make_result(f"v{i}") for i in range(5)]
        text_results = [_make_result(f"t{i}") for i in range(5)]
        fused = reciprocal_rank_fusion(vector_results, text_results, top_k=10)
        scores = [r.score for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_weights_applied(self) -> None:
        """Higher vector_weight should push vector-only chunks up relative to text-only."""
        vec_result = [_make_result("vec_exclusive")]
        text_result = [_make_result("text_exclusive")]

        fused_vector_weighted = reciprocal_rank_fusion(
            vec_result, text_result, top_k=2, vector_weight=2.0, text_weight=1.0
        )
        assert fused_vector_weighted[0].chunk_id == "vec_exclusive"

        fused_text_weighted = reciprocal_rank_fusion(
            vec_result, text_result, top_k=2, vector_weight=1.0, text_weight=2.0
        )
        assert fused_text_weighted[0].chunk_id == "text_exclusive"
