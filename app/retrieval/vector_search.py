"""
MongoDB $vectorSearch retrieval.

Interview note: $vectorSearch is MongoDB 7.0's native vector similarity operator.
It runs as an aggregation pipeline stage, allowing us to combine semantic
search with $text (BM25) and scalar filters in a single query — no second
service, no cross-service network hop, no synchronisation lag.

The index is an HNSW index on the embedding field (768 floats for
text-embedding-004). HNSW provides sub-linear approximate nearest neighbour
search — exact kNN would be O(n) at 1M+ documents.

numCandidates = top_k * 10 is the standard tuning: fetch 10x candidates
from HNSW for rescoring, then return top_k. Higher numCandidates = better
recall at the cost of more RAM. 200 candidates for top_k=20 is a safe default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog

from app.config import get_settings
from app.db.mongodb import get_chunks_collection

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """A single retrieved chunk with its relevance score."""
    chunk_id: str
    content: str
    chunk_type: str
    document_id: int
    document_name: str
    page_number: int
    chunk_index: int
    version: int
    score: float
    metadata: dict


async def vector_search(
    query_embedding: list[float],
    top_k: int | None = None,
    document_id: Optional[int] = None,
) -> list[SearchResult]:
    """
    Run $vectorSearch against MongoDB and return top_k ranked results.

    Args:
        query_embedding: 768-float query embedding from embed_query()
        top_k: number of results to return (defaults to settings.vector_top_k)
        document_id: optional filter — restrict search to a specific document

    Returns list[SearchResult] sorted by descending cosine similarity score.

    Interview note: $vectorSearch must be the FIRST stage in the aggregation
    pipeline. You cannot filter BEFORE $vectorSearch — only $match AFTER it.
    This is a MongoDB constraint, not a design choice.
    """
    settings = get_settings()
    collection = get_chunks_collection()
    k = top_k or settings.vector_top_k

    # numCandidates must be >= limit and is capped by index size
    num_candidates = min(k * 10, 1000)

    # Build optional pre-filter (applied within HNSW, not after)
    # Pre-filters narrow the candidate set before ANN search
    pre_filter: dict = {}
    if document_id is not None:
        pre_filter = {"document_id": {"$eq": document_id}}

    vector_stage: dict = {
        "$vectorSearch": {
            "index": settings.mongodb_vector_index,
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": num_candidates,
            "limit": k,
        }
    }

    # Add filter only if specified — empty filter causes MongoDB error
    if pre_filter:
        vector_stage["$vectorSearch"]["filter"] = pre_filter

    pipeline = [
        vector_stage,
        {
            "$project": {
                "_id": 1,
                "content": 1,
                "chunk_type": 1,
                "document_id": 1,
                "document_name": 1,
                "page_number": 1,
                "chunk_index": 1,
                "version": 1,
                "metadata": 1,
                # $vectorSearch score is accessed via $meta
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        cursor = collection.aggregate(pipeline)
        results: list[SearchResult] = []
        async for doc in cursor:
            results.append(SearchResult(
                chunk_id=str(doc["_id"]),
                content=doc.get("content", ""),
                chunk_type=doc.get("chunk_type", "text"),
                document_id=doc.get("document_id", 0),
                document_name=doc.get("document_name", ""),
                page_number=doc.get("page_number", 1),
                chunk_index=doc.get("chunk_index", 0),
                version=doc.get("version", 1),
                score=float(doc.get("score", 0.0)),
                metadata=doc.get("metadata", {}),
            ))

        logger.info(
            "vector_search.complete",
            results_count=len(results),
            top_k=k,
            document_id=document_id,
        )
        return results

    except Exception as exc:
        logger.error("vector_search.failed", error=str(exc))
        # Return empty list rather than crashing — BM25 fallback still runs
        return []
