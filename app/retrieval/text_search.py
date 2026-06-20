"""
MongoDB $text BM25 full-text search.

Interview note: $text uses an inverted index with TF-IDF/BM25 scoring.
It excels at keyword-heavy queries where vector search underperforms:
  - Exact product codes: "SKU-1234-XL"
  - Proper nouns: "Rajasthan High Court judgment dated 15 March 2023"
  - Technical terms not well-represented in embedding space

RRF fusion combines both signals so we get the best of both worlds.
~12% recall improvement over pure vector search on keyword-heavy queries
(based on BEIR benchmark results for hybrid vs dense-only retrieval).

$text search requires a TEXT index on the content field — created in
mongodb.py at startup. The score is accessed via {"$meta": "textScore"}.
"""

from __future__ import annotations

from typing import Optional

import structlog

from app.config import get_settings
from app.db.mongodb import get_chunks_collection
from app.retrieval.vector_search import SearchResult

logger = structlog.get_logger(__name__)


async def text_search(
    query: str,
    top_k: int | None = None,
    document_id: Optional[int] = None,
) -> list[SearchResult]:
    """
    Run MongoDB $text BM25 search and return top_k ranked results.

    Args:
        query: raw user query string (NOT the embedding — this is keyword search)
        top_k: number of results to return (defaults to settings.text_top_k)
        document_id: optional filter — restrict search to a specific document

    Returns list[SearchResult] sorted by descending textScore.

    Interview note: $text does NOT support regex or phrase search by default.
    Use $text for keyword recall and vector search for semantic recall — they
    are complementary, not redundant.
    """
    settings = get_settings()
    collection = get_chunks_collection()
    k = top_k or settings.text_top_k

    # Build $match stage — document filter + text search
    match_stage: dict = {"$text": {"$search": query}}
    if document_id is not None:
        match_stage["document_id"] = document_id

    pipeline = [
        {"$match": match_stage},
        {
            "$addFields": {
                "score": {"$meta": "textScore"},
            }
        },
        # Sort by textScore descending
        {"$sort": {"score": -1}},
        {"$limit": k},
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
                "score": 1,
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
            "text_search.complete",
            results_count=len(results),
            top_k=k,
            query_preview=query[:50],
        )
        return results

    except Exception as exc:
        # $text search fails if the query contains special characters or if
        # the text index doesn't exist yet. Log and return empty — vector
        # search results will still be used via RRF.
        logger.warning("text_search.failed", error=str(exc), query_preview=query[:50])
        return []
