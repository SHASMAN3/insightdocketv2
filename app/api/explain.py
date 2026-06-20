"""
GET /api/v1/explain/{request_id} — Return chunk-level source breakdown for a past query.

Interview note: The /explain endpoint is the explainability layer —
it answers "why did you return that answer?" by exposing:
  - Which chunks were retrieved and their scores
  - Which pages they came from
  - Whether confidence threshold was met
  - Whether injection was detected

This is critical for enterprise RAG deployments where auditors need to
trace every answer back to its source evidence.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, status

from app.db.mongodb import get_chunks_collection
from app.dependencies import AuthDep
from app.observability.audit_logger import get_audit_log

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/explain/{request_id}",
    summary="Get chunk-level source breakdown for a past query",
)
async def explain_query(
    request_id: str,
    api_key: AuthDep,
) -> dict:
    """
    Return the full retrieval breakdown for a past query identified by request_id.

    Fetches the audit log from MySQL, then hydrates the full chunk content
    from MongoDB for each chunk_id in the log.

    Returns chunk scores, page numbers, chunk types, and content previews.
    """
    # ── Fetch audit log from MySQL ─────────────────────────────
    audit_entry = await get_audit_log(request_id)
    if audit_entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No audit log found for request_id: {request_id!r}",
        )

    # ── Hydrate chunks from MongoDB ────────────────────────────
    chunk_ids: list[str] = audit_entry.get("chunk_ids") or []
    chunk_details: list[dict] = []

    if chunk_ids:
        collection = get_chunks_collection()
        cursor = collection.find(
            {"_id": {"$in": chunk_ids}},
            projection={
                "_id": 1,
                "content": 1,
                "chunk_type": 1,
                "document_name": 1,
                "page_number": 1,
                "chunk_index": 1,
                "version": 1,
                "metadata": 1,
            },
        )

        mongo_chunks: dict[str, dict] = {}
        async for doc in cursor:
            mongo_chunks[str(doc["_id"])] = doc

        # Preserve the order from chunk_ids (which is rank order)
        for i, cid in enumerate(chunk_ids):
            doc = mongo_chunks.get(cid, {})
            chunk_details.append({
                "rank": i + 1,
                "chunk_id": cid,
                "chunk_type": doc.get("chunk_type", "unknown"),
                "document_name": doc.get("document_name", "unknown"),
                "page_number": doc.get("page_number", None),
                "chunk_index": doc.get("chunk_index", None),
                "version": doc.get("version", None),
                "content_preview": (doc.get("content", "")[:300] + "...") if doc.get("content") else "[not found]",
                "section": (doc.get("metadata") or {}).get("section"),
            })

    logger.info(
        "explain.request",
        request_id=request_id,
        chunks_hydrated=len(chunk_details),
        api_key_name=api_key.name,
    )

    return {
        "request_id": request_id,
        "query": audit_entry.get("question"),
        "response_type": audit_entry.get("response_type"),
        "confidence_score": audit_entry.get("confidence_score"),
        "injection_detected": audit_entry.get("injection_detected"),
        "retrieval_latency_ms": audit_entry.get("retrieval_latency_ms"),
        "generation_latency_ms": audit_entry.get("generation_latency_ms"),
        "total_latency_ms": audit_entry.get("total_latency_ms"),
        "page_numbers_retrieved": audit_entry.get("page_numbers", []),
        "chunks": chunk_details,
        "answer_preview": (audit_entry.get("answer") or "")[:500],
        "created_at": audit_entry.get("created_at"),
    }
