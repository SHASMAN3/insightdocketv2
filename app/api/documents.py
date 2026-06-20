"""
GET /api/v1/documents — List all ingested documents with their version history.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.db.models import Document, DocumentVersion
from app.db.mysql import get_db_session
from app.dependencies import AuthDep

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/documents",
    summary="List all ingested documents",
)
async def list_documents(
    api_key: AuthDep,
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    status: str | None = Query(default=None, description="Filter by status: active|processing|failed"),
) -> dict:
    """
    Return paginated list of all ingested documents with version history.

    Each document includes its latest version status and a list of all versions.
    """
    async with get_db_session() as session:
        # Build base query
        base_query = select(Document)
        if status:
            base_query = base_query.where(Document.status == status)

        # Count total
        count_query = select(func.count()).select_from(Document)
        if status:
            count_query = count_query.where(Document.status == status)
        total_result = await session.execute(count_query)
        total = total_result.scalar_one()

        # Paginate
        offset = (page - 1) * page_size
        docs_result = await session.execute(
            base_query.order_by(Document.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        documents: list[Document] = list(docs_result.scalars().all())

        # Fetch version history for each document
        doc_ids = [d.id for d in documents]
        versions_result = await session.execute(
            select(DocumentVersion)
            .where(DocumentVersion.document_id.in_(doc_ids))
            .order_by(DocumentVersion.document_id, DocumentVersion.version.desc())
        )
        all_versions: list[DocumentVersion] = list(versions_result.scalars().all())

    # Group versions by document_id
    versions_by_doc: dict[int, list[dict]] = {}
    for v in all_versions:
        versions_by_doc.setdefault(v.document_id, []).append({
            "version": v.version,
            "ingest_status": v.ingest_status,
            "chunk_count": v.chunk_count,
            "page_count": v.page_count,
            "error_message": v.error_message,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        })

    doc_list = []
    for doc in documents:
        doc_list.append({
            "id": doc.id,
            "filename": doc.filename,
            "s3_key": doc.s3_key,
            "current_version": doc.version,
            "status": doc.status,
            "page_count": doc.page_count,
            "chunk_count": doc.chunk_count,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
            "versions": versions_by_doc.get(doc.id, []),
        })

    return {
        "documents": doc_list,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
    }
