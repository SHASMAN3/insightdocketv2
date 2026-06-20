"""
Audit logger — writes every query event to MySQL audit_logs + daily JSONL backup.

Interview note: Dual-write (MySQL + JSONL) provides defence in depth:
  - MySQL: structured queries, joins with document table, KPI dashboards
  - JSONL: human-readable, grep-able, survives DB failure, easy to ship
    to Splunk/ELK/CloudWatch without schema migration

The JSONL file rotates daily: logs/audit_2024-01-15.jsonl
Writes are async (MySQL via SQLAlchemy, JSONL via aiofiles pattern).
Both writes are best-effort — a write failure is logged but never
bubbles up to the API caller.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog
from sqlalchemy import select

from app.config import get_settings
from app.db.models import AuditLog
from app.db.mysql import get_db_session

logger = structlog.get_logger(__name__)


async def write_audit_log(
    request_id: str,
    question: str,
    answer: Optional[str],
    response_type: str,
    confidence_score: Optional[float],
    retrieval_latency_ms: Optional[int],
    generation_latency_ms: Optional[int],
    total_latency_ms: int,
    chunk_ids: Optional[list[str]],
    page_numbers: Optional[list[int]],
    injection_detected: bool,
    document_id: Optional[int] = None,
) -> None:
    """
    Write an audit log entry to MySQL and the daily JSONL backup file.

    Called at the end of every /query request regardless of outcome.
    All failures are caught and logged — never raised to caller.
    """
    settings = get_settings()

    audit_data = {
        "request_id": request_id,
        "document_id": document_id,
        "question": question,
        "answer": answer,
        "response_type": response_type,
        "confidence_score": confidence_score,
        "retrieval_latency_ms": retrieval_latency_ms,
        "generation_latency_ms": generation_latency_ms,
        "total_latency_ms": total_latency_ms,
        "chunk_ids": chunk_ids or [],
        "page_numbers": page_numbers or [],
        "injection_detected": injection_detected,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    # ── MySQL write ────────────────────────────────────────────
    await _write_mysql(audit_data)

    # ── JSONL backup write ─────────────────────────────────────
    _write_jsonl(audit_data, settings.audit_log_dir)


async def _write_mysql(data: dict) -> None:
    """Insert one row into audit_logs. Best-effort — swallows exceptions."""
    try:
        async with get_db_session() as session:
            log_entry = AuditLog(
                request_id=data["request_id"],
                document_id=data.get("document_id"),
                question=data["question"][:4000],   # MySQL TEXT has practical limit
                answer=(data.get("answer") or "")[:8000],
                response_type=data["response_type"],
                confidence_score=data.get("confidence_score"),
                retrieval_latency_ms=data.get("retrieval_latency_ms"),
                generation_latency_ms=data.get("generation_latency_ms"),
                total_latency_ms=data["total_latency_ms"],
                chunk_ids=data.get("chunk_ids"),
                page_numbers=data.get("page_numbers"),
                injection_detected=data["injection_detected"],
            )
            session.add(log_entry)
        logger.debug("audit.mysql_written", request_id=data["request_id"])
    except Exception as exc:
        logger.error("audit.mysql_failed", request_id=data["request_id"], error=str(exc))


def _write_jsonl(data: dict, log_dir: Path) -> None:
    """
    Append one JSON line to the daily audit log file.

    File: {log_dir}/audit_{YYYY-MM-DD}.jsonl
    Uses synchronous file write — JSONL write is fast and non-blocking
    enough for audit purposes. For high-throughput, use asyncio file IO.
    """
    try:
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        log_file = log_dir / f"audit_{today}.jsonl"

        line = json.dumps(data, default=str, ensure_ascii=False)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        logger.debug("audit.jsonl_written", file=str(log_file))
    except Exception as exc:
        logger.error("audit.jsonl_failed", error=str(exc))


async def get_audit_log(request_id: str) -> Optional[dict]:
    """
    Retrieve an audit log entry by request_id from MySQL.

    Used by the /explain endpoint to surface retrieval details.
    """
    try:
        async with get_db_session() as session:
            result = await session.execute(
                select(AuditLog).where(AuditLog.request_id == request_id)
            )
            entry: Optional[AuditLog] = result.scalar_one_or_none()

        if entry is None:
            return None

        return {
            "request_id": entry.request_id,
            "document_id": entry.document_id,
            "question": entry.question,
            "answer": entry.answer,
            "response_type": entry.response_type,
            "confidence_score": entry.confidence_score,
            "retrieval_latency_ms": entry.retrieval_latency_ms,
            "generation_latency_ms": entry.generation_latency_ms,
            "total_latency_ms": entry.total_latency_ms,
            "chunk_ids": entry.chunk_ids or [],
            "page_numbers": entry.page_numbers or [],
            "injection_detected": entry.injection_detected,
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
        }
    except Exception as exc:
        logger.error("audit.get_failed", request_id=request_id, error=str(exc))
        return None
