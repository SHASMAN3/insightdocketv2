"""
SQLAlchemy ORM models for InsightDocket MySQL tables.

Interview note: MySQL with SQLAlchemy provides ACID guarantees for
document versioning and audit trails — things MongoDB's eventual
consistency model isn't suited for. Vector similarity lives in Mongo;
transactional records live here.

Tables:
  - documents          : canonical document registry
  - document_versions  : full version history per document
  - audit_logs         : immutable query audit trail
  - api_keys           : hashed API credentials with rate limits
"""

from __future__ import annotations

import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    Index,
    Column
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


class Document(Base):
    """
    Canonical document registry.

    Each PDF ingested gets one row here. Re-ingestion bumps version
    and creates a new DocumentVersion row — this row's version field
    always reflects the LATEST active version.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    # s3_key stores the relative path within PDF_STORAGE_PATH
    # (e.g. "2024/01/report.pdf") — abstracted for future S3 swap
    s3_key: Mapped[str] = mapped_column(String(700), nullable=False, unique=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending",
        comment="pending | processing | active | failed | archived"
    )
    
    # ─── ADDED FOR DEDUPLICATION AND COST OPTIMIZATION ───
    file_hash: Mapped[Optional[str]] = mapped_column(
        String(64), 
        nullable=True, 
        index=True,
        comment="SHA-256 hash of file content to prevent redundant processing"
    )
    
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    versions: Mapped[list["DocumentVersion"]] = relationship(
        "DocumentVersion", back_populates="document", cascade="all, delete-orphan"
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        "AuditLog", back_populates="document"
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename!r} v={self.version} hash={self.file_hash}>"
        

class DocumentVersion(Base):
    """
    Immutable version history for each document.

    Every ingestion (initial or re-ingest) appends a row here.
    Old versions are archived in MongoDB by filtering on version field.

    Interview note: This two-table design lets us answer:
    "What chunks existed at version N?" without deleting any data.
    """

    __tablename__ = "document_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    ingest_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending",
        comment="pending | processing | success | failed"
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped["Document"] = relationship("Document", back_populates="versions")

    def __repr__(self) -> str:
        return f"<DocumentVersion doc={self.document_id} v={self.version} status={self.ingest_status}>"


class AuditLog(Base):
    """
    Immutable query audit trail.

    Every /query request writes one row here regardless of outcome.
    chunk_ids and page_numbers stored as JSON arrays for traceability.

    Interview note: Append-only audit logs in MySQL give us
    compliance-grade traceability that a vector DB cannot provide.
    The JSONL backup is an additional safety net.
    """

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    document_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # answer_type: "grounded" | "fallback" | "injection_blocked" | "error"
    response_type: Mapped[str] = mapped_column(String(32), nullable=False, default="grounded")
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    retrieval_latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    generation_latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # JSON arrays — e.g. ["sha256abc", "sha256def"]
    chunk_ids: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    page_numbers: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    injection_detected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped[Optional["Document"]] = relationship("Document", back_populates="audit_logs")

    def __repr__(self) -> str:
        return f"<AuditLog request_id={self.request_id!r} type={self.response_type}>"


class ApiKey(Base):
    """
    API key registry.

    Raw keys are NEVER stored. Only SHA-256 hex digest is persisted.
    The raw key is shown exactly once at creation time (see seed_api_key.py).

    Interview note: SHA-256 is sufficient here because API keys are
    long random strings — no need for bcrypt's slow hash (which is
    designed for short, user-chosen passwords).
    """

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, comment="Human-readable label")
    key_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True,
        comment="SHA-256 hex digest of the raw API key"
    )
    rate_limit_rpm: Mapped[int] = mapped_column(
        Integer, nullable=False, default=60,
        comment="Max requests per minute for this key"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<ApiKey name={self.name!r} active={self.is_active}>"
