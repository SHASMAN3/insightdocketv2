"""
FastAPI dependency injection.

All reusable Depends() callables live here so route handlers
stay thin and testable. Each dependency is independently mockable in tests.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.rate_limiter import GeminiRateLimiter, get_rate_limiter
from app.core.security import ApiKey, authenticate_api_key
from app.db.mongodb import get_chunks_collection
from app.ingestion.storage import LocalFileStorage, get_storage
from motor.motor_asyncio import AsyncIOMotorCollection


# ── Auth dependency ────────────────────────────────────────────────────────────
AuthDep = Annotated[ApiKey, Depends(authenticate_api_key)]


# ── Rate limiter dependency ────────────────────────────────────────────────────
def get_limiter() -> GeminiRateLimiter:
    """Inject the process-wide GeminiRateLimiter singleton."""
    return get_rate_limiter()


RateLimiterDep = Annotated[GeminiRateLimiter, Depends(get_limiter)]


# ── Storage dependency ─────────────────────────────────────────────────────────
def get_pdf_storage() -> LocalFileStorage:
    """Inject the process-wide LocalFileStorage singleton."""
    return get_storage()


StorageDep = Annotated[LocalFileStorage, Depends(get_pdf_storage)]


# ── MongoDB collection dependency ──────────────────────────────────────────────
def get_collection() -> AsyncIOMotorCollection:  # type: ignore[type-arg]
    """Inject the chunks MongoDB collection handle."""
    return get_chunks_collection()


CollectionDep = Annotated[AsyncIOMotorCollection, Depends(get_collection)]  # type: ignore[type-arg]
