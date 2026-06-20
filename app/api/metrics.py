"""
GET /api/v1/metrics — In-process metrics snapshot.

Returns Prometheus-style counters and latency percentiles collected
since process start. No external metrics server required.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.metrics_store import get_metrics
from app.core.rate_limiter import get_rate_limiter
from app.core.sanitiser import get_injection_pattern_count
from app.dependencies import AuthDep

router = APIRouter()


@router.get(
    "/metrics",
    summary="In-process metrics snapshot",
)
async def get_metrics_snapshot(api_key: AuthDep) -> dict:
    """
    Return a snapshot of in-process metrics since last process start.

    Includes: request counts, response type breakdown, fallback rate,
    latency percentiles (avg + P95), confidence distribution,
    Gemini rate limiter stats, and injection pattern count.
    """
    snapshot = get_metrics().snapshot()

    # Enrich with rate limiter stats
    limiter = get_rate_limiter()
    snapshot["gemini_rate_limiter"] = limiter.stats

    # Enrich with injection pattern count
    snapshot["injection_patterns_loaded"] = get_injection_pattern_count()

    return snapshot
