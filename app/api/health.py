"""
GET /api/v1/health — Liveness + readiness probe.

Returns 200 if all dependencies are reachable, 503 otherwise.
Used by Docker HEALTHCHECK and Kubernetes readiness probes.
"""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Response, status

from app.db.mongodb import ping_mongodb
from app.db.mysql import ping_mysql

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/health",
    summary="Liveness and readiness probe",
    include_in_schema=True,
)
async def health_check(response: Response) -> dict:
    """
    Check connectivity to MySQL and MongoDB.

    Returns 200 OK if both databases are reachable.
    Returns 503 Service Unavailable if any dependency is down.

    Interview note: Separating liveness (is the process alive?) from
    readiness (can it serve traffic?) is a Kubernetes best practice.
    This endpoint serves as the readiness probe — if MySQL or MongoDB
    is unreachable, the pod is removed from the load balancer until
    connectivity is restored.
    """
    start = time.monotonic()

    mysql_ok = await ping_mysql()
    mongodb_ok = await ping_mongodb()

    latency_ms = int((time.monotonic() - start) * 1000)
    all_healthy = mysql_ok and mongodb_ok

    if not all_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    result = {
        "status": "healthy" if all_healthy else "degraded",
        "checks": {
            "mysql": "ok" if mysql_ok else "unreachable",
            "mongodb": "ok" if mongodb_ok else "unreachable",
        },
        "latency_ms": latency_ms,
    }

    if not all_healthy:
        logger.warning("health.degraded", checks=result["checks"])

    return result
