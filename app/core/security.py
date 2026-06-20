"""
API key authentication and per-key sliding-window rate limiting.

Interview note: SHA-256 is used instead of bcrypt for API keys because:
  - API keys are long random strings (256-bit entropy) — no need for slow hash
  - bcrypt's slow KDF is designed for short user-chosen passwords
  - SHA-256 lookup is O(1) and doesn't add latency to every request

Sliding-window rate limiter uses an in-process deque per API key.
For multi-process deployments, swap the deque for a Redis sorted set.
"""

from __future__ import annotations

import collections
import hashlib
import time
from typing import Optional

import structlog
from fastapi import HTTPException, Request, status
from sqlalchemy import select

from app.config import get_settings
from app.db.models import ApiKey
from app.db.mysql import get_db_session

logger = structlog.get_logger(__name__)

# ── In-process sliding window store ───────────────────────────────────────────
# Maps key_hash → deque of request timestamps (Unix float seconds)
# deque maxlen = rate_limit_rpm ensures automatic eviction of old entries
_rate_windows: dict[str, collections.deque[float]] = {}


def hash_api_key(raw_key: str) -> str:
    """
    Return the SHA-256 hex digest of a raw API key.

    This is the ONLY function that should touch raw key material.
    The hash is what gets stored in MySQL and compared at auth time.
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


async def authenticate_api_key(request: Request) -> ApiKey:
    """
    FastAPI dependency: authenticate the incoming request's API key.

    1. Extract raw key from X-API-Key header
    2. Hash it with SHA-256
    3. Look up the hash in MySQL api_keys table
    4. Verify the key is active
    5. Apply sliding-window rate limit

    Raises HTTP 401 if missing/invalid, HTTP 429 if rate limited.
    """
    settings = get_settings()
    raw_key = request.headers.get(settings.api_key_header)

    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_hash = hash_api_key(raw_key)

    async with get_db_session() as session:
        result = await session.execute(
            select(ApiKey).where(ApiKey.key_hash == key_hash)
        )
        api_key_obj: Optional[ApiKey] = result.scalar_one_or_none()

    if api_key_obj is None or not api_key_obj.is_active:
        logger.warning("auth.invalid_key", key_hash_prefix=key_hash[:8])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # ── Sliding-window rate limit ──────────────────────────────
    _check_rate_limit(key_hash, api_key_obj.rate_limit_rpm)

    logger.debug("auth.success", key_name=api_key_obj.name, key_hash_prefix=key_hash[:8])
    return api_key_obj


def _check_rate_limit(key_hash: str, rpm_limit: int) -> None:
    """
    Sliding-window rate limit check for a given API key.

    Maintains a deque of request timestamps. Timestamps older than
    60 seconds are popped. If the deque length equals rpm_limit,
    the request is rejected.

    Interview note: Sliding window is more accurate than fixed window
    because it prevents burst attacks at window boundaries.
    """
    now = time.monotonic()
    window_seconds = 60.0

    if key_hash not in _rate_windows:
        _rate_windows[key_hash] = collections.deque()

    window = _rate_windows[key_hash]

    # Evict timestamps outside the 60-second window
    while window and (now - window[0]) > window_seconds:
        window.popleft()

    if len(window) >= rpm_limit:
        oldest = window[0]
        retry_after = int(window_seconds - (now - oldest)) + 1
        logger.warning(
            "auth.rate_limited",
            key_hash_prefix=key_hash[:8],
            rpm_limit=rpm_limit,
            current_count=len(window),
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {rpm_limit} requests per minute.",
            headers={"Retry-After": str(retry_after)},
        )

    window.append(now)
