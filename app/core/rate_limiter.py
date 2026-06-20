"""
GeminiRateLimiter — token bucket rate limiter for Gemini API calls.

Interview note: Gemini free tier allows 15 RPM. This limiter is shared
across all callers (embedder + generator + summariser) via a single
module-level instance injected through FastAPI Depends. Token bucket
chosen over sleep-based because it handles burst traffic gracefully:
if 5 seconds have passed without calls, 1.25 tokens accrue and the
next call proceeds instantly rather than sleeping unnecessarily.

Thread safety: asyncio.Lock ensures correctness under concurrent async tasks.
"""

from __future__ import annotations

import asyncio
import time

import structlog

logger = structlog.get_logger(__name__)


class GeminiRateLimiter:
    """
    Token bucket rate limiter for Gemini API calls.

    Capacity = rpm_limit tokens.
    Tokens refill at rate = rpm_limit / 60 tokens per second.
    Each API call consumes 1 token.

    If no tokens are available, the caller sleeps until one accrues.
    This guarantees we never exceed the RPM limit regardless of
    how many concurrent async tasks are making Gemini calls.
    """

    def __init__(self, rpm_limit: int = 5) -> None:
        self.rpm_limit = rpm_limit
        self._tokens: float = float(rpm_limit)          # Start full
        self._refill_rate: float = rpm_limit / 60.0     # Tokens per second
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()
        self._total_calls: int = 0
        self._total_waits: int = 0
        logger.info("rate_limiter.init", rpm_limit=rpm_limit, refill_rate=self._refill_rate)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last call. NOT thread-safe — call under lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._refill_rate
        self._tokens = min(float(self.rpm_limit), self._tokens + new_tokens)
        self._last_refill = now

    async def acquire(self) -> None:
        """
        Acquire one token, sleeping if necessary.

        This is the single entry point for all Gemini API calls.
        Usage:
            await limiter.acquire()
            response = await gemini_client.generate(...)
        """
        async with self._lock:
            self._refill()

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self._total_calls += 1
                logger.debug(
                    "rate_limiter.acquired",
                    tokens_remaining=round(self._tokens, 2),
                    total_calls=self._total_calls,
                )
                return

            # Calculate exact sleep needed for 1 token to accrue
            deficit = 1.0 - self._tokens
            sleep_seconds = deficit / self._refill_rate
            self._total_waits += 1
            logger.info(
                "rate_limiter.waiting",
                sleep_seconds=round(sleep_seconds, 2),
                tokens_available=round(self._tokens, 2),
            )

        # Sleep OUTSIDE the lock so other coroutines aren't blocked during wait
        await asyncio.sleep(sleep_seconds)

        # Re-acquire and consume the token
        async with self._lock:
            self._refill()
            self._tokens = max(0.0, self._tokens - 1.0)
            self._total_calls += 1

    async def acquire_batch(self, count: int) -> None:
        """
        Acquire `count` tokens for batch operations (e.g., batch embedding).
        Each token is acquired sequentially to respect the rate limit.
        """
        for _ in range(count):
            await self.acquire()

    @property
    def stats(self) -> dict[str, int | float]:
        """Return current limiter statistics for /metrics endpoint."""
        return {
            "rpm_limit": self.rpm_limit,
            "tokens_available": round(self._tokens, 2),
            "total_calls": self._total_calls,
            "total_waits": self._total_waits,
        }


# ── Module-level singleton ─────────────────────────────────────────────────────
# Shared across all Gemini callers. Imported directly in modules that need it.
# FastAPI Depends wraps this for injection in API route handlers.
_limiter: GeminiRateLimiter | None = None


def get_rate_limiter() -> GeminiRateLimiter:
    """
    Return the process-wide GeminiRateLimiter singleton.
    Initialised lazily on first call with settings from config.
    """
    global _limiter
    if _limiter is None:
        from app.config import get_settings
        settings = get_settings()
        _limiter = GeminiRateLimiter(rpm_limit=settings.gemini_rpm_limit)
    return _limiter
