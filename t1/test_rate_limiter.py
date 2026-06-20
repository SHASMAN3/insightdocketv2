"""
Tests for app.core.rate_limiter — token bucket behaviour.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.core.rate_limiter import GeminiRateLimiter


class TestGeminiRateLimiter:

    @pytest.mark.asyncio
    async def test_acquire_within_capacity_is_immediate(self) -> None:
        """Acquiring tokens within bucket capacity should not sleep."""
        limiter = GeminiRateLimiter(rpm_limit=15)
        start = time.monotonic()
        # Acquire 5 tokens — well within capacity of 15
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, f"Expected near-instant, took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_stats_tracks_total_calls(self) -> None:
        limiter = GeminiRateLimiter(rpm_limit=15)
        for _ in range(3):
            await limiter.acquire()
        assert limiter.stats["total_calls"] == 3

    @pytest.mark.asyncio
    async def test_tokens_decrease_after_acquire(self) -> None:
        limiter = GeminiRateLimiter(rpm_limit=10)
        initial_tokens = limiter.stats["tokens_available"]
        await limiter.acquire()
        # After acquiring, tokens should be less than initial
        # (unless refill happened, which it won't in a fast test)
        assert limiter.stats["tokens_available"] < initial_tokens

    @pytest.mark.asyncio
    async def test_rpm_limit_respected_via_delay(self) -> None:
        """
        With rpm_limit=2, acquiring 3 tokens should require waiting.
        We verify that waits are tracked (total_waits > 0).
        Use a very small rpm so the test doesn't actually sleep long.
        We mock asyncio.sleep to avoid real delays.
        """
        from unittest.mock import patch, AsyncMock

        limiter = GeminiRateLimiter(rpm_limit=2)
        sleep_calls = []

        async def mock_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            # Acquire beyond capacity — 3rd call should trigger sleep
            for _ in range(3):
                await limiter.acquire()

        assert len(sleep_calls) > 0, "Expected at least one sleep call when exceeding RPM"

    @pytest.mark.asyncio
    async def test_acquire_batch_calls_acquire_n_times(self) -> None:
        from unittest.mock import AsyncMock, patch
        limiter = GeminiRateLimiter(rpm_limit=15)
        original_acquire = limiter.acquire
        call_count = 0

        async def counting_acquire() -> None:
            nonlocal call_count
            call_count += 1
            await original_acquire()

        limiter.acquire = counting_acquire  # type: ignore[method-assign]
        await limiter.acquire_batch(4)
        assert call_count == 4

    def test_stats_dict_has_expected_keys(self) -> None:
        limiter = GeminiRateLimiter(rpm_limit=15)
        stats = limiter.stats
        assert "rpm_limit" in stats
        assert "tokens_available" in stats
        assert "total_calls" in stats
        assert "total_waits" in stats

    def test_rpm_limit_stored_correctly(self) -> None:
        limiter = GeminiRateLimiter(rpm_limit=10)
        assert limiter.rpm_limit == 10
        assert limiter.stats["rpm_limit"] == 10

    @pytest.mark.asyncio
    async def test_concurrent_acquires_serialised_by_lock(self) -> None:
        """
        Multiple concurrent coroutines should not bypass the lock.
        Verify total_calls equals number of concurrent acquires.
        """
        limiter = GeminiRateLimiter(rpm_limit=15)
        tasks = [asyncio.create_task(limiter.acquire()) for _ in range(10)]
        await asyncio.gather(*tasks)
        assert limiter.stats["total_calls"] == 10
