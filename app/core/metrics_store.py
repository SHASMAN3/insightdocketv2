"""
In-process metrics store.

Provides lightweight Prometheus-style counters and gauges without
requiring a Prometheus server. Metrics are exposed via /api/v1/metrics.

Interview note: For production scale, replace this with the official
prometheus_client library and a /metrics scrape endpoint. This in-process
implementation is sufficient for demonstrating observability patterns
in interviews without adding infrastructure dependencies.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricsStore:
    """
    Thread-safe in-process metrics store.

    Uses threading.Lock (not asyncio.Lock) because metrics may be
    updated from sync code paths (e.g., background tasks).
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── Request counters ───────────────────────────────────────
    requests_total: int = 0
    requests_ingest: int = 0
    requests_query: int = 0
    requests_explain: int = 0

    # ── Response type counters ─────────────────────────────────
    responses_grounded: int = 0
    responses_fallback: int = 0
    responses_injection_blocked: int = 0
    responses_error: int = 0

    # ── Latency tracking (milliseconds) ───────────────────────
    _latency_samples: list[float] = field(default_factory=list, repr=False)
    _retrieval_latency_samples: list[float] = field(default_factory=list, repr=False)
    _generation_latency_samples: list[float] = field(default_factory=list, repr=False)

    # ── Confidence tracking ────────────────────────────────────
    _confidence_samples: list[float] = field(default_factory=list, repr=False)

    # ── Process start time ─────────────────────────────────────
    start_time: float = field(default_factory=time.time)

    def record_request(self, endpoint: str) -> None:
        """Increment request counters for the given endpoint."""
        with self._lock:
            self.requests_total += 1
            if endpoint == "ingest":
                self.requests_ingest += 1
            elif endpoint == "query":
                self.requests_query += 1
            elif endpoint == "explain":
                self.requests_explain += 1

    def record_response(
        self,
        response_type: str,
        total_latency_ms: float,
        retrieval_latency_ms: Optional[float] = None,
        generation_latency_ms: Optional[float] = None,
        confidence_score: Optional[float] = None,
    ) -> None:
        """Record a completed response with its latency and type."""
        with self._lock:
            if response_type == "grounded":
                self.responses_grounded += 1
            elif response_type == "fallback":
                self.responses_fallback += 1
            elif response_type == "injection_blocked":
                self.responses_injection_blocked += 1
            else:
                self.responses_error += 1

            # Keep last 1000 samples to bound memory usage
            self._latency_samples.append(total_latency_ms)
            if len(self._latency_samples) > 1000:
                self._latency_samples = self._latency_samples[-1000:]

            if retrieval_latency_ms is not None:
                self._retrieval_latency_samples.append(retrieval_latency_ms)
                if len(self._retrieval_latency_samples) > 1000:
                    self._retrieval_latency_samples = self._retrieval_latency_samples[-1000:]

            if generation_latency_ms is not None:
                self._generation_latency_samples.append(generation_latency_ms)
                if len(self._generation_latency_samples) > 1000:
                    self._generation_latency_samples = self._generation_latency_samples[-1000:]

            if confidence_score is not None:
                self._confidence_samples.append(confidence_score)
                if len(self._confidence_samples) > 1000:
                    self._confidence_samples = self._confidence_samples[-1000:]

    def snapshot(self) -> dict:
        """Return a metrics snapshot dict for the /metrics endpoint."""
        with self._lock:
            uptime_seconds = int(time.time() - self.start_time)

            def avg(samples: list[float]) -> Optional[float]:
                return round(sum(samples) / len(samples), 2) if samples else None

            def p95(samples: list[float]) -> Optional[float]:
                if not samples:
                    return None
                sorted_samples = sorted(samples)
                idx = int(len(sorted_samples) * 0.95)
                return round(sorted_samples[min(idx, len(sorted_samples) - 1)], 2)

            fallback_rate = (
                round(self.responses_fallback / max(self.responses_grounded + self.responses_fallback, 1), 4)
                if (self.responses_grounded + self.responses_fallback) > 0
                else 0.0
            )

            return {
                "uptime_seconds": uptime_seconds,
                "requests": {
                    "total": self.requests_total,
                    "ingest": self.requests_ingest,
                    "query": self.requests_query,
                    "explain": self.requests_explain,
                },
                "responses": {
                    "grounded": self.responses_grounded,
                    "fallback": self.responses_fallback,
                    "injection_blocked": self.responses_injection_blocked,
                    "error": self.responses_error,
                    "fallback_rate": fallback_rate,
                },
                "latency_ms": {
                    "avg_total": avg(self._latency_samples),
                    "p95_total": p95(self._latency_samples),
                    "avg_retrieval": avg(self._retrieval_latency_samples),
                    "p95_retrieval": p95(self._retrieval_latency_samples),
                    "avg_generation": avg(self._generation_latency_samples),
                    "p95_generation": p95(self._generation_latency_samples),
                },
                "confidence": {
                    "avg": avg(self._confidence_samples),
                    "samples": len(self._confidence_samples),
                },
                "injection_patterns_loaded": None,  # Set by sanitiser at startup
            }


# ── Module-level singleton ─────────────────────────────────────────────────────
_metrics = MetricsStore()


def get_metrics() -> MetricsStore:
    """Return the process-wide MetricsStore singleton."""
    return _metrics
