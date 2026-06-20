"""
Tests for GET /api/v1/health — liveness and readiness probe.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


class TestHealthEndpoint:

    @pytest.mark.asyncio
    async def test_health_returns_200_when_all_healthy(self, async_client) -> None:
        with patch("app.api.health.ping_mysql", new_callable=AsyncMock, return_value=True), \
             patch("app.api.health.ping_mongodb", new_callable=AsyncMock, return_value=True):
            response = await async_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["checks"]["mysql"] == "ok"
        assert data["checks"]["mongodb"] == "ok"
        assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_health_returns_503_when_mysql_down(self, async_client) -> None:
        with patch("app.api.health.ping_mysql", new_callable=AsyncMock, return_value=False), \
             patch("app.api.health.ping_mongodb", new_callable=AsyncMock, return_value=True):
            response = await async_client.get("/api/v1/health")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["checks"]["mysql"] == "unreachable"
        assert data["checks"]["mongodb"] == "ok"

    @pytest.mark.asyncio
    async def test_health_returns_503_when_mongodb_down(self, async_client) -> None:
        with patch("app.api.health.ping_mysql", new_callable=AsyncMock, return_value=True), \
             patch("app.api.health.ping_mongodb", new_callable=AsyncMock, return_value=False):
            response = await async_client.get("/api/v1/health")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["checks"]["mysql"] == "ok"
        assert data["checks"]["mongodb"] == "unreachable"

    @pytest.mark.asyncio
    async def test_health_returns_503_when_both_down(self, async_client) -> None:
        with patch("app.api.health.ping_mysql", new_callable=AsyncMock, return_value=False), \
             patch("app.api.health.ping_mongodb", new_callable=AsyncMock, return_value=False):
            response = await async_client.get("/api/v1/health")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["checks"]["mysql"] == "unreachable"
        assert data["checks"]["mongodb"] == "unreachable"

    @pytest.mark.asyncio
    async def test_health_response_includes_latency(self, async_client) -> None:
        with patch("app.api.health.ping_mysql", new_callable=AsyncMock, return_value=True), \
             patch("app.api.health.ping_mongodb", new_callable=AsyncMock, return_value=True):
            response = await async_client.get("/api/v1/health")

        data = response.json()
        assert isinstance(data["latency_ms"], int)
        assert data["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self) -> None:
        """Health endpoint should be accessible without API key for monitoring."""
        from unittest.mock import patch, AsyncMock
        from httpx import AsyncClient, ASGITransport

        with patch("app.db.mysql.init_mysql", new_callable=AsyncMock), \
             patch("app.db.mongodb.init_mongodb", new_callable=AsyncMock), \
             patch("app.db.mysql.close_mysql", new_callable=AsyncMock), \
             patch("app.db.mongodb.close_mongodb", new_callable=AsyncMock), \
             patch("app.api.health.ping_mysql", new_callable=AsyncMock, return_value=True), \
             patch("app.api.health.ping_mongodb", new_callable=AsyncMock, return_value=True):

            from app.main import create_app
            app = create_app()
            # Note: /health is included in the router without auth dependency
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/v1/health")
            # Health check doesn't require auth (no AuthDep on the handler)
            assert response.status_code in (200, 503)
