"""
Test fixtures shared across all test modules.

Provides:
  - test_client: FastAPI TestClient with mocked DB dependencies
  - mock_settings: Settings instance with test values
  - sample_pdf_bytes: minimal valid PDF for upload tests
  - mock_chunks_collection: in-memory MongoDB collection mock
"""

from __future__ import annotations

import io
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from app.config import Settings, get_settings
from app.main import create_app


# ── Settings override ──────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Return a Settings instance with safe test values."""
    return Settings(
        google_api_key="test-google-key-fake",
        langchain_api_key="",
        langchain_tracing_v2=False,
        mongodb_uri="mongodb://localhost:27017/?directConnection=true",
        mongodb_database="test_insightdocket",
        mysql_host="localhost",
        mysql_user="test",
        mysql_password="test",
        mysql_database="test_insightdocket",
        pdf_storage_path="/tmp/test_insightdocket_storage",
        audit_log_dir="/tmp/test_insightdocket_logs",
        app_env="development",
        gemini_rpm_limit=15,
        confidence_threshold=0.35,
        secret_key="test-secret-key-not-for-production",
    )


# ── Minimal valid PDF bytes ────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def sample_pdf_bytes() -> bytes:
    """
    Return the bytes of a minimal valid 1-page PDF.
    Generated inline — no test file dependency.
    """
    # Minimal PDF structure — passes magic byte check (%PDF) and is parseable
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj

2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj

3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj

4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello InsightDocket) Tj ET
endstream
endobj

5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj

xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000360 00000 n 

trailer
<< /Size 6 /Root 1 0 R >>
startxref
441
%%EOF"""
    return pdf_content


# ── Mock API key ───────────────────────────────────────────────────────────────
TEST_API_KEY_RAW = "test-api-key-abcdef1234567890abcdef1234567890abcdef1234567890abcdef12"
TEST_API_KEY_NAME = "test-key"


@pytest.fixture
def mock_api_key_auth():
    """
    Patch authenticate_api_key to return a mock ApiKey without hitting MySQL.
    Use as a fixture in tests that call authenticated endpoints.
    """
    from app.db.models import ApiKey
    mock_key = MagicMock(spec=ApiKey)
    mock_key.name = TEST_API_KEY_NAME
    mock_key.rate_limit_rpm = 60
    mock_key.is_active = True

    with patch("app.core.security.authenticate_api_key", return_value=mock_key):
        with patch("app.api.ingest.authenticate_api_key", return_value=mock_key):
            with patch("app.api.query.authenticate_api_key", return_value=mock_key):
                with patch("app.api.explain.authenticate_api_key", return_value=mock_key):
                    with patch("app.api.documents.authenticate_api_key", return_value=mock_key):
                        with patch("app.api.metrics.authenticate_api_key", return_value=mock_key):
                            yield mock_key


# ── Async test client ──────────────────────────────────────────────────────────
@pytest.fixture
async def async_client(mock_api_key_auth) -> AsyncGenerator[AsyncClient, None]:
    """
    Return an httpx AsyncClient connected to the FastAPI app.
    Databases are mocked — no real MySQL/MongoDB connections required.
    """
    with patch("app.db.mysql.init_mysql", new_callable=AsyncMock):
        with patch("app.db.mongodb.init_mongodb", new_callable=AsyncMock):
            with patch("app.db.mysql.close_mysql", new_callable=AsyncMock):
                with patch("app.db.mongodb.close_mongodb", new_callable=AsyncMock):
                    app = create_app()
                    async with AsyncClient(
                        transport=ASGITransport(app=app),
                        base_url="http://test",
                        headers={"X-API-Key": TEST_API_KEY_RAW},
                    ) as client:
                        yield client
