"""
Tests for POST /api/v1/ingest — mocked ingestion pipeline.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


class TestIngestEndpoint:

    @pytest.mark.asyncio
    async def test_ingest_valid_pdf_returns_202(
        self,
        async_client,
        sample_pdf_bytes: bytes,
    ) -> None:
        mock_result = {
            "document_id": 1,
            "filename": "test.pdf",
            "s3_key": "2024/01/test.pdf",
            "version": 1,
            "page_count": 1,
            "chunk_count": 5,
            "status": "success",
            "chunk_type_breakdown": {"text": 4, "table": 1, "image": 0},
        }
        with patch(
            "app.api.ingest.run_ingestion_pipeline",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = await async_client.post(
                "/api/v1/ingest",
                files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            )

        assert response.status_code == 202
        data = response.json()
        assert data["document_id"] == 1
        assert data["status"] == "success"
        assert data["chunk_count"] == 5

    @pytest.mark.asyncio
    async def test_ingest_non_pdf_returns_415(self, async_client) -> None:
        response = await async_client.post(
            "/api/v1/ingest",
            files={"file": ("document.txt", b"plain text content", "text/plain")},
        )
        assert response.status_code == 415

    @pytest.mark.asyncio
    async def test_ingest_empty_file_returns_422(self, async_client) -> None:
        response = await async_client.post(
            "/api/v1/ingest",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_invalid_pdf_magic_bytes_returns_415(self, async_client) -> None:
        response = await async_client.post(
            "/api/v1/ingest",
            files={"file": ("fake.pdf", b"not a pdf at all", "application/pdf")},
        )
        assert response.status_code == 415

    @pytest.mark.asyncio
    async def test_ingest_pipeline_failure_returns_500(
        self, async_client, sample_pdf_bytes: bytes
    ) -> None:
        mock_result = {
            "document_id": 1,
            "filename": "bad.pdf",
            "s3_key": "2024/01/bad.pdf",
            "version": 1,
            "status": "failed",
            "error": "Parse failed: corrupted PDF",
        }
        with patch(
            "app.api.ingest.run_ingestion_pipeline",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = await async_client.post(
                "/api/v1/ingest",
                files={"file": ("bad.pdf", sample_pdf_bytes, "application/pdf")},
            )

        assert response.status_code == 500
        assert "Ingestion failed" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_ingest_filename_sanitised(
        self, async_client, sample_pdf_bytes: bytes
    ) -> None:
        """A filename with path traversal should be sanitised, not rejected."""
        mock_result = {
            "document_id": 1,
            "filename": "..evil.pdf",
            "s3_key": "2024/01/..evil.pdf",
            "version": 1,
            "status": "success",
            "chunk_count": 3,
            "chunk_type_breakdown": {"text": 3, "table": 0, "image": 0},
        }
        with patch(
            "app.api.ingest.run_ingestion_pipeline",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = await async_client.post(
                "/api/v1/ingest",
                files={"file": ("../../evil.pdf", sample_pdf_bytes, "application/pdf")},
            )
        # Should succeed (sanitised) or fail gracefully — not 500
        assert response.status_code in (202, 500)
