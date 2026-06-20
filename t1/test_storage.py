"""
Tests for app.ingestion.storage — LocalFileStorage CRUD operations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.ingestion.storage import LocalFileStorage


@pytest.fixture
def storage(tmp_path: Path) -> LocalFileStorage:
    """Return a LocalFileStorage backed by a pytest tmp_path directory."""
    return LocalFileStorage(base_path=tmp_path)


@pytest.fixture
def sample_pdf() -> bytes:
    return b"%PDF-1.4\n1 0 obj\n<</Type /Catalog>>\nendobj\n%%EOF"


class TestLocalFileStorage:

    def test_save_returns_s3_key(self, storage: LocalFileStorage, sample_pdf: bytes) -> None:
        key = storage.save(sample_pdf, "test.pdf")
        assert key.endswith("test.pdf")
        assert "/" in key  # Date prefix present

    def test_save_and_load_roundtrip(self, storage: LocalFileStorage, sample_pdf: bytes) -> None:
        key = storage.save(sample_pdf, "roundtrip.pdf")
        loaded = storage.load(key)
        assert loaded == sample_pdf

    def test_load_nonexistent_raises_file_not_found(self, storage: LocalFileStorage) -> None:
        with pytest.raises(FileNotFoundError):
            storage.load("2024/01/nonexistent.pdf")

    def test_exists_true_after_save(self, storage: LocalFileStorage, sample_pdf: bytes) -> None:
        key = storage.save(sample_pdf, "exists.pdf")
        assert storage.exists(key) is True

    def test_exists_false_before_save(self, storage: LocalFileStorage) -> None:
        assert storage.exists("2024/01/missing.pdf") is False

    def test_delete_removes_file(self, storage: LocalFileStorage, sample_pdf: bytes) -> None:
        key = storage.save(sample_pdf, "to_delete.pdf")
        assert storage.exists(key) is True
        result = storage.delete(key)
        assert result is True
        assert storage.exists(key) is False

    def test_delete_nonexistent_returns_false(self, storage: LocalFileStorage) -> None:
        result = storage.delete("2024/01/phantom.pdf")
        assert result is False

    def test_list_keys_returns_saved_files(self, storage: LocalFileStorage, sample_pdf: bytes) -> None:
        storage.save(sample_pdf, "file1.pdf")
        storage.save(sample_pdf, "file2.pdf")
        keys = storage.list_keys()
        assert len(keys) >= 2

    def test_list_keys_empty_storage(self, storage: LocalFileStorage) -> None:
        keys = storage.list_keys()
        assert keys == []

    def test_get_metadata_returns_expected_fields(self, storage: LocalFileStorage, sample_pdf: bytes) -> None:
        key = storage.save(sample_pdf, "meta.pdf")
        meta = storage.get_metadata(key)
        assert meta["s3_key"] == key
        assert meta["size_bytes"] == len(sample_pdf)
        assert len(meta["sha256"]) == 64  # SHA-256 hex = 64 chars
        assert meta["content_type"] == "application/pdf"
        assert "modified_at" in meta

    def test_path_traversal_raises_value_error(self, storage: LocalFileStorage) -> None:
        with pytest.raises((ValueError, FileNotFoundError)):
            storage.load("../../etc/passwd")

    def test_load_path_returns_absolute_path(self, storage: LocalFileStorage, sample_pdf: bytes) -> None:
        key = storage.save(sample_pdf, "path_test.pdf")
        path = storage.load_path(key)
        assert path.is_absolute()
        assert path.exists()

    def test_copy_creates_duplicate(self, storage: LocalFileStorage, sample_pdf: bytes) -> None:
        src_key = storage.save(sample_pdf, "source.pdf")
        from datetime import datetime, timezone
        now = datetime.now(tz=timezone.utc)
        dest_key = f"{now.strftime('%Y/%m')}/copy.pdf"
        storage.copy(src_key, dest_key)
        assert storage.exists(dest_key)
        assert storage.load(dest_key) == sample_pdf

    def test_overwrite_existing_file(self, storage: LocalFileStorage, sample_pdf: bytes) -> None:
        key = storage.save(sample_pdf, "overwrite.pdf")
        new_content = b"%PDF-1.4 updated content %%EOF"
        storage.save(new_content, "overwrite.pdf")
        # Load the latest — should have new content
        loaded = storage.load(key)
        assert loaded == new_content
