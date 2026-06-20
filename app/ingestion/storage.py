"""
LocalFileStorage — S3-compatible interface for local PDF storage.

Interview note: The interface is intentionally identical to what a boto3 S3
client would expose (save/load/list/delete + s3_key path abstraction).
Swapping to real S3 requires only changing this class — all callers
reference the abstract interface, not the local filesystem directly.

s3_key is a relative path within PDF_STORAGE_PATH, e.g.:
  "2024/01/report.pdf"
  "uploads/invoice_v2.pdf"
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)


class LocalFileStorage:
    """
    Local filesystem PDF storage with an S3-compatible interface.

    All paths are relative to PDF_STORAGE_PATH from settings.
    Absolute paths are never exposed outside this class.
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        settings = get_settings()
        self.base_path: Path = (base_path or settings.pdf_storage_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info("storage.init", base_path=str(self.base_path))

    def _resolve(self, s3_key: str) -> Path:
        """Resolve an s3_key to an absolute local path. Prevents path traversal."""
        # Normalise and reject any path traversal attempt
        safe_key = Path(s3_key).as_posix().lstrip("/")
        resolved = (self.base_path / safe_key).resolve()
        if not str(resolved).startswith(str(self.base_path)):
            raise ValueError(f"Path traversal detected in s3_key: {s3_key!r}")
        return resolved

    def save(self, file_bytes: bytes, filename: str) -> str:
        """
        Save file_bytes to storage. Returns the s3_key for future retrieval.

        s3_key format: "YYYY/MM/filename" — date-sharded for scale.
        If a file with the same name exists, it is overwritten.
        """
        now = datetime.now(tz=timezone.utc)
        date_prefix = now.strftime("%Y/%m")
        s3_key = f"{date_prefix}/{filename}"
        dest_path = self._resolve(s3_key)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(file_bytes)
        size_kb = len(file_bytes) / 1024
        logger.info("storage.saved", s3_key=s3_key, size_kb=round(size_kb, 1))
        return s3_key

    def load(self, s3_key: str) -> bytes:
        """
        Load and return file bytes for the given s3_key.
        Raises FileNotFoundError if not found.
        """
        path = self._resolve(s3_key)
        if not path.exists():
            raise FileNotFoundError(f"File not found in storage: {s3_key!r}")
        data = path.read_bytes()
        logger.info("storage.loaded", s3_key=s3_key, size_kb=round(len(data) / 1024, 1))
        return data

    def load_path(self, s3_key: str) -> Path:
        """
        Return the absolute local path for a stored file.
        Used when unstructured.partition.pdf needs a file path, not bytes.
        """
        path = self._resolve(s3_key)
        if not path.exists():
            raise FileNotFoundError(f"File not found in storage: {s3_key!r}")
        return path

    def delete(self, s3_key: str) -> bool:
        """
        Delete the file at s3_key. Returns True if deleted, False if not found.
        """
        path = self._resolve(s3_key)
        if path.exists():
            path.unlink()
            logger.info("storage.deleted", s3_key=s3_key)
            return True
        logger.warning("storage.delete_not_found", s3_key=s3_key)
        return False

    def list_keys(self, prefix: str = "") -> list[str]:
        """
        List all stored file s3_keys, optionally filtered by prefix.

        Returns keys sorted by modification time (newest first).
        """
        base = self.base_path
        if prefix:
            scan_root = self._resolve(prefix) if (base / prefix).exists() else base
        else:
            scan_root = base

        keys: list[tuple[float, str]] = []
        for path in scan_root.rglob("*.pdf"):
            rel = path.relative_to(base).as_posix()
            if rel.startswith(prefix):
                keys.append((path.stat().st_mtime, rel))

        keys.sort(key=lambda x: x[0], reverse=True)
        return [k for _, k in keys]

    def exists(self, s3_key: str) -> bool:
        """Check whether a file exists in storage."""
        try:
            return self._resolve(s3_key).exists()
        except ValueError:
            return False

    def get_metadata(self, s3_key: str) -> dict:
        """
        Return metadata for a stored file (size, sha256, modified_at).
        Mirrors what S3's HeadObject would return.
        """
        path = self._resolve(s3_key)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {s3_key!r}")
        data = path.read_bytes()
        stat = path.stat()
        return {
            "s3_key": s3_key,
            "size_bytes": stat.st_size,
            "sha256": hashlib.sha256(data).hexdigest(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "content_type": "application/pdf",
        }

    def copy(self, source_key: str, dest_key: str) -> str:
        """Copy a file within storage. Returns dest_key."""
        src = self._resolve(source_key)
        dst = self._resolve(dest_key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info("storage.copied", source=source_key, dest=dest_key)
        return dest_key


# ── Module-level singleton ─────────────────────────────────────────────────────
_storage: LocalFileStorage | None = None


def get_storage() -> LocalFileStorage:
    """Return the process-wide LocalFileStorage singleton."""
    global _storage
    if _storage is None:
        _storage = LocalFileStorage()
    return _storage
