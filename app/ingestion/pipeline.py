"""
Ingestion pipeline orchestrator.

Coordinates: parse → summarise → embed → store_mongo → update_mysql

Interview note: Each stage is independently testable and failure-isolated.
If summarisation fails for one chunk, we log the error and continue with
the raw text rather than failing the entire document. This is a deliberate
production trade-off — partial ingestion beats total failure.

Document versioning: each re-ingest of the same filename creates a new
version in MySQL. Old version chunks in MongoDB are marked with their
version number and excluded from search unless explicitly requested.
Content hashing prevents wasting API cost/compute on unchanged duplicate files.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import select

from app.config import get_settings
from app.db.models import Document, DocumentVersion
from app.db.mongodb import get_chunks_collection
from app.db.mysql import get_db_session
from app.ingestion.embedder import embed_chunks
from app.ingestion.parser import ParsedChunk, count_pages, parse_pdf
from app.ingestion.storage import get_storage
from app.ingestion.summariser import summarise_chunks

logger = structlog.get_logger(__name__)


def calculate_file_hash(file_bytes: bytes) -> str:
    """Calculate SHA-256 hash of file contents to detect duplicates."""
    return hashlib.sha256(file_bytes).hexdigest()


async def run_ingestion_pipeline(
    filename: str,
    file_bytes: bytes,
) -> dict:
    """
    Full ingestion pipeline for a single PDF.

    Steps:
      1. Calculate content hash to check for duplicates
      2. Fast optimization check in MySQL (Exits early if content match or backfill possible)
      3. Save PDF to local storage → get s3_key
      4. Create DocumentVersion tracking token row with status=processing
      5. Parse PDF → list[ParsedChunk]
      6. Summarise table/image chunks via Gemini Vision
      7. Embed all chunks → list[(chunk, vector)]
      8. Upsert chunks into MongoDB
      9. Update DocumentVersion + Document status → success

    Returns a summary dict consumed by the /ingest API response.
    """
    settings = get_settings()
    storage = get_storage()
    collection = get_chunks_collection()

    # ── Step 1: Content Hashing for Cost/Time Optimization ──────
    file_hash = calculate_file_hash(file_bytes)
    logger.info("pipeline.incoming_request", filename=filename, file_hash=file_hash)

    # Scoping variables for cross-session database execution integration
    document_id: int
    version: int
    version_id: int

    # ── Step 2: MySQL Smart Duplicate Check & Version Extraction ───
    async with get_db_session() as session:
        # Query for ANY matching record by filename OR content hash
        result = await session.execute(
            select(Document).where(
                (Document.file_hash == file_hash) | (Document.filename == filename)
            )
        )
        existing_doc: Optional[Document] = result.scalar_one_or_none()

        if existing_doc:
            # SCENARIO A: Exact content match already active -> Early Exit (Zero cost)
            if existing_doc.file_hash == file_hash and existing_doc.status == "active":
                logger.info(
                    "pipeline.duplicate_detected", 
                    filename=filename, 
                    document_id=existing_doc.id,
                    version=existing_doc.version
                )
                return {
                    "document_id": existing_doc.id,
                    "filename": filename,
                    "s3_key": existing_doc.s3_key,
                    "version": existing_doc.version,
                    "page_count": existing_doc.page_count,
                    "chunk_count": existing_doc.chunk_count,
                    "status": "success",
                    "note": "Skipped processing. Identical file content hash already active.",
                }
            
            # SCENARIO B: Filename matches, its content is active, but hash column was NULL.
            # Fixed variant: Backfills the empty tracking column immediately without re-processing.
            elif existing_doc.filename == filename and existing_doc.status == "active" and existing_doc.file_hash is None:
                logger.info(
                    "pipeline.backfill_hash_shortcut",
                    filename=filename,
                    document_id=existing_doc.id,
                    note="File content matches active document with missing hash. Backfilling hash and skipping processing."
                )
                existing_doc.file_hash = file_hash
                await session.flush()
                
                return {
                    "document_id": existing_doc.id,
                    "filename": filename,
                    "s3_key": existing_doc.s3_key,
                    "version": existing_doc.version,
                    "page_count": existing_doc.page_count,
                    "chunk_count": existing_doc.chunk_count,
                    "status": "success",
                    "note": "Skipped processing. Backfilled missing hash identifier for existing active document.",
                }

            # SCENARIO C: Filename matches, but content changed (New file hash) -> True Re-ingest
            else:
                existing_doc.status = "processing"
                existing_doc.file_hash = file_hash  
                version = existing_doc.version + 1
                existing_doc.version = version
                await session.flush()
                document_id = existing_doc.id
                logger.info("pipeline.re_ingest_new_content", document_id=document_id, new_version=version)
        
        else:
            # SCENARIO D: Brand new document ingestion profile
            # Temporary initialization values before disk analysis writes updates
            new_doc = Document(
                filename=filename,
                s3_key="pending_storage_allocation",
                version=1,
                status="processing",
                page_count=0,
                file_hash=file_hash,
            )
            session.add(new_doc)
            await session.flush()
            document_id = new_doc.id
            version = 1
            logger.info("pipeline.first_ingest", document_id=document_id)

    # ── Step 3: File Storage Allocation & Page Measurement ───
    s3_key = storage.save(file_bytes, filename)
    file_path = storage.load_path(s3_key)
    page_count = count_pages(file_path)

    logger.info("pipeline.storage_allocated", filename=filename, s3_key=s3_key, page_count=page_count)

    # ── Step 4: Register Job State Row Token ───
    async with get_db_session() as session:
        # Update record pointers built in step 2 with real disk configurations
        res = await session.execute(select(Document).where(Document.id == document_id))
        doc = res.scalar_one()
        doc.s3_key = s3_key
        doc.page_count = page_count
        await session.flush()

        doc_version = DocumentVersion(
            document_id=document_id,
            version=version,
            ingest_status="processing",
            page_count=page_count,
        )
        session.add(doc_version)
        await session.flush()
        version_id = doc_version.id

    error_message: Optional[str] = None

    try:
        # ── Step 5: Parse PDF ──────────────────────────────────
        chunks: list[ParsedChunk] = parse_pdf(file_path, filename)

        if not chunks:
            raise ValueError("PDF produced zero chunks — file may be empty or image-only.")

        # ── Step 6: Summarise tables + images ──────────────────
        chunks = await summarise_chunks(chunks)

        # ── Step 7: Embed all chunks ───────────────────────────
        embedded_chunks = await embed_chunks(chunks)

        # ── Step 8: Upsert into MongoDB ────────────────────────
        mongo_docs = []
        for chunk, embedding in embedded_chunks:
            mongo_doc = {
                "_id": chunk.chunk_id,
                "content": chunk.content,
                "embedding": embedding,
                "chunk_type": chunk.chunk_type,
                "document_id": document_id,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "version": version,
                "metadata": {
                    "table_html": chunk.raw_table_html,
                    "image_b64": None,         
                    "section": chunk.section,
                },
            }
            mongo_docs.append(mongo_doc)

        if mongo_docs:
            # Clean up older version tracking metrics to prevent vector retrieval pollutions
            await collection.delete_many(
                {"document_id": document_id, "version": {"$lt": version}}
            )
            
            from pymongo import ReplaceOne
            operations = [
                ReplaceOne({"_id": doc["_id"]}, doc, upsert=True)
                for doc in mongo_docs
            ]
            result = await collection.bulk_write(operations, ordered=False)
            logger.info(
                "pipeline.mongo_upsert",
                upserted=result.upserted_count,
                modified=result.modified_count,
            )

        chunk_count = len(chunks)

        # ── Step 9: Update MySQL to success ───────────────────
        async with get_db_session() as session:
            res = await session.execute(select(Document).where(Document.id == document_id))
            doc = res.scalar_one()
            doc.status = "active"
            doc.chunk_count = chunk_count
            doc.page_count = page_count

            res2 = await session.execute(
                select(DocumentVersion).where(DocumentVersion.id == version_id)
            )
            dv = res2.scalar_one()
            dv.ingest_status = "success"
            dv.chunk_count = chunk_count
            dv.page_count = page_count

        logger.info(
            "pipeline.success",
            filename=filename,
            document_id=document_id,
            version=version,
            chunk_count=chunk_count,
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "s3_key": s3_key,
            "version": version,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "status": "success",
            "chunk_type_breakdown": {
                "text": sum(1 for c in chunks if c.chunk_type == "text"),
                "table": sum(1 for c in chunks if c.chunk_type == "table"),
                "image": sum(1 for c in chunks if c.chunk_type == "image"),
            },
        }

    except Exception as exc:
        error_message = str(exc)
        logger.error("pipeline.failed", filename=filename, error=error_message, exc_info=True)

        # Update MySQL target entries to failed status
        async with get_db_session() as session:
            res = await session.execute(select(Document).where(Document.id == document_id))
            doc = res.scalar_one()
            doc.status = "failed"

            res2 = await session.execute(
                select(DocumentVersion).where(DocumentVersion.id == version_id)
            )
            dv = res2.scalar_one()
            dv.ingest_status = "failed"
            dv.error_message = error_message[:1000]

        return {
            "document_id": document_id,
            "filename": filename,
            "s3_key": s3_key,
            "version": version,
            "status": "failed",
            "error": error_message,
        }