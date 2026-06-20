"""
Motor async MongoDB client for InsightDocket.

Interview note: We use MongoDB for vectors because $vectorSearch and $text
can be combined in a single aggregation pipeline — no second service needed,
no cross-service latency, no synchronisation bugs between vector DB and BM25 index.

directConnection=true is required for single-container setups (no replica set).
retryWrites=false avoids Motor attempting retryable writes without an oplog.
"""

from __future__ import annotations

import structlog
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import ASCENDING, TEXT

from app.config import get_settings

logger = structlog.get_logger(__name__)

# Module-level client — created once at lifespan startup.
_client: AsyncIOMotorClient | None = None  # type: ignore[type-arg]


async def init_mongodb() -> None:
    """
    Initialise the Motor client and ensure all required indexes exist.

    Called once from FastAPI lifespan. Index creation is idempotent.

    Vector index ($vectorSearch) must be created via mongosh or Atlas UI
    because the index definition uses a special 'vectorSearch' index type
    not expressible through Motor's create_index(). The README documents
    the exact mongosh command. We create the $text index here programmatically.
    """
    global _client

    settings = get_settings()

    _client = AsyncIOMotorClient(
        settings.mongodb_uri,
        retryWrites=False,   # Single container — no oplog for retryable writes
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=30000,
    )

    # Verify connectivity
    await _client.admin.command("ping")

    db = _client[settings.mongodb_database]
    collection = db[settings.mongodb_collection]

    # ── BM25 full-text index ────────────────────────────────────
    # $text search uses this index. Compound with document_name for
    # filtering by document. Weights boost content over metadata.
    await collection.create_index(
        [("content", TEXT), ("document_name", TEXT)],
        name="text_search_index",
        weights={"content": 10, "document_name": 1},
        default_language="english",
        background=True,
    )

    # ── Scalar indexes for filtered queries ────────────────────
    await collection.create_index(
        [("document_id", ASCENDING), ("version", ASCENDING)],
        name="document_version_index",
        background=True,
    )
    await collection.create_index(
        [("chunk_type", ASCENDING)],
        name="chunk_type_index",
        background=True,
    )
    await collection.create_index(
        [("page_number", ASCENDING)],
        name="page_number_index",
        background=True,
    )

    logger.info(
        "mongodb.initialised",
        database=settings.mongodb_database,
        collection=settings.mongodb_collection,
    )
    logger.warning(
        "mongodb.vector_index_reminder",
        message=(
            "Ensure the '$vectorSearch' index named 'vector_index' exists. "
            "Run: db.chunks.createIndex({embedding:'cosmosSearch'}, ...) "
            "See README for exact mongosh command."
        ),
    )


async def close_mongodb() -> None:
    """Close the Motor client. Called from FastAPI lifespan shutdown."""
    global _client
    if _client:
        _client.close()
        logger.info("mongodb.closed")


def get_client() -> AsyncIOMotorClient:  # type: ignore[type-arg]
    """Return the module-level Motor client. Raises if not initialised."""
    if _client is None:
        raise RuntimeError("MongoDB client not initialised. Call init_mongodb() first.")
    return _client


def get_database() -> AsyncIOMotorDatabase:  # type: ignore[type-arg]
    """Return the configured database handle."""
    settings = get_settings()
    return get_client()[settings.mongodb_database]


def get_chunks_collection() -> AsyncIOMotorCollection:  # type: ignore[type-arg]
    """Return the chunks collection handle used by all retrieval + ingestion code."""
    settings = get_settings()
    return get_database()[settings.mongodb_collection]


async def ping_mongodb() -> bool:
    """
    Liveness probe for /health endpoint.
    Returns True if MongoDB is reachable, False otherwise.
    """
    try:
        client = get_client()
        await client.admin.command("ping")
        return True
    except Exception as exc:
        logger.warning("mongodb.ping_failed", error=str(exc))
        return False
