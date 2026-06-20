"""
MySQL async engine and session factory.

Uses SQLAlchemy 2.0 async API with aiomysql driver.
Connection pool is configured for FastAPI's async request lifecycle.
"""


from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.db.models import Base

logger = structlog.get_logger(__name__)

# Module-level engine and session factory — created once at lifespan startup.
_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_mysql() -> None:
    """
    Initialise the async engine and create all tables.

    Called once from FastAPI lifespan. Idempotent — CREATE TABLE IF NOT EXISTS.
    """
    global _engine, _async_session_factory

    settings = get_settings()

    _engine = create_async_engine(
        settings.mysql_dsn,
        echo=False,                   # Set to True for SQL debug logging
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=1800,            # Recycle connections every 30 min
        pool_pre_ping=True,           # Detect stale connections before use
    )

    _async_session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,       # Keep ORM objects usable after commit
        autocommit=False,
        autoflush=False,
    )

    # Create all tables defined in models.py
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("mysql.initialised", dsn=settings.mysql_dsn.split("@")[-1])


async def close_mysql() -> None:
    """Dispose the engine connection pool. Called from FastAPI lifespan shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("mysql.closed")


def get_engine() -> AsyncEngine:
    """Return the module-level async engine. Raises if not initialised."""
    if _engine is None:
        raise RuntimeError("MySQL engine not initialised. Call init_mysql() first.")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the session factory. Raises if not initialised."""
    if _async_session_factory is None:
        raise RuntimeError("MySQL session factory not initialised.")
    return _async_session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager yielding a scoped session.

    Commits on success, rolls back on any exception.
    Usage:
        async with get_db_session() as session:
            session.add(obj)
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def ping_mysql() -> bool:
    """
    Liveness probe for /health endpoint.
    Returns True if MySQL is reachable, False otherwise.
    """
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            from sqlalchemy import text
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("mysql.ping_failed", error=str(exc))
        return False
