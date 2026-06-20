"""
InsightDocket — FastAPI application factory.

Lifespan manages startup/shutdown of database connections.
All routers are registered under /api/v1/.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import documents, explain, health, ingest, metrics, query
from app.config import get_settings
from app.db.mongodb import close_mongodb, init_mongodb
from app.db.mysql import close_mysql, init_mysql

# ── Structured logging setup ───────────────────────────────────────────────────
def _configure_logging(log_level: str) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if os.getenv("APP_ENV") != "production"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan: initialise databases on startup, close on shutdown.

    Using lifespan (vs @app.on_event) is the modern FastAPI pattern —
    startup and shutdown are co-located, making the lifecycle explicit.
    """
    settings = get_settings()
    _configure_logging(settings.app_log_level)

    logger.info(
        "app.starting",
        env=settings.app_env,
        version="1.0.0",
    )

    # Startup — order matters: MySQL first (schema creation), then MongoDB
    await init_mysql()
    await init_mongodb()

    logger.info("app.ready")
    yield  # ← Application serves requests here

    # Shutdown
    logger.info("app.shutting_down")
    await close_mongodb()
    await close_mysql()
    logger.info("app.stopped")


def create_app() -> FastAPI:
    """
    Application factory — returns a configured FastAPI instance.

    Factory pattern allows creating test instances with different settings
    without importing app-level state at module load time.
    """
    settings = get_settings()

    app = FastAPI(
        title="InsightDocket",
        description=(
            "Multimodal RAG PDF QA System. "
            "Hybrid MongoDB vector + BM25 search, Gemini Vision, "
            "cross-encoder reranking, and full audit trails."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler ───────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "app.unhandled_exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True,
        )
        from app.core.metrics_store import get_metrics
        get_metrics().record_response("error", total_latency_ms=0)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal error occurred. Check server logs."},
        )

    # ── Routers ────────────────────────────────────────────────
    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix, tags=["Health"])
    app.include_router(metrics.router, prefix=prefix, tags=["Metrics"])
    app.include_router(ingest.router, prefix=prefix, tags=["Ingestion"])
    app.include_router(query.router, prefix=prefix, tags=["Query"])
    app.include_router(explain.router, prefix=prefix, tags=["Explain"])
    app.include_router(documents.router, prefix=prefix, tags=["Documents"])

    return app


# Module-level app instance used by uvicorn
app = create_app()
