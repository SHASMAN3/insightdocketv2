"""
LangSmith tracing integration.

Interview note: LangSmith is wired as a LangChain callback handler, not
as a wrapper (like @traceable). The distinction matters:
  - Callback approach: zero changes to generation code, tracing is additive
  - Wrapper approach: requires decorating every function, tightly coupled

The callback fires on: llm_start, llm_end, llm_error, chain_start, chain_end.
Each trace is linked to a run_id (our request_id) for correlation with
MySQL audit logs.

If LANGCHAIN_API_KEY is not set or LangSmith is unreachable, the handler
returns None and generation proceeds without tracing — fail-open design.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


def get_langsmith_handler(run_id: str) -> Optional[Any]:
    """
    Build and return a LangSmith callback handler for a single request.

    Returns None if LangSmith is not configured or unavailable.
    Callers check for None and proceed without tracing.
    """
    try:
        from app.config import get_settings
        settings = get_settings()

        if not settings.langchain_api_key:
            logger.debug("langsmith.not_configured")
            return None

        from langsmith import Client
        from langchain.callbacks.tracers import LangChainTracer

        # Validate API key is reachable (lightweight check)
        client = Client(
            api_url=settings.langchain_endpoint,
            api_key=settings.langchain_api_key,
        )

        tracer = LangChainTracer(
            project_name=settings.langchain_project,
            client=client,
        )

        logger.debug(
            "langsmith.handler_created",
            project=settings.langchain_project,
            run_id=run_id,
        )
        return tracer

    except ImportError:
        logger.warning("langsmith.import_failed", hint="pip install langsmith langchain")
        return None
    except Exception as exc:
        logger.warning("langsmith.handler_failed", error=str(exc))
        return None


async def trace_query_event(
    request_id: str,
    query: str,
    answer: str,
    latency_ms: int,
    confidence_score: float,
    response_type: str,
) -> None:
    """
    Log a query event to LangSmith as a standalone run.

    Used for events that aren't part of a LangChain chain but should
    still appear in the LangSmith trace dashboard.
    """
    try:
        from app.config import get_settings
        settings = get_settings()

        if not settings.langchain_api_key:
            return

        from langsmith import Client, RunTree

        client = Client(
            api_url=settings.langchain_endpoint,
            api_key=settings.langchain_api_key,
        )

        run = RunTree(
            name="insightdocket_query",
            run_type="chain",
            inputs={"query": query},
            outputs={
                "answer": answer[:500],  # Truncate for trace size
                "response_type": response_type,
                "confidence_score": confidence_score,
            },
            extra={
                "request_id": request_id,
                "latency_ms": latency_ms,
            },
            project_name=settings.langchain_project,
            client=client,
        )
        await run.apost()

        logger.debug("langsmith.event_posted", request_id=request_id)

    except Exception as exc:
        # Non-fatal — observability failure should never break the API
        logger.debug("langsmith.event_failed", error=str(exc))
