"""
Batch embedder using the native google-genai SDK.

Interview note: We batch embedding calls (default 10 chunks per call)
to reduce the number of API round trips while still respecting the 15 RPM
Gemini rate limit. Each batch counts as one RPM token.

Embedding model: text-embedding-004 (768-dim, free tier).
For higher quality retrieval, upgrade to embedding-001 (3072-dim) and
update MONGODB_VECTOR_DIMENSIONS in the vector index config.
"""

from __future__ import annotations

import asyncio
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.rate_limiter import get_rate_limiter
from app.ingestion.parser import ParsedChunk

# Modern google-genai SDK imports
from google import genai
from google.genai import types

logger = structlog.get_logger(__name__)


def get_genai_client() -> genai.Client:
    """
    Construct and return a genai.Client instance configured with project settings.
    """
    settings = get_settings()
    return genai.Client(api_key=settings.google_api_key)


async def embed_chunks(chunks: list[ParsedChunk]) -> list[tuple[ParsedChunk, list[float]]]:
    """
    Embed all chunks in batches. Returns list of (chunk, embedding_vector) pairs.

    Each batch acquires one rate limiter token before calling the API.
    Batches are processed sequentially (not concurrently) to avoid
    exceeding the RPM limit with parallel requests.
    """
    settings = get_settings()
    limiter = get_rate_limiter()
    client = get_genai_client()
    batch_size = settings.embedding_batch_size

    results: list[tuple[ParsedChunk, list[float]]] = []

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [chunk.content for chunk in batch]

        # Acquire rate limiter token before the API call
        await limiter.acquire()

        embeddings = await _embed_batch_with_retry(
            client=client,
            model_name=settings.embedding_model,
            texts=texts,
            dimensions=settings.embedding_dimensions
        )

        for chunk, embedding in zip(batch, embeddings):
            results.append((chunk, embedding))

        logger.info(
            "embedder.batch_complete",
            batch_start=batch_start,
            batch_size=len(batch),
            total_embedded=len(results),
            total_chunks=len(chunks),
        )

    return results


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True,
)
async def _embed_batch_with_retry(
    client: genai.Client, model_name: str, texts: list[str], dimensions: int
) -> list[list[float]]:
    """
    Call the embedding API for a batch of texts. Retries up to 3x.
    Uses asyncio.to_thread to execute the synchronous SDK call without blocking the loop.
    """
    try:
        config = types.EmbedContentConfig(
            output_dimensionality=dimensions,
            task_type="RETRIEVAL_DOCUMENT"  # Optimised for document embedding storage
        )
        
        # Offload synchronous SDK call to an event-loop background thread
        result = await asyncio.to_thread(
            client.models.embed_content,
            model=model_name,
            contents=texts,
            config=config
        )
        
        # Unpack embedding objects to list of float list values
        return [embedding.values for embedding in result.embeddings]
        
    except Exception as exc:
        logger.error("embedder.batch_failed", error=str(exc), batch_size=len(texts))
        raise


async def embed_query(query_text: str) -> list[float]:
    """
    Embed a single query string for retrieval.

    Uses RETRIEVAL_QUERY task type (distinct from RETRIEVAL_DOCUMENT)
    as recommended by Google for asymmetric search optimization.
    """
    settings = get_settings()
    limiter = get_rate_limiter()
    client = get_genai_client()

    await limiter.acquire()

    try:
        config = types.EmbedContentConfig(
            output_dimensionality=settings.embedding_dimensions,
            task_type="RETRIEVAL_QUERY"  # Query-side asymmetric search task optimization
        )

        result = await asyncio.to_thread(
            client.models.embed_content,
            model=settings.embedding_model,
            contents=query_text,
            config=config
        )
        
        # Since it's a single input string, unpack the single returned embedding element
        embedding = result.embeddings[0].values
        logger.debug("embedder.query_embedded", dim=len(embedding))
        return embedding
        
    except Exception as exc:
        logger.error("embedder.query_failed", error=str(exc))
        raise