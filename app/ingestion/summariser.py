"""
Gemini Vision summariser for table and image chunks.

Interview note: We summarise tables and images rather than storing raw
OCR text because:
  - Raw table HTML is not embeddable — cosine similarity on "<td>42</td>"
    is meaningless
  - Image text (OCR) loses visual context (charts, diagrams)
  - Gemini Vision produces semantically rich summaries that embed well
  - Summaries are retrievable via natural language questions like
    "What does the revenue chart show?"

Rate limiter is acquired before every Gemini call to respect 15 RPM.
"""

from __future__ import annotations

import base64

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.rate_limiter import get_rate_limiter
from app.ingestion.parser import ParsedChunk

# Modern google-genai SDK imports
from google import genai
from google.genai import types

logger = structlog.get_logger(__name__)

# Prompts are module-level constants so they're easy to audit and tune
_TABLE_SUMMARY_PROMPT = """You are a precise document analyst. Summarise the following HTML table in 2-4 sentences.
Include: what the table shows, key values or trends, and column/row structure.
Do NOT invent data not present in the table.
Output only the summary, no preamble.

Table HTML:
{table_html}"""

_IMAGE_SUMMARY_PROMPT = """You are a precise document analyst. Describe the image in 2-4 sentences.
Include: what type of visual it is (chart/graph/diagram/photo), what it shows,
and any key values, labels, or trends visible.
Do NOT invent details not visible in the image.
Output only the description, no preamble."""


async def summarise_chunks(chunks: list[ParsedChunk]) -> list[ParsedChunk]:
    """
    Summarise all table and image chunks using Gemini Vision.

    Text chunks are returned unchanged.
    Table/image chunks have their .content field replaced with the summary.
    Rate limiter is acquired before each Gemini call.

    Returns the same list with summaries filled in.
    """
    settings = get_settings()
    limiter = get_rate_limiter()

    # Initialize the modern SDK client
    client = genai.Client(api_key=settings.google_api_key)

    summarised: list[ParsedChunk] = []
    for chunk in chunks:
        if chunk.chunk_type == "text":
            summarised.append(chunk)
            continue

        if chunk.chunk_type == "table":
            summary = await _summarise_table(chunk, client, settings.gemini_vision_model, limiter)
        elif chunk.chunk_type == "image":
            summary = await _summarise_image(chunk, client, settings.gemini_vision_model, limiter)
        else:
            summary = chunk.content

        chunk.content = summary
        summarised.append(chunk)

    table_count = sum(1 for c in chunks if c.chunk_type == "table")
    image_count = sum(1 for c in chunks if c.chunk_type == "image")
    logger.info(
        "summariser.complete",
        table_chunks_summarised=table_count,
        image_chunks_summarised=image_count,
    )
    return summarised


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True,
)
async def _summarise_table(
    chunk: ParsedChunk, 
    client: genai.Client, 
    model_name: str, 
    limiter: object
) -> str:
    """Summarise a table chunk using Gemini. Retries up to 3x on failure."""
    from app.core.rate_limiter import GeminiRateLimiter
    assert isinstance(limiter, GeminiRateLimiter)

    table_html = chunk.raw_table_html or chunk.content
    prompt = _TABLE_SUMMARY_PROMPT.format(table_html=table_html[:4000])  # Truncate for token limit

    await limiter.acquire()

    try:
        # Use client.models.generate_content for the modern API
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        summary = response.text.strip()
        logger.debug("summariser.table_done", chunk_id=chunk.chunk_id, summary_len=len(summary))
        return summary
    except Exception as exc:
        logger.error("summariser.table_failed", chunk_id=chunk.chunk_id, error=str(exc))
        # Return truncated HTML as fallback — still embeddable, just lower quality
        return f"Table on page {chunk.page_number}: {(chunk.raw_table_html or chunk.content)[:500]}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True,
)
async def _summarise_image(
    chunk: ParsedChunk, 
    client: genai.Client, 
    model_name: str, 
    limiter: object
) -> str:
    """Summarise an image chunk using Gemini Vision. Retries up to 3x on failure."""
    from app.core.rate_limiter import GeminiRateLimiter
    assert isinstance(limiter, GeminiRateLimiter)

    if not chunk.raw_image_b64:
        logger.warning("summariser.image_no_data", chunk_id=chunk.chunk_id)
        return f"[Image on page {chunk.page_number} — no data available]"

    await limiter.acquire()

    try:
        # Decode base64 to bytes 
        image_bytes = base64.b64decode(chunk.raw_image_b64)

        # Build multimodal contents payload using types.Part.from_bytes
        response = client.models.generate_content(
            model=model_name,
            contents=[
                _IMAGE_SUMMARY_PROMPT,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                ),
            ],
        )
        summary = response.text.strip()
        logger.debug("summariser.image_done", chunk_id=chunk.chunk_id, summary_len=len(summary))
        return summary
    except Exception as exc:
        logger.error("summariser.image_failed", chunk_id=chunk.chunk_id, error=str(exc))
        return f"[Image on page {chunk.page_number} — summarisation failed: {str(exc)[:100]}]"