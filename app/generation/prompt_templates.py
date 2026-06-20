"""
Prompt templates for grounded RAG generation.

Interview note: The system prompt is the primary defence against hallucination.
Key design choices:
  1. Explicit "ONLY use the provided context" instruction — repeated twice
  2. Citation format is prescribed in the prompt — LLM must cite [Source: ...]
  3. "If the answer is not in the context" is handled explicitly — prevents
     the model from drawing on parametric memory
  4. Context is structured with chunk type labels so the LLM knows whether
     it's reading a table summary vs running prose

The prompt template uses Python f-strings (not LangChain PromptTemplate)
to keep the dependency surface minimal and the template auditable.
"""

from __future__ import annotations

from app.retrieval.vector_search import SearchResult

SYSTEM_PROMPT = """You are InsightDocket, a precise document question-answering assistant.

RULES — follow these exactly:
1. Answer ONLY using information from the CONTEXT sections below.
2. Do NOT use any external knowledge, assumptions, or information not present in the context.
3. Every factual claim in your answer MUST be followed by a citation in this format:
   [Source: <document_name>, Page <page_number>, Type: <chunk_type>]
4. If the answer cannot be found in the provided context, respond ONLY with:
   "The provided documents do not contain sufficient information to answer this question."
5. Do not speculate, extrapolate, or infer beyond what is explicitly stated.
6. Keep your answer concise and structured. Use bullet points for lists.
7. If multiple sources support the same claim, cite all of them.

CRITICAL: You have access ONLY to the context provided below. Do not use any other knowledge."""


def build_generation_prompt(
    query: str,
    chunks: list[SearchResult],
) -> str:
    """
    Build the full generation prompt from the query and retrieved chunks.

    Each chunk is formatted with its metadata so the LLM can construct
    accurate citations. Chunk content is truncated at 2000 chars to fit
    within the Gemini context window when many chunks are provided.
    """
    context_sections: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        chunk_header = (
            f"--- CONTEXT {i} ---\n"
            f"Document: {chunk.document_name}\n"
            f"Page: {chunk.page_number}\n"
            f"Type: {chunk.chunk_type}\n"
            f"Relevance Score: {chunk.score:.4f}\n"
            f"Content:\n"
        )
        # Truncate very long chunks to avoid exceeding context window
        content = chunk.content[:2000]
        if len(chunk.content) > 2000:
            content += "\n[... truncated ...]"

        context_sections.append(chunk_header + content)

    context_block = "\n\n".join(context_sections)

    prompt = f"""{SYSTEM_PROMPT}

=== CONTEXT ===
{context_block}

=== QUESTION ===
{query}

=== ANSWER ===
Answer the question using ONLY the context above. Include citations for every claim."""

    return prompt


def build_explain_summary(
    query: str,
    chunks: list[SearchResult],
    confidence_score: float,
    response_type: str,
) -> str:
    """
    Build a human-readable explanation for the /explain endpoint.

    This surfaces the retrieval internals to the caller — useful for
    debugging, auditing, and demonstrating explainability in interviews.
    """
    lines: list[str] = [
        f"Query: {query}",
        f"Response Type: {response_type}",
        f"Confidence Score: {confidence_score:.4f}",
        f"Chunks Used: {len(chunks)}",
        "",
        "=== Retrieved Chunks (ranked by reranker score) ===",
    ]

    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"\nChunk {i}:"
            f"\n  ID: {chunk.chunk_id[:16]}..."
            f"\n  Document: {chunk.document_name}"
            f"\n  Page: {chunk.page_number}"
            f"\n  Type: {chunk.chunk_type}"
            f"\n  Score: {chunk.score:.6f}"
            f"\n  Content preview: {chunk.content[:150]}..."
        )

    return "\n".join(lines)
