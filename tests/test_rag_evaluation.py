"""
End-to-end RAG quality evaluation using DeepEval metrics.

Interview note: This suite evaluates the *quality* of retrieval and
generation (faithfulness, relevancy, contextual precision) rather than
just unit-level correctness. It exercises the real pipeline functions
directly — vector_search, text_search, reciprocal_rank_fusion, rerank,
compute_confidence, generate_answer — matching their actual signatures
in app/retrieval/ and app/generation/.

Requires a running MongoDB instance with ingested chunks and a valid
GOOGLE_API_KEY for embedding/generation calls. Marked separately from
the unit test suite since these are slower, network-dependent,
LLM-as-judge evaluations rather than deterministic assertions.

Run with: uv run pytest tests/test_rag_evaluation.py -v -m evaluation
"""

from __future__ import annotations

import os
import time
import asyncio
import uuid

import pytest
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from google import genai
from google.genai.errors import ClientError

from app.db.mongodb import close_mongodb, init_mongodb
from app.generation.generator import generate_answer
from app.ingestion.embedder import embed_query
from app.retrieval.confidence import compute_confidence
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.retrieval.text_search import text_search
from app.retrieval.vector_search import SearchResult, vector_search

CONFIDENCE_THRESHOLD = 0.35
EVAL_METRIC_THRESHOLD = 0.60

pytestmark = pytest.mark.evaluation


class GeminiEvaluationModel(DeepEvalBaseLLM):
    """
    Custom DeepEval LLM wrapper for Google Gemini models to handle evaluations
    without requiring an OpenAI API key wrapper. Tracks and handles rate limits cleanly.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.client = genai.Client()

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        """Synchronous generation wrapper with linear backoff safety buffers."""
        for attempt in range(4):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                return response.text
            except ClientError as e:
                if e.code == 429 and attempt < 3:
                    time.sleep(40 * (attempt + 1))
                    continue
                raise

    async def a_generate(self, prompt: str) -> str:
        """Asynchronous generation hook executed under the hood by DeepEval metrics."""
        for attempt in range(4):
            try:
                # Run the synchronous API client execution thread safely inside an async executor block
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                    )
                )
                return response.text
            except ClientError as e:
                if e.code == 429 and attempt < 3:
                    await asyncio.sleep(40 * (attempt + 1))
                    continue
                raise

    def get_model_name(self) -> str:
        return self.model_name


@pytest.fixture(scope="module", autouse=True)
@pytest.mark.asyncio(loop_scope="module")
async def setup_database_lifecycle():
    """Initialise the real Motor/MongoDB connection for the module's test run."""
    await init_mongodb()
    yield
    await close_mongodb()


class RetrievalPipeline:
    """Thin orchestration wrapper replicating the exact microservice sequence."""

    async def execute(self, query: str, limit: int = 5) -> list[SearchResult]:
        query_embedding = await embed_query(query)
        vector_results = await vector_search(query_embedding, top_k=20)
        text_results = await text_search(query, top_k=20)

        fused_candidates = reciprocal_rank_fusion(
            vector_results=vector_results,
            text_results=text_results,
            top_k=40,
        )

        reranked_chunks = await rerank(query=query, candidates=fused_candidates, top_k=limit)
        return reranked_chunks

    def calculate_confidence(self, chunks: list[SearchResult]) -> float:
        result = compute_confidence(reranked_chunks=chunks, threshold=CONFIDENCE_THRESHOLD)
        return result.score


@pytest.fixture(scope="module")
def retrieval_pipeline() -> RetrievalPipeline:
    return RetrievalPipeline()


@pytest.fixture(scope="module")
def deepeval_gemini_model() -> GeminiEvaluationModel:
    """Instantiates an isolated judge instance pointing explicitly to gemini-1.5-flash."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable is missing.")
    return GeminiEvaluationModel(model_name="gemini-2.5-flash")


@pytest.mark.asyncio(loop_scope="module")
async def test_insight_docket_rag_pipeline(
    retrieval_pipeline: RetrievalPipeline, deepeval_gemini_model: GeminiEvaluationModel
) -> None:
    """Evaluates pipeline precision, truthfulness and relevancy metrics natively using Gemini 1.5."""
    input_query = "What is the total revenue for Q3 and what were the primary drivers?"
    expected_output = (
        "$12.4M driven by enterprise software licensing and 15% growth in APAC cloud services."
    )

    # 1. Execute the real retrieval pipeline step
    retrieved_chunks: list[SearchResult] = await retrieval_pipeline.execute(
        query=input_query, limit=5
    )
    assert len(retrieved_chunks) > 0, (
        "No chunks retrieved — ensure a document has been ingested before running this eval."
    )

    retrieval_context = [chunk.content for chunk in retrieved_chunks]

    # 2. Execute generation pipeline using the available test infrastructure
    request_id = str(uuid.uuid4())
    generation_result = await generate_answer(
        query=input_query,
        chunks=retrieved_chunks,
        request_id=request_id,
    )
    actual_output = generation_result["answer"]

    # 3. Formulate the explicit test instance wrapper
    test_case = LLMTestCase(
        input=input_query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
    )

    # 4. Enforce separate sequential metric tracking to mitigate concurrent RPM spikes
    metrics = [
        FaithfulnessMetric(threshold=EVAL_METRIC_THRESHOLD, model=deepeval_gemini_model, async_mode=False),
        AnswerRelevancyMetric(threshold=EVAL_METRIC_THRESHOLD, model=deepeval_gemini_model, async_mode=False),
        ContextualRelevancyMetric(threshold=EVAL_METRIC_THRESHOLD, model=deepeval_gemini_model, async_mode=False),
        ContextualPrecisionMetric(threshold=EVAL_METRIC_THRESHOLD, model=deepeval_gemini_model, async_mode=False),
    ]

    for i, metric in enumerate(metrics):
        if i > 0:
            # Pacing delay to guarantee clean request balancing on standard tiered setups
            await asyncio.sleep(12)

        results = evaluate(
            test_cases=[test_case],
            metrics=[metric],
            # print_results=False,
            # ignore_errors=False,
        )
        
        # Verify the target structural result container flags a passing score
        assert results.test_results[0].success


@pytest.mark.asyncio(loop_scope="module")
async def test_confidence_threshold_fallback(retrieval_pipeline: RetrievalPipeline) -> None:
    """Ensures an out-of-domain query produces a confidence score below the configured fallback threshold."""
    out_of_domain_query = "How do I bake a sourdough bread loaf from scratch?"

    retrieved_chunks = await retrieval_pipeline.execute(query=out_of_domain_query, limit=5)
    assert len(retrieved_chunks) > 0
    confidence_score = retrieval_pipeline.calculate_confidence(retrieved_chunks)

    assert confidence_score < CONFIDENCE_THRESHOLD, (
        f"Expected confidence below {CONFIDENCE_THRESHOLD} for out-of-domain input, got {confidence_score}"
    )