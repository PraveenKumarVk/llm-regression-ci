"""Phase 3 Step 2: Evaluation runner."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from src.earnings_qa import answer_earnings_question
from src.golden_loader import load_dataset
from src.models import EarningsAnswer, EarningsQuery, GoldenTestCase, ScoreResult
from src.scorer import (
    score_citation_accuracy,
    score_faithfulness_llm_judge,
    score_numerical_accuracy,
    score_refusal_correctness,
    score_temporal_precision,
)

_WEIGHTS: dict[str, float] = {
    "numerical_accuracy": 0.30,
    "refusal_correctness": 0.20,
    "faithfulness": 0.25,
    "citation_accuracy": 0.15,
    "temporal_precision": 0.10,
}

_DIMENSIONS: list[str] = list(_WEIGHTS)

# gpt-4o pricing, per token, approximate
_COST_PER_INPUT_TOKEN: float = 0.0000025
_COST_PER_OUTPUT_TOKEN: float = 0.000010


class TestCaseResult(BaseModel):
    test_case_id: str
    question: str
    difficulty: str
    failure_mode_category: str
    answer: EarningsAnswer
    scores: dict[str, ScoreResult]
    composite_score: float
    passed: bool


class EvalRunResult(BaseModel):
    run_id: str
    prompt_version: str
    model: str
    dataset_version: str
    timestamp: datetime
    test_results: list[TestCaseResult]
    overall_pass_rate: float
    pass_rate_by_dimension: dict[str, float | None]  # None = all cases skipped
    pass_rate_by_difficulty: dict[str, float]
    pass_rate_by_category: dict[str, float]
    total_cost_usd: float
    avg_latency_ms: float


def _composite_score(scores: dict[str, ScoreResult]) -> float:
    """Weighted average over non-skipped dimensions with weight renormalization."""
    active = {dim: w for dim, w in _WEIGHTS.items() if not scores[dim].skipped}
    total_weight = sum(active.values())
    if not total_weight:
        return 0.0
    return sum(scores[dim].score * w / total_weight for dim, w in active.items())


def _case_passed(composite: float, scores: dict[str, ScoreResult]) -> bool:
    """composite >= 0.85 AND refusal_correctness hard gate (skipped counts as passing)."""
    refusal = scores["refusal_correctness"]
    return composite >= 0.85 and (refusal.skipped or refusal.passed)


def _dimension_pass_rate(results: list[TestCaseResult], dim: str) -> float | None:
    """Pass rate for one dimension; None when every case for that dimension is skipped."""
    eligible = [r for r in results if not r.scores[dim].skipped]
    if not eligible:
        return None
    return sum(1 for r in eligible if r.scores[dim].passed) / len(eligible)


async def _run_single_case(
    tc: GoldenTestCase,
    prompt_version: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> TestCaseResult:
    async with semaphore:
        query = EarningsQuery(
            question=tc.question,
            context_chunk=tc.context_chunk,
            document_id=tc.document_id,
            chunk_id=tc.chunk_id,
            time_period=tc.time_period,
        )
        answer = await answer_earnings_question(query, prompt_version, model)
        score_results = await asyncio.gather(
            score_numerical_accuracy(tc, answer),
            score_refusal_correctness(tc, answer),
            score_faithfulness_llm_judge(tc, answer),
            score_citation_accuracy(tc, answer),
            score_temporal_precision(tc, answer),
        )
        scores = {s.dimension: s for s in score_results}
        composite = _composite_score(scores)
        return TestCaseResult(
            test_case_id=tc.id,
            question=tc.question,
            difficulty=tc.difficulty,
            failure_mode_category=tc.failure_mode_category,
            answer=answer,
            scores=scores,
            composite_score=composite,
            passed=_case_passed(composite, scores),
        )


async def run_eval(
    dataset_path: str | Path,
    prompt_version: str = "v1.0.0",
    model: str = "gpt-4o",
    concurrency: int = 5,
) -> EvalRunResult:
    dataset = load_dataset(Path(dataset_path))
    semaphore = asyncio.Semaphore(concurrency)
    results: list[TestCaseResult] = list(
        await asyncio.gather(
            *[_run_single_case(tc, prompt_version, model, semaphore) for tc in dataset.cases]
        )
    )

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)

    pass_rate_by_difficulty: dict[str, float] = {}
    for diff in ("easy", "medium", "hard", "adversarial"):
        bucket = [r for r in results if r.difficulty == diff]
        if bucket:
            pass_rate_by_difficulty[diff] = sum(1 for r in bucket if r.passed) / len(bucket)

    pass_rate_by_category: dict[str, float] = {}
    for cat in {r.failure_mode_category for r in results}:
        bucket = [r for r in results if r.failure_mode_category == cat]
        pass_rate_by_category[cat] = sum(1 for r in bucket if r.passed) / len(bucket)

    return EvalRunResult(
        run_id=str(uuid.uuid4()),
        prompt_version=prompt_version,
        model=model,
        dataset_version=dataset.version,
        timestamp=datetime.now(tz=timezone.utc),
        test_results=results,
        overall_pass_rate=passed_count / total if total else 0.0,
        pass_rate_by_dimension={dim: _dimension_pass_rate(results, dim) for dim in _DIMENSIONS},
        pass_rate_by_difficulty=pass_rate_by_difficulty,
        pass_rate_by_category=pass_rate_by_category,
        total_cost_usd=sum(
            r.answer.input_tokens * _COST_PER_INPUT_TOKEN
            + r.answer.output_tokens * _COST_PER_OUTPUT_TOKEN
            for r in results
        ),
        avg_latency_ms=sum(r.answer.latency_ms for r in results) / total if total else 0.0,
    )
