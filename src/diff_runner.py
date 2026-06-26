"""Phase 3 Step 3: Comparison and regression detection between two eval runs."""

from __future__ import annotations

from typing import Literal

from scipy.stats import binomtest

from pydantic import BaseModel

from src.eval_runner import EvalRunResult


class RegressionEvent(BaseModel):
    test_case_id: str
    question: str
    difficulty: str
    category: str
    previous_composite: float
    current_composite: float
    score_delta: float
    dimensions_that_regressed: list[str]
    previous_answer: str
    current_answer: str


class ImprovementEvent(BaseModel):
    test_case_id: str
    question: str
    difficulty: str
    category: str
    previous_composite: float
    current_composite: float


class EvalDiff(BaseModel):
    baseline_run_id: str
    current_run_id: str
    baseline_pass_rate: float
    current_pass_rate: float
    overall_delta: float
    regressions: list[RegressionEvent]
    improvements: list[ImprovementEvent]
    dimension_deltas: dict[str, float | None]   # None when either side is all-skipped
    difficulty_deltas: dict[str, float]
    category_deltas: dict[str, float]
    is_statistically_significant: bool
    p_value: float
    severity: Literal["clean", "warning", "critical"]


def _delta_or_none(cur: float | None, bas: float | None) -> float | None:
    if cur is None or bas is None:
        return None
    return cur - bas


def _severity(overall_delta: float, regressions: list[RegressionEvent]) -> Literal["clean", "warning", "critical"]:
    hard_adversarial = sum(1 for r in regressions if r.difficulty in ("hard", "adversarial"))
    if overall_delta < -0.08 or hard_adversarial >= 3:
        return "critical"
    if overall_delta < -0.03 or len(regressions) >= 2:
        return "warning"
    return "clean"


def diff_eval_runs(baseline: EvalRunResult, current: EvalRunResult) -> EvalDiff:
    baseline_by_id = {r.test_case_id: r for r in baseline.test_results}
    current_by_id = {r.test_case_id: r for r in current.test_results}

    regressions: list[RegressionEvent] = []
    improvements: list[ImprovementEvent] = []

    for case_id, cur in current_by_id.items():
        if case_id not in baseline_by_id:
            continue
        bas = baseline_by_id[case_id]
        delta = cur.composite_score - bas.composite_score

        if (bas.passed and not cur.passed) or delta < -0.10:
            regressed_dims = [
                dim for dim in cur.scores
                if not cur.scores[dim].skipped
                and not bas.scores[dim].skipped
                and cur.scores[dim].score < bas.scores[dim].score - 0.05
            ]
            regressions.append(RegressionEvent(
                test_case_id=case_id,
                question=cur.question,
                difficulty=cur.difficulty,
                category=cur.failure_mode_category,
                previous_composite=bas.composite_score,
                current_composite=cur.composite_score,
                score_delta=delta,
                dimensions_that_regressed=regressed_dims,
                previous_answer=bas.answer.answer,
                current_answer=cur.answer.answer,
            ))
        elif not bas.passed and cur.passed:
            improvements.append(ImprovementEvent(
                test_case_id=case_id,
                question=cur.question,
                difficulty=cur.difficulty,
                category=cur.failure_mode_category,
                previous_composite=bas.composite_score,
                current_composite=cur.composite_score,
            ))

    all_dims = set(current.pass_rate_by_dimension) | set(baseline.pass_rate_by_dimension)
    dimension_deltas = {
        dim: _delta_or_none(
            current.pass_rate_by_dimension.get(dim),
            baseline.pass_rate_by_dimension.get(dim),
        )
        for dim in all_dims
    }

    all_diffs = set(current.pass_rate_by_difficulty) | set(baseline.pass_rate_by_difficulty)
    difficulty_deltas = {
        diff: current.pass_rate_by_difficulty.get(diff, 0.0) - baseline.pass_rate_by_difficulty.get(diff, 0.0)
        for diff in all_diffs
    }

    all_cats = set(current.pass_rate_by_category) | set(baseline.pass_rate_by_category)
    category_deltas = {
        cat: current.pass_rate_by_category.get(cat, 0.0) - baseline.pass_rate_by_category.get(cat, 0.0)
        for cat in all_cats
    }

    n = len(current.test_results)
    k = sum(1 for r in current.test_results if r.passed)
    p_baseline = baseline.overall_pass_rate
    if n == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(k, n, p_baseline, alternative="two-sided").pvalue)

    overall_delta = current.overall_pass_rate - baseline.overall_pass_rate

    return EvalDiff(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        baseline_pass_rate=baseline.overall_pass_rate,
        current_pass_rate=current.overall_pass_rate,
        overall_delta=overall_delta,
        regressions=regressions,
        improvements=improvements,
        dimension_deltas=dimension_deltas,
        difficulty_deltas=difficulty_deltas,
        category_deltas=category_deltas,
        is_statistically_significant=p_value < 0.05,
        p_value=p_value,
        severity=_severity(overall_delta, regressions),
    )
