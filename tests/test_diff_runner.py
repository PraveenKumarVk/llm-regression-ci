"""Tests for Phase 3 Step 3: diff_eval_runs regression detection."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from scipy.stats import binomtest

from src.diff_runner import (
    EvalDiff,
    ImprovementEvent,
    RegressionEvent,
    _delta_or_none,
    _severity,
    diff_eval_runs,
)
from src.eval_runner import EvalRunResult, TestCaseResult
from src.models import EarningsAnswer, ScoreResult

_DIMS = [
    "numerical_accuracy",
    "refusal_correctness",
    "faithfulness",
    "citation_accuracy",
    "temporal_precision",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ans() -> EarningsAnswer:
    return EarningsAnswer(
        answer="test answer",
        citation=None,
        is_refusal=False,
        raw_response="test",
        prompt_version="v1.0.0",
        content_hash="abc",
        model="gpt-4o",
        input_tokens=100,
        output_tokens=50,
        latency_ms=500.0,
    )


def _score(dim: str, score: float = 1.0, passed: bool = True, skipped: bool = False) -> ScoreResult:
    return ScoreResult(score=score, passed=passed, reasoning="", dimension=dim, skipped=skipped)  # type: ignore[arg-type]


def _all_scores(score: float = 1.0, passed: bool = True) -> dict[str, ScoreResult]:
    return {dim: _score(dim, score=score, passed=passed) for dim in _DIMS}


def _tcr(
    test_case_id: str = "TC_001",
    composite: float = 1.0,
    passed: bool = True,
    difficulty: str = "easy",
    category: str = "numerical_extraction",
    scores: dict[str, ScoreResult] | None = None,
    answer_text: str = "test answer",
) -> TestCaseResult:
    ans = EarningsAnswer(
        answer=answer_text,
        citation=None,
        is_refusal=False,
        raw_response=answer_text,
        prompt_version="v1.0.0",
        content_hash="abc",
        model="gpt-4o",
        input_tokens=100,
        output_tokens=50,
        latency_ms=500.0,
    )
    return TestCaseResult(
        test_case_id=test_case_id,
        question=f"Question for {test_case_id}",
        difficulty=difficulty,
        failure_mode_category=category,
        answer=ans,
        scores=scores if scores is not None else _all_scores(score=composite, passed=passed),
        composite_score=composite,
        passed=passed,
    )


def _make_run(
    run_id: str = "run-a",
    test_results: list[TestCaseResult] | None = None,
    overall_pass_rate: float = 1.0,
    pass_rate_by_dimension: dict[str, float | None] | None = None,
    pass_rate_by_difficulty: dict[str, float] | None = None,
    pass_rate_by_category: dict[str, float] | None = None,
) -> EvalRunResult:
    results = test_results or []
    return EvalRunResult(
        run_id=run_id,
        prompt_version="v1.0.0",
        model="gpt-4o",
        dataset_version="v1.0.0",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        test_results=results,
        overall_pass_rate=overall_pass_rate,
        pass_rate_by_dimension=pass_rate_by_dimension or {dim: overall_pass_rate for dim in _DIMS},
        pass_rate_by_difficulty=pass_rate_by_difficulty or {"easy": overall_pass_rate},
        pass_rate_by_category=pass_rate_by_category or {"numerical_extraction": overall_pass_rate},
        total_cost_usd=0.01,
        avg_latency_ms=500.0,
    )


# ---------------------------------------------------------------------------
# _delta_or_none
# ---------------------------------------------------------------------------


class TestDeltaOrNone:
    def test_both_floats(self):
        assert _delta_or_none(0.8, 0.6) == pytest.approx(0.2)

    def test_current_none(self):
        assert _delta_or_none(None, 0.8) is None

    def test_baseline_none(self):
        assert _delta_or_none(0.8, None) is None

    def test_both_none(self):
        assert _delta_or_none(None, None) is None

    def test_zero_delta(self):
        assert _delta_or_none(0.75, 0.75) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _severity
# ---------------------------------------------------------------------------


class TestSeverity:
    def _reg(self, difficulty: str = "easy") -> RegressionEvent:
        return RegressionEvent(
            test_case_id="TC_001",
            question="q",
            difficulty=difficulty,
            category="numerical_extraction",
            previous_composite=0.9,
            current_composite=0.7,
            score_delta=-0.2,
            dimensions_that_regressed=["numerical_accuracy"],
            previous_answer="prev",
            current_answer="curr",
        )

    def test_clean_no_regressions(self):
        assert _severity(0.0, []) == "clean"

    def test_clean_small_delta(self):
        assert _severity(-0.02, []) == "clean"

    def test_warning_two_regressions(self):
        assert _severity(0.0, [self._reg(), self._reg()]) == "warning"

    def test_warning_delta_exceeds_threshold(self):
        assert _severity(-0.04, []) == "warning"

    def test_warning_threshold_is_strict(self):
        # < -0.03 triggers warning; exactly -0.03 does not
        assert _severity(-0.031, []) == "warning"
        assert _severity(-0.03, []) == "clean"

    def test_critical_large_delta(self):
        assert _severity(-0.09, []) == "critical"

    def test_critical_threshold_is_strict(self):
        # < -0.08 triggers critical; exactly -0.08 does not
        assert _severity(-0.081, []) == "critical"
        assert _severity(-0.08, []) == "warning"

    def test_critical_three_hard_adversarial_regressions(self):
        regs = [self._reg("hard"), self._reg("adversarial"), self._reg("hard")]
        assert _severity(0.0, regs) == "critical"

    def test_critical_not_triggered_by_easy_regressions(self):
        regs = [self._reg("easy"), self._reg("easy"), self._reg("easy")]
        # 3 easy regressions → warning (len >= 2) not critical
        assert _severity(0.0, regs) == "warning"

    def test_critical_wins_over_warning(self):
        # Both warning and critical conditions met → critical
        regs = [self._reg("hard"), self._reg("adversarial"), self._reg("hard")]
        assert _severity(-0.10, regs) == "critical"


# ---------------------------------------------------------------------------
# diff_eval_runs — regression/improvement detection
# ---------------------------------------------------------------------------


class TestRegressionDetection:
    def test_pass_to_fail_flagged_as_regression(self):
        bas = _make_run("run-a", [_tcr("TC_001", composite=0.9, passed=True)])
        cur = _make_run("run-b", [_tcr("TC_001", composite=0.5, passed=False)])
        diff = diff_eval_runs(bas, cur)
        assert len(diff.regressions) == 1
        assert diff.regressions[0].test_case_id == "TC_001"
        assert diff.regressions[0].score_delta == pytest.approx(-0.4)

    def test_large_score_drop_flagged_even_if_both_fail(self):
        # Neither passed, but composite dropped 0.20 > threshold of 0.10
        bas = _make_run("run-a", [_tcr("TC_001", composite=0.7, passed=False)])
        cur = _make_run("run-b", [_tcr("TC_001", composite=0.5, passed=False)])
        diff = diff_eval_runs(bas, cur)
        assert len(diff.regressions) == 1

    def test_small_drop_not_flagged(self):
        bas = _make_run("run-a", [_tcr("TC_001", composite=0.9, passed=True)])
        cur = _make_run("run-b", [_tcr("TC_001", composite=0.85, passed=True)])
        diff = diff_eval_runs(bas, cur)
        assert len(diff.regressions) == 0

    def test_case_not_in_baseline_skipped(self):
        bas = _make_run("run-a", [])
        cur = _make_run("run-b", [_tcr("TC_999", composite=0.0, passed=False)])
        diff = diff_eval_runs(bas, cur)
        assert len(diff.regressions) == 0

    def test_dimensions_that_regressed_populated(self):
        bas_scores = _all_scores()
        cur_scores = {**_all_scores(), "numerical_accuracy": _score("numerical_accuracy", score=0.5, passed=False)}
        bas = _make_run("run-a", [_tcr("TC_001", composite=0.9, passed=True, scores=bas_scores)])
        cur = _make_run("run-b", [_tcr("TC_001", composite=0.5, passed=False, scores=cur_scores)])
        diff = diff_eval_runs(bas, cur)
        assert "numerical_accuracy" in diff.regressions[0].dimensions_that_regressed

    def test_skipped_dimension_excluded_from_regressed_dims(self):
        bas_scores = _all_scores()
        cur_scores = {
            **_all_scores(),
            "temporal_precision": _score("temporal_precision", score=0.0, passed=False, skipped=True),
        }
        bas = _make_run("run-a", [_tcr("TC_001", composite=0.9, passed=True, scores=bas_scores)])
        cur = _make_run("run-b", [_tcr("TC_001", composite=0.5, passed=False, scores=cur_scores)])
        diff = diff_eval_runs(bas, cur)
        assert "temporal_precision" not in diff.regressions[0].dimensions_that_regressed

    def test_answer_text_captured_in_regression_event(self):
        bas = _make_run("run-a", [_tcr("TC_001", composite=0.9, passed=True, answer_text="old answer")])
        cur = _make_run("run-b", [_tcr("TC_001", composite=0.5, passed=False, answer_text="new answer")])
        diff = diff_eval_runs(bas, cur)
        assert diff.regressions[0].previous_answer == "old answer"
        assert diff.regressions[0].current_answer == "new answer"


class TestImprovementDetection:
    def test_fail_to_pass_flagged_as_improvement(self):
        bas = _make_run("run-a", [_tcr("TC_001", composite=0.5, passed=False)])
        cur = _make_run("run-b", [_tcr("TC_001", composite=0.95, passed=True)])
        diff = diff_eval_runs(bas, cur)
        assert len(diff.improvements) == 1
        assert diff.improvements[0].test_case_id == "TC_001"

    def test_pass_to_pass_not_improvement(self):
        bas = _make_run("run-a", [_tcr("TC_001", composite=0.9, passed=True)])
        cur = _make_run("run-b", [_tcr("TC_001", composite=1.0, passed=True)])
        diff = diff_eval_runs(bas, cur)
        assert len(diff.improvements) == 0

    def test_fail_to_fail_not_improvement(self):
        bas = _make_run("run-a", [_tcr("TC_001", composite=0.5, passed=False)])
        cur = _make_run("run-b", [_tcr("TC_001", composite=0.6, passed=False)])
        diff = diff_eval_runs(bas, cur)
        assert len(diff.improvements) == 0

    def test_case_not_in_both_not_improvement(self):
        bas = _make_run("run-a", [])
        cur = _make_run("run-b", [_tcr("TC_001", composite=1.0, passed=True)])
        diff = diff_eval_runs(bas, cur)
        assert len(diff.improvements) == 0


# ---------------------------------------------------------------------------
# diff_eval_runs — aggregate fields
# ---------------------------------------------------------------------------


class TestAggregateFields:
    def test_run_ids_propagated(self):
        bas = _make_run("baseline-id")
        cur = _make_run("current-id")
        diff = diff_eval_runs(bas, cur)
        assert diff.baseline_run_id == "baseline-id"
        assert diff.current_run_id == "current-id"

    def test_overall_delta_computed(self):
        bas = _make_run(overall_pass_rate=0.80)
        cur = _make_run(overall_pass_rate=0.70)
        diff = diff_eval_runs(bas, cur)
        assert diff.overall_delta == pytest.approx(-0.10)

    def test_dimension_deltas_computed(self):
        bas = _make_run(pass_rate_by_dimension={"numerical_accuracy": 0.8, "refusal_correctness": 0.9,
                                                "faithfulness": 0.85, "citation_accuracy": 0.75, "temporal_precision": 1.0})
        cur = _make_run(pass_rate_by_dimension={"numerical_accuracy": 0.6, "refusal_correctness": 0.9,
                                                "faithfulness": 0.85, "citation_accuracy": 0.75, "temporal_precision": 1.0})
        diff = diff_eval_runs(bas, cur)
        assert diff.dimension_deltas["numerical_accuracy"] == pytest.approx(-0.2)
        assert diff.dimension_deltas["refusal_correctness"] == pytest.approx(0.0)

    def test_dimension_delta_none_when_either_side_skipped(self):
        bas = _make_run(pass_rate_by_dimension={"numerical_accuracy": 0.8, "refusal_correctness": 0.9,
                                                "faithfulness": 0.85, "citation_accuracy": 0.75, "temporal_precision": None})
        cur = _make_run(pass_rate_by_dimension={"numerical_accuracy": 0.6, "refusal_correctness": 0.9,
                                                "faithfulness": 0.85, "citation_accuracy": 0.75, "temporal_precision": None})
        diff = diff_eval_runs(bas, cur)
        assert diff.dimension_deltas["temporal_precision"] is None

    def test_difficulty_deltas_computed(self):
        bas = _make_run(pass_rate_by_difficulty={"easy": 1.0, "hard": 0.6})
        cur = _make_run(pass_rate_by_difficulty={"easy": 0.8, "hard": 0.6})
        diff = diff_eval_runs(bas, cur)
        assert diff.difficulty_deltas["easy"] == pytest.approx(-0.2)
        assert diff.difficulty_deltas["hard"] == pytest.approx(0.0)

    def test_difficulty_only_in_baseline_shows_negative_delta(self):
        bas = _make_run(pass_rate_by_difficulty={"easy": 1.0, "hard": 0.5})
        cur = _make_run(pass_rate_by_difficulty={"easy": 1.0})
        diff = diff_eval_runs(bas, cur)
        # hard only in baseline; current defaults to 0
        assert diff.difficulty_deltas["hard"] == pytest.approx(-0.5)

    def test_category_deltas_computed(self):
        bas = _make_run(pass_rate_by_category={"numerical_extraction": 0.9, "faithfulness": 0.7})
        cur = _make_run(pass_rate_by_category={"numerical_extraction": 0.7, "faithfulness": 0.9})
        diff = diff_eval_runs(bas, cur)
        assert diff.category_deltas["numerical_extraction"] == pytest.approx(-0.2)
        assert diff.category_deltas["faithfulness"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------


class TestStatisticalSignificance:
    def test_significant_when_p_below_005(self):
        # n=50, k=50, baseline=0.5 → very unlikely → significant
        results = [_tcr(f"TC_{i:03d}", passed=True) for i in range(50)]
        bas = _make_run(overall_pass_rate=0.5)
        cur = _make_run(test_results=results, overall_pass_rate=1.0)
        diff = diff_eval_runs(bas, cur)
        assert diff.is_statistically_significant is True
        assert diff.p_value < 0.05

    def test_not_significant_when_consistent_with_baseline(self):
        # n=10, k=5, baseline=0.5 → consistent → not significant
        results = [_tcr(f"TC_{i:03d}", passed=i < 5) for i in range(10)]
        bas = _make_run(overall_pass_rate=0.5)
        cur = _make_run(test_results=results, overall_pass_rate=0.5)
        diff = diff_eval_runs(bas, cur)
        assert diff.is_statistically_significant is False
        assert diff.p_value > 0.05

    def test_p_value_matches_scipy_directly(self):
        results = [_tcr(f"TC_{i:03d}", passed=i < 7) for i in range(10)]
        bas = _make_run(overall_pass_rate=0.5)
        cur = _make_run(test_results=results, overall_pass_rate=0.7)
        diff = diff_eval_runs(bas, cur)
        expected = float(binomtest(7, 10, 0.5, alternative="two-sided").pvalue)
        assert diff.p_value == pytest.approx(expected)

    def test_empty_test_results_not_significant(self):
        bas = _make_run(overall_pass_rate=0.8)
        cur = _make_run(test_results=[], overall_pass_rate=0.0)
        diff = diff_eval_runs(bas, cur)
        assert diff.is_statistically_significant is False
        assert diff.p_value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Severity end-to-end via diff_eval_runs
# ---------------------------------------------------------------------------


class TestSeverityEndToEnd:
    def test_clean_when_no_regressions_and_small_delta(self):
        bas = _make_run(overall_pass_rate=0.90)
        cur = _make_run(overall_pass_rate=0.89)
        diff = diff_eval_runs(bas, cur)
        assert diff.severity == "clean"

    def test_warning_when_two_regressions(self):
        baseline_results = [
            _tcr("TC_001", composite=0.9, passed=True),
            _tcr("TC_002", composite=0.9, passed=True),
        ]
        current_results = [
            _tcr("TC_001", composite=0.5, passed=False),
            _tcr("TC_002", composite=0.5, passed=False),
        ]
        bas = _make_run("run-a", baseline_results, overall_pass_rate=1.0)
        cur = _make_run("run-b", current_results, overall_pass_rate=0.0)
        diff = diff_eval_runs(bas, cur)
        assert len(diff.regressions) == 2
        assert diff.severity in ("warning", "critical")

    def test_critical_when_large_overall_delta(self):
        bas = _make_run(overall_pass_rate=0.95)
        cur = _make_run(overall_pass_rate=0.85)
        diff = diff_eval_runs(bas, cur)
        assert diff.severity == "critical"
