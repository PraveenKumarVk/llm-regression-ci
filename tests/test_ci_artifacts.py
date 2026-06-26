"""Tests for src/ci_artifacts.py: build_summary and write_severity."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ci_artifacts import build_summary, write_severity
from src.diff_runner import EvalDiff, ImprovementEvent, RegressionEvent
from src.eval_runner import EvalRunResult

_DIMS = [
    "numerical_accuracy",
    "refusal_correctness",
    "faithfulness",
    "citation_accuracy",
    "temporal_precision",
]

_REPORT_URL = "https://reports.example.com/run-001.html"

_counter = 0


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def current_run() -> EvalRunResult:
    global _counter
    _counter += 1
    return EvalRunResult(
        run_id=f"run-{_counter:04d}",
        prompt_version="v1.0.0",
        model="gpt-4o",
        dataset_version="v1.0.0",
        timestamp=datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc),
        test_results=[],
        overall_pass_rate=0.85,
        pass_rate_by_dimension={d: 0.85 for d in _DIMS},
        pass_rate_by_difficulty={"easy": 0.85},
        pass_rate_by_category={"numerical_extraction": 0.85},
        total_cost_usd=0.05,
        avg_latency_ms=420.0,
    )


@pytest.fixture()
def diff() -> EvalDiff:
    return EvalDiff(
        baseline_run_id="run-baseline",
        current_run_id="run-current",
        baseline_pass_rate=0.90,
        current_pass_rate=0.85,
        overall_delta=-0.05,
        regressions=[
            RegressionEvent(
                test_case_id="TC_001",
                question="What was net income?",
                difficulty="easy",
                category="numerical_extraction",
                previous_composite=0.90,
                current_composite=0.60,
                score_delta=-0.30,
                dimensions_that_regressed=["numerical_accuracy"],
                previous_answer="$12B",
                current_answer="NOT_IN_DOCUMENT",
            )
        ],
        improvements=[
            ImprovementEvent(
                test_case_id="TC_002",
                question="What was revenue?",
                difficulty="medium",
                category="temporal_precision",
                previous_composite=0.60,
                current_composite=0.92,
            )
        ],
        dimension_deltas={d: -0.02 for d in _DIMS},
        difficulty_deltas={"easy": -0.05},
        category_deltas={"numerical_extraction": -0.05},
        is_statistically_significant=True,
        p_value=0.03,
        severity="warning",
    )


@pytest.fixture()
def drift_warning() -> str:
    return "SLOW DRIFT DETECTED: 7-run average dropped 10.0% (90.0% → 80.0%)."


@pytest.fixture()
def report_url() -> str:
    return _REPORT_URL


# ---------------------------------------------------------------------------
# build_summary: required fields (the two tests from the user's sketch)
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def test_summary_json_contains_required_ci_fields(
        self, current_run: EvalRunResult, diff: EvalDiff, drift_warning: str, report_url: str
    ):
        summary = build_summary(current_run, diff, drift_warning, report_url)
        required_fields = {
            "severity", "baseline_pass_rate", "current_pass_rate",
            "delta", "regression_count", "improvement_count",
            "dimension_summary", "drift_warning", "report_url",
        }
        assert required_fields.issubset(summary.keys())

    def test_severity_matches_diff(
        self, current_run: EvalRunResult, diff: EvalDiff, report_url: str
    ):
        summary = build_summary(current_run, diff, None, report_url)
        assert summary["severity"] == diff.severity

    def test_baseline_pass_rate_matches_diff(
        self, current_run: EvalRunResult, diff: EvalDiff, report_url: str
    ):
        summary = build_summary(current_run, diff, None, report_url)
        assert summary["baseline_pass_rate"] == diff.baseline_pass_rate

    def test_current_pass_rate_matches_diff(
        self, current_run: EvalRunResult, diff: EvalDiff, report_url: str
    ):
        summary = build_summary(current_run, diff, None, report_url)
        assert summary["current_pass_rate"] == diff.current_pass_rate

    def test_delta_matches_diff(
        self, current_run: EvalRunResult, diff: EvalDiff, report_url: str
    ):
        summary = build_summary(current_run, diff, None, report_url)
        assert summary["delta"] == diff.overall_delta

    def test_regression_count_is_len_of_diff_regressions(
        self, current_run: EvalRunResult, diff: EvalDiff, report_url: str
    ):
        summary = build_summary(current_run, diff, None, report_url)
        assert summary["regression_count"] == len(diff.regressions)

    def test_improvement_count_is_len_of_diff_improvements(
        self, current_run: EvalRunResult, diff: EvalDiff, report_url: str
    ):
        summary = build_summary(current_run, diff, None, report_url)
        assert summary["improvement_count"] == len(diff.improvements)

    def test_drift_warning_passed_through(
        self, current_run: EvalRunResult, diff: EvalDiff, drift_warning: str, report_url: str
    ):
        summary = build_summary(current_run, diff, drift_warning, report_url)
        assert summary["drift_warning"] == drift_warning

    def test_drift_warning_none_when_not_provided(
        self, current_run: EvalRunResult, diff: EvalDiff, report_url: str
    ):
        summary = build_summary(current_run, diff, None, report_url)
        assert summary["drift_warning"] is None

    def test_report_url_passed_through(
        self, current_run: EvalRunResult, diff: EvalDiff, report_url: str
    ):
        summary = build_summary(current_run, diff, None, report_url)
        assert summary["report_url"] == report_url

    # dimension_summary structure
    def test_dimension_summary_has_all_five_dims(
        self, current_run: EvalRunResult, diff: EvalDiff, report_url: str
    ):
        summary = build_summary(current_run, diff, None, report_url)
        assert set(summary["dimension_summary"].keys()) == set(_DIMS)

    def test_dimension_summary_entries_have_pass_rate_and_delta(
        self, current_run: EvalRunResult, diff: EvalDiff, report_url: str
    ):
        summary = build_summary(current_run, diff, None, report_url)
        for dim, entry in summary["dimension_summary"].items():
            assert "pass_rate" in entry, f"{dim} missing pass_rate"
            assert "delta" in entry, f"{dim} missing delta"

    def test_dimension_pass_rate_matches_run(
        self, current_run: EvalRunResult, diff: EvalDiff, report_url: str
    ):
        summary = build_summary(current_run, diff, None, report_url)
        for dim in _DIMS:
            assert summary["dimension_summary"][dim]["pass_rate"] == current_run.pass_rate_by_dimension.get(dim)

    def test_dimension_delta_matches_diff(
        self, current_run: EvalRunResult, diff: EvalDiff, report_url: str
    ):
        summary = build_summary(current_run, diff, None, report_url)
        for dim in _DIMS:
            assert summary["dimension_summary"][dim]["delta"] == diff.dimension_deltas.get(dim)

    def test_none_dimension_pass_rate_serialises_as_none(
        self, diff: EvalDiff, report_url: str
    ):
        run = EvalRunResult(
            run_id="run-x",
            prompt_version="v1.0.0",
            model="gpt-4o",
            dataset_version="v1.0.0",
            timestamp=datetime(2024, 6, 15, tzinfo=timezone.utc),
            test_results=[],
            overall_pass_rate=0.85,
            pass_rate_by_dimension={"numerical_accuracy": None, **{d: 0.9 for d in _DIMS if d != "numerical_accuracy"}},
            pass_rate_by_difficulty={},
            pass_rate_by_category={},
            total_cost_usd=0.0,
            avg_latency_ms=0.0,
        )
        summary = build_summary(run, diff, None, report_url)
        assert summary["dimension_summary"]["numerical_accuracy"]["pass_rate"] is None

    def test_none_dimension_delta_serialises_as_none(
        self, current_run: EvalRunResult, report_url: str
    ):
        d = EvalDiff(
            baseline_run_id="b",
            current_run_id="c",
            baseline_pass_rate=0.90,
            current_pass_rate=0.85,
            overall_delta=-0.05,
            regressions=[],
            improvements=[],
            dimension_deltas={"numerical_accuracy": None, **{d: -0.01 for d in _DIMS if d != "numerical_accuracy"}},
            difficulty_deltas={},
            category_deltas={},
            is_statistically_significant=False,
            p_value=0.5,
            severity="clean",
        )
        summary = build_summary(current_run, d, None, report_url)
        assert summary["dimension_summary"]["numerical_accuracy"]["delta"] is None

    def test_return_value_is_json_serialisable(
        self, current_run: EvalRunResult, diff: EvalDiff, drift_warning: str, report_url: str
    ):
        import json
        summary = build_summary(current_run, diff, drift_warning, report_url)
        # Must not raise
        json.dumps(summary)

    def test_zero_regressions_and_improvements(
        self, current_run: EvalRunResult, report_url: str
    ):
        d = EvalDiff(
            baseline_run_id="b",
            current_run_id="c",
            baseline_pass_rate=0.90,
            current_pass_rate=0.90,
            overall_delta=0.0,
            regressions=[],
            improvements=[],
            dimension_deltas={d: 0.0 for d in _DIMS},
            difficulty_deltas={},
            category_deltas={},
            is_statistically_significant=False,
            p_value=1.0,
            severity="clean",
        )
        summary = build_summary(current_run, d, None, report_url)
        assert summary["regression_count"] == 0
        assert summary["improvement_count"] == 0


# ---------------------------------------------------------------------------
# write_severity (the test from the user's sketch + extra coverage)
# ---------------------------------------------------------------------------


class TestWriteSeverity:
    def test_severity_txt_is_exact_string(self, tmp_path: Path, diff: EvalDiff):
        write_severity(diff.severity, tmp_path / "severity.txt")
        content = (tmp_path / "severity.txt").read_text().strip()
        assert content in {"clean", "warning", "critical"}

    def test_write_clean(self, tmp_path: Path):
        write_severity("clean", tmp_path / "severity.txt")
        assert (tmp_path / "severity.txt").read_text() == "clean"

    def test_write_warning(self, tmp_path: Path):
        write_severity("warning", tmp_path / "sev.txt")
        assert (tmp_path / "sev.txt").read_text() == "warning"

    def test_write_critical(self, tmp_path: Path):
        write_severity("critical", tmp_path / "sev.txt")
        assert (tmp_path / "sev.txt").read_text() == "critical"

    def test_file_created_at_given_path(self, tmp_path: Path):
        target = tmp_path / "subdir" / "severity.txt"
        target.parent.mkdir()
        write_severity("clean", target)
        assert target.exists()

    def test_file_contains_no_trailing_newline(self, tmp_path: Path):
        write_severity("warning", tmp_path / "s.txt")
        raw = (tmp_path / "s.txt").read_bytes()
        assert raw == b"warning"
