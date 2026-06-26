"""Tests for Phase 4 Step 1: generate_html_report."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.diff_runner import EvalDiff, ImprovementEvent, RegressionEvent
from src.eval_runner import EvalRunResult
from src.report_generator import (
    _fmt_delta,
    _fmt_rate,
    generate_html_report,
)

_DIMS = [
    "numerical_accuracy",
    "refusal_correctness",
    "faithfulness",
    "citation_accuracy",
    "temporal_precision",
]

_TS = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)

_counter = 0


def _run(
    pass_rate: float = 0.90,
    *,
    run_id: str | None = None,
    prompt_version: str = "v1.0.0",
    model: str = "gpt-4o",
    dim_rates: dict[str, float | None] | None = None,
    timestamp: datetime = _TS,
) -> EvalRunResult:
    global _counter
    _counter += 1
    rates = dim_rates if dim_rates is not None else {d: pass_rate for d in _DIMS}
    return EvalRunResult(
        run_id=run_id or f"run-{_counter:04d}",
        prompt_version=prompt_version,
        model=model,
        dataset_version="v1.0.0",
        timestamp=timestamp,
        test_results=[],
        overall_pass_rate=pass_rate,
        pass_rate_by_dimension=rates,
        pass_rate_by_difficulty={"easy": pass_rate},
        pass_rate_by_category={"numerical_extraction": pass_rate},
        total_cost_usd=0.05,
        avg_latency_ms=420.0,
    )


def _regression(
    tc_id: str = "TC_001",
    question: str = "What was net income?",
    difficulty: str = "easy",
    category: str = "numerical_extraction",
    prev: float = 0.90,
    cur: float = 0.60,
    dims: list[str] | None = None,
    prev_answer: str = "Net income was $12B.",
    cur_answer: str = "NOT_IN_DOCUMENT",
) -> RegressionEvent:
    return RegressionEvent(
        test_case_id=tc_id,
        question=question,
        difficulty=difficulty,
        category=category,
        previous_composite=prev,
        current_composite=cur,
        score_delta=cur - prev,
        dimensions_that_regressed=dims if dims is not None else ["numerical_accuracy"],
        previous_answer=prev_answer,
        current_answer=cur_answer,
    )


def _improvement(
    tc_id: str = "TC_002",
    question: str = "What was revenue?",
    difficulty: str = "medium",
    category: str = "temporal_precision",
    prev: float = 0.60,
    cur: float = 0.90,
) -> ImprovementEvent:
    return ImprovementEvent(
        test_case_id=tc_id,
        question=question,
        difficulty=difficulty,
        category=category,
        previous_composite=prev,
        current_composite=cur,
    )


def _diff(
    regressions: list[RegressionEvent] | None = None,
    improvements: list[ImprovementEvent] | None = None,
    overall_delta: float = -0.05,
    severity: str = "warning",
    dim_deltas: dict[str, float | None] | None = None,
    p_value: float = 0.03,
    significant: bool = True,
) -> EvalDiff:
    return EvalDiff(
        baseline_run_id="run-baseline",
        current_run_id="run-current",
        baseline_pass_rate=0.90,
        current_pass_rate=0.85,
        overall_delta=overall_delta,
        regressions=regressions or [],
        improvements=improvements or [],
        dimension_deltas=dim_deltas or {d: -0.02 for d in _DIMS},
        difficulty_deltas={"easy": overall_delta},
        category_deltas={"numerical_extraction": overall_delta},
        is_statistically_significant=significant,
        p_value=p_value,
        severity=severity,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Helper: _fmt_rate and _fmt_delta
# ---------------------------------------------------------------------------


class TestFmtHelpers:
    def test_fmt_rate_none_returns_em_dash(self):
        assert "8212" in _fmt_rate(None)  # &#8212; = em dash

    def test_fmt_rate_float_formats_percent(self):
        assert _fmt_rate(0.875) == "87.5%"

    def test_fmt_rate_zero(self):
        assert _fmt_rate(0.0) == "0.0%"

    def test_fmt_rate_one(self):
        assert _fmt_rate(1.0) == "100.0%"

    def test_fmt_delta_none_returns_neutral_span(self):
        result = _fmt_delta(None)
        assert "delta-neutral" in result
        assert "8212" in result  # em dash entity

    def test_fmt_delta_positive_uses_positive_class(self):
        result = _fmt_delta(0.05)
        assert "delta-positive" in result
        assert "+5.0%" in result

    def test_fmt_delta_negative_uses_negative_class(self):
        result = _fmt_delta(-0.03)
        assert "delta-negative" in result
        assert "-3.0%" in result

    def test_fmt_delta_zero_is_positive(self):
        result = _fmt_delta(0.0)
        assert "delta-positive" in result


# ---------------------------------------------------------------------------
# Basic output structure
# ---------------------------------------------------------------------------


class TestHtmlStructure:
    def test_returns_string(self):
        result = generate_html_report(_run(), _diff(), [])
        assert isinstance(result, str)

    def test_has_doctype(self):
        result = generate_html_report(_run(), _diff(), [])
        assert "<!DOCTYPE html>" in result

    def test_has_html_tags(self):
        result = generate_html_report(_run(), _diff(), [])
        assert "<html>" in result and "</html>" in result

    def test_has_head_and_body(self):
        result = generate_html_report(_run(), _diff(), [])
        assert "<head>" in result or "<head>" in result
        assert "<body>" in result and "</body>" in result

    def test_has_css_style_block(self):
        result = generate_html_report(_run(), _diff(), [])
        assert "<style>" in result

    def test_has_h1(self):
        result = generate_html_report(_run(), _diff(), [])
        assert "<h1>" in result

    def test_contains_prompt_version(self):
        result = generate_html_report(_run(prompt_version="v2.0.0"), _diff(), [])
        assert "v2.0.0" in result

    def test_contains_model(self):
        result = generate_html_report(_run(model="claude-opus-4-8"), _diff(), [])
        assert "claude-opus-4-8" in result

    def test_contains_run_id(self):
        run = _run(run_id="test-run-abc")
        result = generate_html_report(run, _diff(), [])
        assert "test-run-abc" in result


# ---------------------------------------------------------------------------
# XSS / HTML escaping
# ---------------------------------------------------------------------------


class TestXssEscaping:
    def test_question_with_script_tag_escaped(self):
        r = _regression(question="<script>alert('xss')</script>")
        result = generate_html_report(_run(), _diff(regressions=[r]), [])
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_answer_with_html_escaped(self):
        r = _regression(cur_answer='<b>bold</b> & "quoted"')
        result = generate_html_report(_run(), _diff(regressions=[r]), [])
        assert "<b>" not in result
        assert "&lt;b&gt;" in result
        assert "&amp;" in result
        assert "&quot;" in result

    def test_prompt_version_with_angle_brackets_escaped(self):
        result = generate_html_report(
            _run(prompt_version="v1<evil>"),
            _diff(),
            [],
        )
        assert "<evil>" not in result
        assert "&lt;evil&gt;" in result

    def test_model_name_with_ampersand_escaped(self):
        result = generate_html_report(
            _run(model="gpt-4o&turbo"),
            _diff(),
            [],
        )
        assert "gpt-4o&turbo" not in result
        assert "gpt-4o&amp;turbo" in result

    def test_improvement_question_escaped(self):
        i = _improvement(question="<img src=x onerror=alert(1)>")
        result = generate_html_report(_run(), _diff(improvements=[i]), [])
        assert "<img" not in result
        assert "&lt;img" in result


# ---------------------------------------------------------------------------
# Severity banner
# ---------------------------------------------------------------------------


class TestSeverityBanner:
    def test_clean_severity_banner_class(self):
        result = generate_html_report(_run(), _diff(severity="clean"), [])
        assert 'severity-clean' in result

    def test_warning_severity_banner_class(self):
        result = generate_html_report(_run(), _diff(severity="warning"), [])
        assert 'severity-warning' in result

    def test_critical_severity_banner_class(self):
        result = generate_html_report(_run(), _diff(severity="critical"), [])
        assert 'severity-critical' in result

    def test_significant_p_value_in_banner(self):
        result = generate_html_report(_run(), _diff(p_value=0.012, significant=True), [])
        assert "0.012" in result
        assert "Statistical significance" in result

    def test_non_significant_message_in_banner(self):
        result = generate_html_report(_run(), _diff(p_value=0.42, significant=False), [])
        assert "0.420" in result
        assert "not statistically significant" in result.lower()

    def test_regression_count_in_banner(self):
        regs = [_regression("TC_001"), _regression("TC_002")]
        result = generate_html_report(_run(), _diff(regressions=regs), [])
        assert "2 regression(s)" in result

    def test_improvement_count_in_banner(self):
        imps = [_improvement("TC_003")]
        result = generate_html_report(_run(), _diff(improvements=imps), [])
        assert "1 improvement(s)" in result


# ---------------------------------------------------------------------------
# Drift section
# ---------------------------------------------------------------------------


class TestDriftSection:
    def test_no_drift_section_with_short_history(self):
        history = [_run(0.9) for _ in range(5)]
        result = generate_html_report(_run(), _diff(), history)
        assert "SLOW DRIFT" not in result

    def test_drift_warning_shown_when_detected(self):
        history = [_run(0.90) for _ in range(7)] + [_run(0.84) for _ in range(7)]
        result = generate_html_report(_run(), _diff(), history)
        assert "SLOW DRIFT DETECTED" in result

    def test_no_drift_section_with_empty_history(self):
        result = generate_html_report(_run(), _diff(), [])
        assert "SLOW DRIFT" not in result


# ---------------------------------------------------------------------------
# Scorecard section
# ---------------------------------------------------------------------------


class TestScorecardSection:
    def test_scorecard_section_header_present(self):
        result = generate_html_report(_run(), _diff(), [])
        assert "Scorecard vs Baseline" in result

    def test_overall_pass_rate_in_scorecard(self):
        result = generate_html_report(_run(pass_rate=0.88), _diff(), [])
        assert "88.0%" in result

    def test_none_dimension_renders_em_dash_not_crash(self):
        run = _run(dim_rates={"numerical_accuracy": None, **{d: 0.9 for d in _DIMS if d != "numerical_accuracy"}})
        d = _diff(dim_deltas={"numerical_accuracy": None, **{d: -0.01 for d in _DIMS if d != "numerical_accuracy"}})
        result = generate_html_report(run, d, [])
        assert "&#8212;" in result  # em dash for None rate

    def test_negative_delta_uses_negative_class(self):
        d = _diff(dim_deltas={dim: -0.05 for dim in _DIMS})
        result = generate_html_report(_run(), d, [])
        assert "delta-negative" in result

    def test_positive_delta_uses_positive_class(self):
        d = _diff(dim_deltas={dim: 0.05 for dim in _DIMS}, overall_delta=0.05, severity="clean")
        result = generate_html_report(_run(), d, [])
        assert "delta-positive" in result

    def test_all_five_dimension_labels_present(self):
        result = generate_html_report(_run(), _diff(), [])
        for expected in [
            "Numerical Accuracy",
            "Refusal Correctness",
            "Faithfulness",
            "Citation Accuracy",
            "Temporal Precision",
        ]:
            assert expected in result, f"Missing label: {expected}"


# ---------------------------------------------------------------------------
# Regression cards
# ---------------------------------------------------------------------------


class TestRegressionCards:
    def test_no_regressions_shows_zero_message(self):
        result = generate_html_report(_run(), _diff(regressions=[]), [])
        assert "No regressions detected" in result
        assert "Regressions (0)" in result

    def test_regression_card_shows_test_case_id(self):
        r = _regression(tc_id="TC_042")
        result = generate_html_report(_run(), _diff(regressions=[r]), [])
        assert "TC_042" in result

    def test_regression_card_shows_question(self):
        r = _regression(question="What was operating cash flow in Q2 2024?")
        result = generate_html_report(_run(), _diff(regressions=[r]), [])
        assert "What was operating cash flow in Q2 2024?" in result

    def test_regression_card_shows_difficulty(self):
        r = _regression(difficulty="adversarial")
        result = generate_html_report(_run(), _diff(regressions=[r]), [])
        assert "adversarial" in result

    def test_regression_card_shows_score_delta(self):
        r = _regression(prev=0.90, cur=0.50)
        result = generate_html_report(_run(), _diff(regressions=[r]), [])
        assert "-0.40" in result

    def test_regression_card_shows_both_answers(self):
        r = _regression(
            prev_answer="Net income was $12.3B.",
            cur_answer="I cannot find that information.",
        )
        result = generate_html_report(_run(), _diff(regressions=[r]), [])
        assert "Net income was $12.3B." in result
        assert "I cannot find that information." in result
        assert "Baseline Answer" in result
        assert "Current Answer" in result

    def test_regression_count_in_heading(self):
        regs = [_regression("TC_001"), _regression("TC_002"), _regression("TC_003")]
        result = generate_html_report(_run(), _diff(regressions=regs), [])
        assert "Regressions (3)" in result

    def test_regression_dimensions_shown(self):
        r = _regression(dims=["faithfulness", "citation_accuracy"])
        result = generate_html_report(_run(), _diff(regressions=[r]), [])
        assert "faithfulness" in result
        assert "citation_accuracy" in result

    def test_empty_dimensions_list_shows_none_identified(self):
        r = _regression(dims=[])
        result = generate_html_report(_run(), _diff(regressions=[r]), [])
        assert "none identified" in result


# ---------------------------------------------------------------------------
# Improvement cards
# ---------------------------------------------------------------------------


class TestImprovementCards:
    def test_no_improvements_shows_zero_message(self):
        result = generate_html_report(_run(), _diff(improvements=[]), [])
        assert "No improvements detected" in result
        assert "Improvements (0)" in result

    def test_improvement_card_shows_test_case_id(self):
        i = _improvement(tc_id="TC_010")
        result = generate_html_report(_run(), _diff(improvements=[i]), [])
        assert "TC_010" in result

    def test_improvement_card_shows_question(self):
        i = _improvement(question="What was the gross margin?")
        result = generate_html_report(_run(), _diff(improvements=[i]), [])
        assert "What was the gross margin?" in result

    def test_improvement_count_in_heading(self):
        imps = [_improvement("TC_010"), _improvement("TC_011")]
        result = generate_html_report(_run(), _diff(improvements=imps), [])
        assert "Improvements (2)" in result

    def test_improvement_delta_shown(self):
        i = _improvement(prev=0.60, cur=0.95)
        result = generate_html_report(_run(), _diff(improvements=[i]), [])
        assert "+0.35" in result


# ---------------------------------------------------------------------------
# Historical trend table
# ---------------------------------------------------------------------------


class TestHistoryTable:
    def test_history_section_header_present(self):
        result = generate_html_report(_run(), _diff(), [_run(0.9)])
        assert "Historical Trend" in result

    def test_history_shows_prompt_version(self):
        run = _run(pass_rate=0.80, prompt_version="v2.1.0")
        result = generate_html_report(_run(), _diff(), [run])
        assert "v2.1.0" in result

    def test_history_shows_pass_rate(self):
        run = _run(pass_rate=0.75)
        result = generate_html_report(_run(), _diff(), [run])
        assert "75.0%" in result

    def test_history_shows_cost(self):
        result = generate_html_report(_run(), _diff(), [_run()])
        assert "$0.0500" in result

    def test_history_capped_at_ten_runs(self):
        history = [_run(pass_rate=0.80) for _ in range(15)]
        result = generate_html_report(_run(), _diff(), history)
        tr_count = result.count("<tr>")
        # Metadata table has 7 rows; history table has 1 header + up to 10 data rows.
        # Capped: 7 + 11 = 18. Uncapped (15 rows): 7 + 16 = 23. Assert we're capped.
        assert tr_count <= 18

    def test_none_dimension_in_history_renders_em_dash(self):
        dim_rates: dict[str, float | None] = {
            "numerical_accuracy": None,
            "refusal_correctness": 0.9,
            "faithfulness": 0.9,
            "citation_accuracy": 0.9,
            "temporal_precision": None,
        }
        run = _run(dim_rates=dim_rates)
        result = generate_html_report(_run(), _diff(), [run])
        assert "&#8212;" in result

    def test_history_table_columns_present(self):
        result = generate_html_report(_run(), _diff(), [_run()])
        for col in ["Run", "Prompt Version", "Overall", "Faithfulness", "Numerical", "Refusal", "Temporal", "Cost"]:
            assert col in result

    def test_empty_history_renders_table_with_no_rows(self):
        result = generate_html_report(_run(), _diff(), [])
        assert "Historical Trend" in result
        # Table header columns still present
        assert "Overall" in result


# ---------------------------------------------------------------------------
# Metadata table
# ---------------------------------------------------------------------------


class TestMetadataTable:
    def test_metadata_table_shows_all_keys(self):
        result = generate_html_report(_run(), _diff(), [])
        for label in ["Prompt Version", "Model", "Dataset Version", "Timestamp", "Run ID", "Total Cost", "Avg Latency"]:
            assert label in result

    def test_cost_formatted_to_four_decimals(self):
        run = _run()
        run = EvalRunResult(
            **{**run.model_dump(), "total_cost_usd": 0.0312},
        )
        result = generate_html_report(run, _diff(), [])
        assert "$0.0312" in result

    def test_latency_formatted_without_decimal(self):
        result = generate_html_report(_run(), _diff(), [])
        assert "420ms" in result
