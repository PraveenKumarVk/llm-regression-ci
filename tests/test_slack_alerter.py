"""Tests for Phase 4 Step 2: send_slack_alert and _build_blocks."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
import respx

from src.diff_runner import EvalDiff, ImprovementEvent, RegressionEvent
from src.eval_runner import EvalRunResult
from src.slack_alerter import _build_blocks, _dim_line, send_slack_alert

_DIMS = [
    "numerical_accuracy",
    "refusal_correctness",
    "faithfulness",
    "citation_accuracy",
    "temporal_precision",
]

_TS = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)
_WEBHOOK = "https://hooks.slack.com/services/T000/B000/xxxx"
_REPORT_URL = "https://reports.example.com/run-001.html"

_counter = 0


def _run(
    pass_rate: float = 0.90,
    *,
    run_id: str | None = None,
    prompt_version: str = "v1.0.0",
    model: str = "gpt-4o",
    dim_rates: dict[str, float | None] | None = None,
) -> EvalRunResult:
    global _counter
    _counter += 1
    rates = dim_rates if dim_rates is not None else {d: pass_rate for d in _DIMS}
    return EvalRunResult(
        run_id=run_id or f"run-{_counter:04d}",
        prompt_version=prompt_version,
        model=model,
        dataset_version="v1.0.0",
        timestamp=_TS,
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
    difficulty: str = "easy",
    dims: list[str] | None = None,
    prev: float = 0.90,
    cur: float = 0.60,
) -> RegressionEvent:
    return RegressionEvent(
        test_case_id=tc_id,
        question="What was net income?",
        difficulty=difficulty,
        category="numerical_extraction",
        previous_composite=prev,
        current_composite=cur,
        score_delta=cur - prev,
        dimensions_that_regressed=dims if dims is not None else ["numerical_accuracy"],
        previous_answer="$12B",
        current_answer="NOT_IN_DOCUMENT",
    )


def _improvement(tc_id: str = "TC_002") -> ImprovementEvent:
    return ImprovementEvent(
        test_case_id=tc_id,
        question="What was revenue?",
        difficulty="medium",
        category="temporal_precision",
        previous_composite=0.60,
        current_composite=0.92,
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
# _dim_line helper
# ---------------------------------------------------------------------------


class TestDimLine:
    def test_both_values_present(self):
        line = _dim_line("numerical_accuracy", 0.80, -0.05)
        assert "Numerical Accuracy" in line
        assert "80.0%" in line
        assert "↓5.0%" in line

    def test_positive_delta_uses_up_arrow(self):
        line = _dim_line("faithfulness", 0.90, 0.03)
        assert "↑3.0%" in line

    def test_none_rate_shows_na(self):
        line = _dim_line("temporal_precision", None, -0.02)
        assert "N/A" in line

    def test_none_delta_shows_na(self):
        line = _dim_line("citation_accuracy", 0.75, None)
        assert "(N/A)" in line

    def test_both_none_does_not_crash(self):
        line = _dim_line("refusal_correctness", None, None)
        assert "N/A" in line
        assert "(N/A)" in line

    def test_zero_delta_uses_up_arrow(self):
        line = _dim_line("faithfulness", 0.85, 0.0)
        assert "↑" in line

    def test_label_title_cased(self):
        line = _dim_line("refusal_correctness", 0.80, 0.0)
        assert "Refusal Correctness" in line


# ---------------------------------------------------------------------------
# _build_blocks: block structure
# ---------------------------------------------------------------------------


class TestBuildBlocksStructure:
    def test_returns_list(self):
        blocks = _build_blocks(_diff(), _run(), _REPORT_URL, None)
        assert isinstance(blocks, list)

    def test_always_has_header_block(self):
        blocks = _build_blocks(_diff(), _run(), _REPORT_URL, None)
        assert blocks[0]["type"] == "header"

    def test_always_has_summary_section(self):
        blocks = _build_blocks(_diff(), _run(), _REPORT_URL, None)
        section = blocks[1]
        assert section["type"] == "section"
        assert "fields" in section

    def test_always_has_dimension_section(self):
        blocks = _build_blocks(_diff(), _run(), _REPORT_URL, None)
        texts = [b.get("text", {}).get("text", "") for b in blocks if b["type"] == "section" and "text" in b]
        assert any("By Dimension" in t for t in texts)

    def test_always_has_actions_block_with_button(self):
        blocks = _build_blocks(_diff(), _run(), _REPORT_URL, None)
        action_blocks = [b for b in blocks if b["type"] == "actions"]
        assert len(action_blocks) == 1
        assert action_blocks[0]["elements"][0]["type"] == "button"

    def test_button_url_is_report_url(self):
        blocks = _build_blocks(_diff(), _run(), _REPORT_URL, None)
        btn = next(b for b in blocks if b["type"] == "actions")["elements"][0]
        assert btn["url"] == _REPORT_URL


# ---------------------------------------------------------------------------
# _build_blocks: severity header
# ---------------------------------------------------------------------------


class TestBuildBlocksSeverity:
    def test_clean_severity_in_header(self):
        blocks = _build_blocks(_diff(severity="clean"), _run(), _REPORT_URL, None)
        assert "CLEAN" in blocks[0]["text"]["text"]

    def test_warning_severity_in_header(self):
        blocks = _build_blocks(_diff(severity="warning"), _run(), _REPORT_URL, None)
        assert "WARNING" in blocks[0]["text"]["text"]

    def test_critical_severity_in_header(self):
        blocks = _build_blocks(_diff(severity="critical"), _run(), _REPORT_URL, None)
        assert "CRITICAL" in blocks[0]["text"]["text"]

    def test_button_style_primary_when_clean(self):
        blocks = _build_blocks(_diff(severity="clean"), _run(), _REPORT_URL, None)
        btn = next(b for b in blocks if b["type"] == "actions")["elements"][0]
        assert btn["style"] == "primary"

    def test_button_style_danger_when_warning(self):
        blocks = _build_blocks(_diff(severity="warning"), _run(), _REPORT_URL, None)
        btn = next(b for b in blocks if b["type"] == "actions")["elements"][0]
        assert btn["style"] == "danger"

    def test_button_style_danger_when_critical(self):
        blocks = _build_blocks(_diff(severity="critical"), _run(), _REPORT_URL, None)
        btn = next(b for b in blocks if b["type"] == "actions")["elements"][0]
        assert btn["style"] == "danger"


# ---------------------------------------------------------------------------
# _build_blocks: summary fields
# ---------------------------------------------------------------------------


class TestBuildBlocksSummaryFields:
    def _fields(self, blocks: list[dict]) -> list[str]:
        return [f["text"] for f in blocks[1]["fields"]]

    def test_prompt_version_in_fields(self):
        blocks = _build_blocks(_diff(), _run(prompt_version="v2.1.0"), _REPORT_URL, None)
        assert any("v2.1.0" in f for f in self._fields(blocks))

    def test_model_in_fields(self):
        blocks = _build_blocks(_diff(), _run(model="claude-opus-4-8"), _REPORT_URL, None)
        assert any("claude-opus-4-8" in f for f in self._fields(blocks))

    def test_pass_rate_in_fields(self):
        blocks = _build_blocks(_diff(), _run(pass_rate=0.88), _REPORT_URL, None)
        assert any("88.0%" in f for f in self._fields(blocks))

    def test_overall_delta_down_arrow(self):
        blocks = _build_blocks(_diff(overall_delta=-0.05), _run(), _REPORT_URL, None)
        fields = self._fields(blocks)
        assert any("↓5.0%" in f for f in fields)

    def test_overall_delta_up_arrow(self):
        blocks = _build_blocks(
            _diff(overall_delta=0.03, severity="clean"), _run(), _REPORT_URL, None
        )
        fields = self._fields(blocks)
        assert any("↑3.0%" in f for f in fields)

    def test_regression_improvement_counts_in_fields(self):
        blocks = _build_blocks(
            _diff(regressions=[_regression()], improvements=[_improvement()]),
            _run(),
            _REPORT_URL,
            None,
        )
        assert any("1 / 1" in f for f in self._fields(blocks))


# ---------------------------------------------------------------------------
# _build_blocks: regression block
# ---------------------------------------------------------------------------


class TestBuildBlocksRegressions:
    def test_no_regression_block_when_empty(self):
        blocks = _build_blocks(_diff(regressions=[]), _run(), _REPORT_URL, None)
        texts = [b.get("text", {}).get("text", "") for b in blocks]
        assert not any("Regressions" in t and "*Regressions*" in t for t in texts)

    def test_regression_block_present_when_non_empty(self):
        blocks = _build_blocks(_diff(regressions=[_regression()]), _run(), _REPORT_URL, None)
        texts = [b.get("text", {}).get("text", "") for b in blocks]
        assert any("*Regressions*" in t for t in texts)

    def test_regression_shows_test_case_id(self):
        blocks = _build_blocks(
            _diff(regressions=[_regression("TC_042")]), _run(), _REPORT_URL, None
        )
        texts = "\n".join(b.get("text", {}).get("text", "") for b in blocks)
        assert "TC_042" in texts

    def test_regression_shows_difficulty(self):
        blocks = _build_blocks(
            _diff(regressions=[_regression(difficulty="adversarial")]),
            _run(),
            _REPORT_URL,
            None,
        )
        texts = "\n".join(b.get("text", {}).get("text", "") for b in blocks)
        assert "adversarial" in texts

    def test_regression_shows_score_delta(self):
        blocks = _build_blocks(
            _diff(regressions=[_regression(prev=0.90, cur=0.50)]),
            _run(),
            _REPORT_URL,
            None,
        )
        texts = "\n".join(b.get("text", {}).get("text", "") for b in blocks)
        assert "-0.40" in texts

    def test_regression_capped_at_five(self):
        regs = [_regression(tc_id=f"TC_{i:03d}") for i in range(8)]
        blocks = _build_blocks(_diff(regressions=regs), _run(), _REPORT_URL, None)
        reg_block = next(
            b for b in blocks
            if b["type"] == "section" and "*Regressions*" in b.get("text", {}).get("text", "")
        )
        text = reg_block["text"]["text"]
        # Only 5 bullets (lines starting with •), plus the overflow note
        assert text.count("•") == 5
        assert "3 more" in text

    def test_regression_no_overflow_note_for_exactly_five(self):
        regs = [_regression(tc_id=f"TC_{i:03d}") for i in range(5)]
        blocks = _build_blocks(_diff(regressions=regs), _run(), _REPORT_URL, None)
        reg_block = next(
            b for b in blocks
            if b["type"] == "section" and "*Regressions*" in b.get("text", {}).get("text", "")
        )
        assert "more" not in reg_block["text"]["text"]

    def test_empty_dimensions_list_shows_none_identified(self):
        blocks = _build_blocks(
            _diff(regressions=[_regression(dims=[])]), _run(), _REPORT_URL, None
        )
        texts = "\n".join(b.get("text", {}).get("text", "") for b in blocks)
        assert "none identified" in texts


# ---------------------------------------------------------------------------
# _build_blocks: drift warning block
# ---------------------------------------------------------------------------


class TestBuildBlocksDrift:
    def test_no_drift_block_when_none(self):
        blocks = _build_blocks(_diff(), _run(), _REPORT_URL, None)
        texts = [b.get("text", {}).get("text", "") for b in blocks]
        assert not any("Slow Drift" in t for t in texts)

    def test_drift_block_present_when_provided(self):
        blocks = _build_blocks(_diff(), _run(), _REPORT_URL, "SLOW DRIFT DETECTED: ...")
        texts = [b.get("text", {}).get("text", "") for b in blocks]
        assert any("Slow Drift Detected" in t for t in texts)

    def test_drift_message_content_in_block(self):
        msg = "SLOW DRIFT DETECTED: 7-run average dropped 10.0% (90.0% → 80.0%)."
        blocks = _build_blocks(_diff(), _run(), _REPORT_URL, msg)
        texts = "\n".join(b.get("text", {}).get("text", "") for b in blocks)
        assert "10.0%" in texts


# ---------------------------------------------------------------------------
# _build_blocks: dimension breakdown
# ---------------------------------------------------------------------------


class TestBuildBlocksDimensions:
    def test_all_five_dimensions_present(self):
        blocks = _build_blocks(_diff(), _run(), _REPORT_URL, None)
        dim_text = next(
            b["text"]["text"] for b in blocks
            if b["type"] == "section" and "By Dimension" in b.get("text", {}).get("text", "")
        )
        for label in [
            "Numerical Accuracy", "Refusal Correctness", "Faithfulness",
            "Citation Accuracy", "Temporal Precision",
        ]:
            assert label in dim_text, f"Missing: {label}"

    def test_none_dimension_rate_shows_na_not_crash(self):
        run = _run(
            dim_rates={
                "numerical_accuracy": None,
                **{d: 0.9 for d in _DIMS if d != "numerical_accuracy"},
            }
        )
        d = _diff(
            dim_deltas={
                "numerical_accuracy": None,
                **{d: -0.01 for d in _DIMS if d != "numerical_accuracy"},
            }
        )
        blocks = _build_blocks(d, run, _REPORT_URL, None)
        dim_text = next(
            b["text"]["text"] for b in blocks
            if "By Dimension" in b.get("text", {}).get("text", "")
        )
        assert "N/A" in dim_text

    def test_none_dimension_delta_shows_na_not_crash(self):
        d = _diff(dim_deltas={"numerical_accuracy": None, **{x: -0.01 for x in _DIMS if x != "numerical_accuracy"}})
        blocks = _build_blocks(d, _run(), _REPORT_URL, None)
        dim_text = next(
            b["text"]["text"] for b in blocks
            if "By Dimension" in b.get("text", {}).get("text", "")
        )
        assert "(N/A)" in dim_text


# ---------------------------------------------------------------------------
# send_slack_alert: HTTP behaviour
# ---------------------------------------------------------------------------


class TestSendSlackAlert:
    @respx.mock
    async def test_posts_to_webhook_url(self):
        route = respx.post(_WEBHOOK).mock(return_value=httpx.Response(200, text="ok"))
        await send_slack_alert(_diff(), _run(), _REPORT_URL, _WEBHOOK)
        assert route.called

    @respx.mock
    async def test_payload_contains_blocks_key(self):
        respx.post(_WEBHOOK).mock(return_value=httpx.Response(200, text="ok"))
        await send_slack_alert(_diff(), _run(), _REPORT_URL, _WEBHOOK)
        request = respx.calls.last.request
        import json
        body = json.loads(request.content)
        assert "blocks" in body
        assert isinstance(body["blocks"], list)

    @respx.mock
    async def test_http_error_raises(self):
        respx.post(_WEBHOOK).mock(return_value=httpx.Response(400, text="invalid_payload"))
        with pytest.raises(httpx.HTTPStatusError):
            await send_slack_alert(_diff(), _run(), _REPORT_URL, _WEBHOOK)

    @respx.mock
    async def test_server_error_raises(self):
        respx.post(_WEBHOOK).mock(return_value=httpx.Response(500, text="error"))
        with pytest.raises(httpx.HTTPStatusError):
            await send_slack_alert(_diff(), _run(), _REPORT_URL, _WEBHOOK)

    @respx.mock
    async def test_drift_warning_forwarded_to_blocks(self):
        respx.post(_WEBHOOK).mock(return_value=httpx.Response(200, text="ok"))
        await send_slack_alert(
            _diff(), _run(), _REPORT_URL, _WEBHOOK,
            drift_warning="SLOW DRIFT DETECTED: something dropped.",
        )
        request = respx.calls.last.request
        import json
        body = json.loads(request.content)
        all_text = str(body["blocks"])
        assert "Slow Drift Detected" in all_text

    @respx.mock
    async def test_default_drift_warning_is_none(self):
        respx.post(_WEBHOOK).mock(return_value=httpx.Response(200, text="ok"))
        # Should not raise — drift_warning=None is the default
        await send_slack_alert(_diff(), _run(), _REPORT_URL, _WEBHOOK)
        request = respx.calls.last.request
        import json
        body = json.loads(request.content)
        all_text = str(body["blocks"])
        assert "Slow Drift" not in all_text
