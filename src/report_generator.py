"""Phase 4 Step 1: HTML diff report generator."""

from __future__ import annotations

import html

from src.diff_runner import EvalDiff
from src.drift_detector import detect_slow_drift
from src.eval_runner import EvalRunResult

_DIMS = [
    "numerical_accuracy",
    "refusal_correctness",
    "faithfulness",
    "citation_accuracy",
    "temporal_precision",
]


def _e(value: object) -> str:
    """Escape any user-controlled value before interpolating into HTML."""
    return html.escape(str(value))


def _fmt_rate(rate: float | None) -> str:
    return "&#8212;" if rate is None else f"{rate:.1%}"


def _fmt_delta(delta: float | None) -> str:
    if delta is None:
        return '<span class="delta-neutral">&#8212;</span>'
    css = "delta-positive" if delta >= 0 else "delta-negative"
    sign = "+" if delta >= 0 else ""
    return f'<span class="{css}">{sign}{delta:.1%}</span>'


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _metadata_table(run: EvalRunResult) -> str:
    rows = [
        ("Prompt Version", _e(run.prompt_version)),
        ("Model", _e(run.model)),
        ("Dataset Version", _e(run.dataset_version)),
        ("Timestamp", _e(run.timestamp.isoformat())),
        ("Run ID", _e(run.run_id)),
        ("Total Cost", f"${run.total_cost_usd:.4f}"),
        ("Avg Latency", f"{run.avg_latency_ms:.0f}ms"),
    ]
    trs = "".join(
        f"<tr><td><strong>{label}</strong></td><td>{value}</td></tr>"
        for label, value in rows
    )
    return f'<table style="margin-bottom:24px;">{trs}</table>'


def _severity_banner(diff: EvalDiff) -> str:
    sig_text = (
        f"Statistical significance: p={diff.p_value:.3f}"
        if diff.is_statistically_significant
        else f"Delta not statistically significant (p={diff.p_value:.3f})"
    )
    return (
        f'<div class="severity-{_e(diff.severity)}">'
        f"<strong>Status: {_e(diff.severity.upper())}</strong> &mdash; "
        f"{len(diff.regressions)} regression(s), "
        f"{len(diff.improvements)} improvement(s) detected. "
        f"{_e(sig_text)}"
        f"</div>"
    )


def _drift_section(drift_warning: str | None) -> str:
    if not drift_warning:
        return ""
    return (
        '<div style="background:#fff3cd;border:1px solid #ffeeba;'
        'border-radius:8px;padding:16px;margin-top:12px;">'
        f"&#9888;&#65039; {_e(drift_warning)}"
        "</div>"
    )


def _scorecard_card(label: str, rate: float | None, delta: float | None) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-value">{_fmt_rate(rate)}</div>'
        f'<div class="metric-label">{_e(label)}</div>'
        f"{_fmt_delta(delta)}"
        "</div>"
    )


def _scorecard_section(run: EvalRunResult, diff: EvalDiff) -> str:
    cards = [_scorecard_card("Overall Pass Rate", run.overall_pass_rate, diff.overall_delta)]
    for dim in _DIMS:
        cards.append(_scorecard_card(
            dim.replace("_", " ").title(),
            run.pass_rate_by_dimension.get(dim),
            diff.dimension_deltas.get(dim),
        ))
    return f'<h2>Scorecard vs Baseline</h2><div class="scorecard">{"".join(cards)}</div>'


def _regression_cards(diff: EvalDiff) -> str:
    if not diff.regressions:
        return "<h2>Regressions (0)</h2><p>No regressions detected.</p>"

    cards = []
    for r in diff.regressions:
        dims = _e(", ".join(r.dimensions_that_regressed) or "none identified")
        cards.append(
            '<div class="regression-card">'
            f"<strong>{_e(r.test_case_id)}</strong> | "
            f"Difficulty: {_e(r.difficulty)} | "
            f"Category: {_e(r.category)}<br>"
            f"<em>{_e(r.question)}</em><br>"
            f"<small>Score: {r.previous_composite:.2f} &rarr; {r.current_composite:.2f} "
            f"({r.score_delta:+.2f}) | Regressed on: {dims}</small>"
            '<div class="answer-comparison">'
            f'<div class="answer-box"><strong>Baseline Answer</strong><br>{_e(r.previous_answer)}</div>'
            f'<div class="answer-box"><strong>Current Answer</strong><br>{_e(r.current_answer)}</div>'
            "</div>"
            "</div>"
        )
    return f"<h2>Regressions ({len(diff.regressions)})</h2>{''.join(cards)}"


def _improvement_cards(diff: EvalDiff) -> str:
    if not diff.improvements:
        return "<h2>Improvements (0)</h2><p>No improvements detected.</p>"

    cards = []
    for i in diff.improvements:
        delta = i.current_composite - i.previous_composite
        cards.append(
            '<div style="background:#f0fff4;border:1px solid #c3e6cb;'
            'border-radius:8px;padding:12px;margin:8px 0;">'
            f"<strong>{_e(i.test_case_id)}</strong> | "
            f"{_e(i.difficulty)} | {_e(i.category)}<br>"
            f"<em>{_e(i.question)}</em><br>"
            f"<small>{i.previous_composite:.2f} &rarr; {i.current_composite:.2f} "
            f"({delta:+.2f})</small>"
            "</div>"
        )
    return f"<h2>Improvements ({len(diff.improvements)})</h2>{''.join(cards)}"


def _history_table(run_history: list[EvalRunResult]) -> str:
    rows = []
    for r in run_history[-10:]:
        rows.append(
            "<tr>"
            f"<td>{_e(r.timestamp.strftime('%m/%d %H:%M'))}</td>"
            f"<td>{_e(r.prompt_version)}</td>"
            f"<td>{_fmt_rate(r.overall_pass_rate)}</td>"
            f"<td>{_fmt_rate(r.pass_rate_by_dimension.get('faithfulness'))}</td>"
            f"<td>{_fmt_rate(r.pass_rate_by_dimension.get('numerical_accuracy'))}</td>"
            f"<td>{_fmt_rate(r.pass_rate_by_dimension.get('refusal_correctness'))}</td>"
            f"<td>{_fmt_rate(r.pass_rate_by_dimension.get('temporal_precision'))}</td>"
            f"<td>${r.total_cost_usd:.4f}</td>"
            "</tr>"
        )
    header = (
        "<tr>"
        "<th>Run</th><th>Prompt Version</th><th>Overall</th>"
        "<th>Faithfulness</th><th>Numerical</th><th>Refusal</th>"
        "<th>Temporal</th><th>Cost</th>"
        "</tr>"
    )
    return (
        "<h2>Historical Trend</h2>"
        f"<table>{header}{''.join(rows)}</table>"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_CSS = """
body { font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
.scorecard { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }
.metric-card { background: #f8f9fa; border-radius: 8px; padding: 16px; text-align: center; }
.metric-value { font-size: 32px; font-weight: bold; }
.metric-label { font-size: 12px; color: #666; margin-top: 4px; }
.delta-positive { color: #28a745; }
.delta-negative { color: #dc3545; }
.delta-neutral { color: #6c757d; }
.severity-critical { background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; padding: 16px; }
.severity-warning  { background: #fff3cd; border: 1px solid #ffeeba; border-radius: 8px; padding: 16px; }
.severity-clean    { background: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; padding: 16px; }
.regression-card { border: 1px solid #f5c6cb; border-radius: 8px; padding: 16px; margin: 12px 0; background: #fff5f5; }
.answer-comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }
.answer-box { background: #f8f9fa; border-radius: 4px; padding: 12px; font-size: 13px; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #dee2e6; }
th { background: #f8f9fa; font-weight: 600; }
"""


def generate_html_report(
    current_run: EvalRunResult,
    diff: EvalDiff,
    run_history: list[EvalRunResult],
) -> str:
    drift_warning = detect_slow_drift(run_history)
    title = (
        f"Eval Report &mdash; {_e(current_run.prompt_version)} &mdash; "
        f"{_e(current_run.timestamp.strftime('%Y-%m-%d %H:%M'))}"
    )
    return "\n".join([
        "<!DOCTYPE html>",
        "<html>",
        f"<head><title>{title}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        _metadata_table(current_run),
        _severity_banner(diff),
        _drift_section(drift_warning),
        _scorecard_section(current_run, diff),
        _regression_cards(diff),
        _improvement_cards(diff),
        _history_table(run_history),
        "</body>",
        "</html>",
    ])
