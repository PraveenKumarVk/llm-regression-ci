"""Phase 4 Step 2: Slack alert sender."""

from __future__ import annotations

import httpx

from src.diff_runner import EvalDiff
from src.eval_runner import EvalRunResult

_DIMS = [
    "numerical_accuracy",
    "refusal_correctness",
    "faithfulness",
    "citation_accuracy",
    "temporal_precision",
]

_SEVERITY_EMOJI: dict[str, str] = {
    "critical": "\U0001f534",
    "warning": "\U0001f7e1",
    "clean": "\U0001f7e2",
}


def _dim_line(dim: str, rate: float | None, delta: float | None) -> str:
    label = dim.replace("_", " ").title()
    rate_str = f"{rate:.1%}" if rate is not None else "N/A"
    if delta is None:
        delta_str = "(N/A)"
    else:
        arrow = "↑" if delta >= 0 else "↓"
        delta_str = f"({arrow}{abs(delta):.1%})"
    return f"• {label}: {rate_str} {delta_str}"


def _build_blocks(
    diff: EvalDiff,
    current_run: EvalRunResult,
    report_url: str,
    drift_warning: str | None,
) -> list[dict]:
    arrow = "↑" if diff.overall_delta >= 0 else "↓"
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{_SEVERITY_EMOJI[diff.severity]} Eval Run — {diff.severity.upper()}",
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Prompt Version*\n{current_run.prompt_version}"},
                {"type": "mrkdwn", "text": f"*Model*\n{current_run.model}"},
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*Overall Pass Rate*\n{current_run.overall_pass_rate:.1%} "
                        f"({arrow}{abs(diff.overall_delta):.1%} vs baseline)"
                    ),
                },
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*Regressions / Improvements*\n"
                        f"{len(diff.regressions)} / {len(diff.improvements)}"
                    ),
                },
            ],
        },
    ]

    if diff.regressions:
        lines = [
            f"• `{r.test_case_id}` [{r.difficulty}] — "
            f"{r.score_delta:+.2f} on {', '.join(r.dimensions_that_regressed) or 'none identified'}"
            for r in diff.regressions[:5]
        ]
        if len(diff.regressions) > 5:
            lines.append(f"_...and {len(diff.regressions) - 5} more_")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Regressions*\n" + "\n".join(lines)},
        })

    if drift_warning:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"⚠️ *Slow Drift Detected*\n{drift_warning}"},
        })

    dim_lines = [
        _dim_line(
            dim,
            current_run.pass_rate_by_dimension.get(dim),
            diff.dimension_deltas.get(dim),
        )
        for dim in _DIMS
    ]
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*By Dimension*\n" + "\n".join(dim_lines)},
    })

    blocks.append({
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "View Full Report"},
                "url": report_url,
                "style": "primary" if diff.severity == "clean" else "danger",
            }
        ],
    })

    return blocks


async def send_slack_alert(
    diff: EvalDiff,
    current_run: EvalRunResult,
    report_url: str,
    webhook_url: str,
    drift_warning: str | None = None,
) -> None:
    blocks = _build_blocks(diff, current_run, report_url, drift_warning)
    async with httpx.AsyncClient() as client:
        response = await client.post(webhook_url, json={"blocks": blocks})
        response.raise_for_status()
