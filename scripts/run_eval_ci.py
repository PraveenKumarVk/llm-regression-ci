"""CI entry point: run eval, diff against baseline, write report artifacts."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import click

from src.ci_artifacts import build_summary, write_severity
from src.diff_runner import EvalDiff, diff_eval_runs
from src.drift_detector import detect_slow_drift
from src.eval_runner import EvalRunResult, run_eval
from src.prompt_loader import load_latest
from src.report_generator import generate_html_report


def _load_history(history_dir: Path) -> list[EvalRunResult]:
    runs: list[EvalRunResult] = []
    for p in sorted(history_dir.glob("*.json")):
        try:
            runs.append(EvalRunResult.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception as exc:
            click.echo(f"Warning: skipping corrupt history file {p.name}: {exc}", err=True)
    return sorted(runs, key=lambda r: r.timestamp)


def _save_run(history_dir: Path, run: EvalRunResult) -> None:
    path = history_dir / f"{run.run_id}.json"
    path.write_text(run.model_dump_json(), encoding="utf-8")


def _no_baseline_diff(run: EvalRunResult) -> EvalDiff:
    """Produce a neutral diff when no prior run exists to compare against."""
    return EvalDiff(
        baseline_run_id="none",
        current_run_id=run.run_id,
        baseline_pass_rate=run.overall_pass_rate,
        current_pass_rate=run.overall_pass_rate,
        overall_delta=0.0,
        regressions=[],
        improvements=[],
        dimension_deltas={dim: 0.0 for dim in run.pass_rate_by_dimension},
        difficulty_deltas={d: 0.0 for d in run.pass_rate_by_difficulty},
        category_deltas={c: 0.0 for c in run.pass_rate_by_category},
        is_statistically_significant=False,
        p_value=1.0,
        severity="clean",
    )


@click.command()
@click.option(
    "--prompt-version",
    default=None,
    help="Prompt version tag (e.g. v1.0.0). Defaults to the latest version in prompts/.",
)
@click.option("--model", default="gpt-4o", show_default=True)
@click.option(
    "--dataset",
    required=True,
    type=click.Path(exists=True),
    help="Path to the golden dataset JSON.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory to write report.html, summary.json, and severity.txt.",
)
@click.option(
    "--history-dir",
    required=True,
    type=click.Path(),
    help="Directory for persisted eval run history (one JSON per run).",
)
@click.option("--concurrency", default=5, show_default=True, help="Max concurrent scorer calls.")
def main(
    prompt_version: str | None,
    model: str,
    dataset: str,
    output_dir: str,
    history_dir: str,
    concurrency: int,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = Path(history_dir)
    history.mkdir(parents=True, exist_ok=True)

    if prompt_version is None:
        prompt_version = load_latest().version
        click.echo(f"Resolved prompt version: {prompt_version}")

    click.echo(f"Running eval — prompt={prompt_version}  model={model}  dataset={dataset}")
    current_run = asyncio.run(run_eval(dataset, prompt_version, model, concurrency))
    click.echo(f"Eval done. overall_pass_rate={current_run.overall_pass_rate:.1%}  cost=${current_run.total_cost_usd:.4f}")

    run_history = _load_history(history)
    baseline = run_history[-1] if run_history else None
    diff = diff_eval_runs(baseline, current_run) if baseline else _no_baseline_diff(current_run)

    full_history = run_history + [current_run]
    drift_warning = detect_slow_drift(full_history)
    if drift_warning:
        click.echo(f"Drift: {drift_warning}", err=True)

    report_url = os.environ.get("EVAL_REPORT_URL", "")

    html = generate_html_report(current_run, diff, full_history)
    (output / "report.html").write_text(html, encoding="utf-8")

    summary = build_summary(current_run, diff, drift_warning, report_url)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))

    write_severity(diff.severity, output / "severity.txt")

    _save_run(history, current_run)

    click.echo(f"Artifacts written to {output}/  severity={diff.severity}")


if __name__ == "__main__":
    main()
