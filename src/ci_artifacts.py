"""CI artifact helpers: JSON summary and severity text file for pipeline integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.diff_runner import EvalDiff
from src.eval_runner import EvalRunResult

_DIMS = [
    "numerical_accuracy",
    "refusal_correctness",
    "faithfulness",
    "citation_accuracy",
    "temporal_precision",
]


def build_summary(
    current_run: EvalRunResult,
    diff: EvalDiff,
    drift_warning: str | None,
    report_url: str,
) -> dict[str, Any]:
    """Return a JSON-serialisable dict suitable for CI step outputs or artifact upload."""
    dimension_summary = {
        dim: {
            "pass_rate": current_run.pass_rate_by_dimension.get(dim),
            "delta": diff.dimension_deltas.get(dim),
        }
        for dim in _DIMS
    }
    return {
        "severity": diff.severity,
        "baseline_pass_rate": diff.baseline_pass_rate,
        "current_pass_rate": diff.current_pass_rate,
        "delta": diff.overall_delta,
        "regression_count": len(diff.regressions),
        "improvement_count": len(diff.improvements),
        "dimension_summary": dimension_summary,
        "drift_warning": drift_warning,
        "report_url": report_url,
    }


def write_severity(severity: str, path: Path) -> None:
    """Write the severity string to *path* so a CI step can read it with $(<file)."""
    path.write_text(severity)
