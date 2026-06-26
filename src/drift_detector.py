"""Phase 3 Step 4: Slow drift detection across a history of eval runs."""

from __future__ import annotations

from src.eval_runner import EvalRunResult


def detect_slow_drift(
    run_history: list[EvalRunResult],
    window_size: int = 7,
) -> str | None:
    """
    Compares the moving average of the most recent window against the window
    before it. Returns a warning string when the average dropped more than 5
    percentage points, or None when no drift is detected.

    Requires at least 2 * window_size runs to make the comparison; returns
    None when the history is too short.
    """
    if len(run_history) < window_size * 2:
        return None

    recent = run_history[-window_size:]
    previous = run_history[-(window_size * 2):-window_size]

    recent_avg = sum(r.overall_pass_rate for r in recent) / window_size
    previous_avg = sum(r.overall_pass_rate for r in previous) / window_size

    drift = recent_avg - previous_avg

    if drift < -0.05:
        return (
            f"SLOW DRIFT DETECTED: {window_size}-run moving average "
            f"dropped {abs(drift):.1%} "
            f"({previous_avg:.1%} → {recent_avg:.1%}). "
            f"No single run triggered a warning but cumulative "
            f"degradation is significant."
        )

    return None
