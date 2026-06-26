"""Tests for Phase 3 Step 4: detect_slow_drift."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.drift_detector import detect_slow_drift
from src.eval_runner import EvalRunResult

_DIMS = [
    "numerical_accuracy",
    "refusal_correctness",
    "faithfulness",
    "citation_accuracy",
    "temporal_precision",
]

_counter = 0


def _run(pass_rate: float) -> EvalRunResult:
    global _counter
    _counter += 1
    return EvalRunResult(
        run_id=f"run-{_counter:04d}",
        prompt_version="v1.0.0",
        model="gpt-4o",
        dataset_version="v1.0.0",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        test_results=[],
        overall_pass_rate=pass_rate,
        pass_rate_by_dimension={dim: pass_rate for dim in _DIMS},
        pass_rate_by_difficulty={"easy": pass_rate},
        pass_rate_by_category={"numerical_extraction": pass_rate},
        total_cost_usd=0.01,
        avg_latency_ms=500.0,
    )


def _window(pass_rate: float, size: int = 7) -> list[EvalRunResult]:
    return [_run(pass_rate) for _ in range(size)]


# ---------------------------------------------------------------------------
# Insufficient history
# ---------------------------------------------------------------------------


class TestInsufficientHistory:
    def test_empty_history_returns_none(self):
        assert detect_slow_drift([]) is None

    def test_one_window_returns_none(self):
        assert detect_slow_drift(_window(0.9, 7), window_size=7) is None

    def test_one_fewer_than_two_windows_returns_none(self):
        runs = _window(0.9, 7) + _window(0.7, 6)  # 13 < 14
        assert detect_slow_drift(runs, window_size=7) is None

    def test_exactly_two_windows_is_sufficient(self):
        runs = _window(0.9) + _window(0.7)  # 14 == 7*2
        assert detect_slow_drift(runs, window_size=7) is not None

    def test_custom_window_size_respected(self):
        # window_size=3 needs 6 runs; 5 is not enough
        assert detect_slow_drift(_window(0.7, 5), window_size=3) is None
        assert detect_slow_drift(_window(0.9, 3) + _window(0.7, 3), window_size=3) is not None


# ---------------------------------------------------------------------------
# Threshold behaviour
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_drift_below_threshold_returns_none(self):
        # drift = 0.86 - 0.90 = -0.04, clearly above the -0.05 threshold
        runs = _window(0.90) + _window(0.86)
        assert detect_slow_drift(runs) is None

    def test_drift_above_threshold_returns_message(self):
        # drift = 0.84 - 0.90 = -0.06 < -0.05
        runs = _window(0.90) + _window(0.84)
        assert detect_slow_drift(runs) is not None

    def test_no_drift_returns_none(self):
        runs = _window(0.9) + _window(0.9)
        assert detect_slow_drift(runs) is None

    def test_positive_drift_returns_none(self):
        runs = _window(0.7) + _window(0.9)
        assert detect_slow_drift(runs) is None

    def test_large_drift_returns_message(self):
        runs = _window(0.95) + _window(0.80)
        assert detect_slow_drift(runs) is not None


# ---------------------------------------------------------------------------
# Message content
# ---------------------------------------------------------------------------


class TestMessageContent:
    def test_message_contains_window_size(self):
        runs = _window(0.90) + _window(0.80)
        msg = detect_slow_drift(runs, window_size=7)
        assert "7" in msg  # type: ignore[operator]

    def test_message_contains_formatted_percentages(self):
        runs = _window(0.90) + _window(0.80)
        msg = detect_slow_drift(runs, window_size=7)
        assert msg is not None
        assert "90.0%" in msg   # previous avg
        assert "80.0%" in msg   # recent avg
        assert "10.0%" in msg   # abs drift

    def test_message_contains_slow_drift_label(self):
        runs = _window(0.90) + _window(0.70)
        msg = detect_slow_drift(runs, window_size=7)
        assert msg is not None
        assert "SLOW DRIFT DETECTED" in msg

    def test_message_uses_custom_window_size(self):
        runs = _window(0.90, 3) + _window(0.70, 3)
        msg = detect_slow_drift(runs, window_size=3)
        assert msg is not None
        assert "3" in msg


# ---------------------------------------------------------------------------
# Window slicing
# ---------------------------------------------------------------------------


class TestWindowSlicing:
    def test_only_last_two_windows_compared(self):
        # 3 windows: old=0.5, prev=0.9, curr=0.84
        # drift between prev and curr = -0.06 → triggers
        # drift between old and prev = +0.4 → irrelevant
        runs = _window(0.50) + _window(0.90) + _window(0.84)
        msg = detect_slow_drift(runs, window_size=7)
        assert msg is not None
        assert "90.0%" in msg   # previous avg from second window, not first

    def test_old_window_does_not_affect_result(self):
        # Even with very bad old history, if last two windows are stable → clean
        runs = _window(0.20) + _window(0.90) + _window(0.88)
        assert detect_slow_drift(runs, window_size=7) is None

    def test_extra_runs_beyond_two_windows_ignored(self):
        # 4 windows; only last two matter
        runs = _window(0.95) + _window(0.95) + _window(0.90) + _window(0.80)
        msg = detect_slow_drift(runs, window_size=7)
        assert msg is not None
        # Recent=0.80, previous=0.90 — not 0.95
        assert "80.0%" in msg
        assert "90.0%" in msg
