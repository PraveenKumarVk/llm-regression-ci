"""Tests for Phase 3 Step 2: the evaluation runner."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
import respx
from httpx import Response

from src.eval_runner import (
    EvalRunResult,
    TestCaseResult,
    _case_passed,
    _composite_score,
    _dimension_pass_rate,
    run_eval,
)
from src.golden_loader import save_dataset
from src.models import (
    EarningsAnswer,
    GoldenDataset,
    GoldenTestCase,
    ScoreResult,
)

_DIMENSIONS = [
    "numerical_accuracy",
    "refusal_correctness",
    "faithfulness",
    "citation_accuracy",
    "temporal_precision",
]

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_CONTEXT = "iPhone net sales were $39.3 billion for the third quarter of fiscal 2024."


def _tc(id: str = "TC_001", **overrides: object) -> GoldenTestCase:
    defaults: dict[str, object] = {
        "id": id,
        "created_at": datetime(2024, 1, 1),
        "dataset_version": "v1.0.0",
        "question": "What was iPhone revenue in Q3 2024?",
        "context_chunk": _CONTEXT,
        "document_id": "AAPL_10Q_Q3_2024",
        "chunk_id": "chunk_001",
        "time_period": "Q3 2024",
        "expected_answer_contains": ["39.3 billion"],
        "expected_citation_contains": "39.3 billion",
        "expected_is_refusal": False,
        "difficulty": "easy",
        "failure_mode_category": "numerical_extraction",
        "notes": "Basic test",
        "known_tricky_aspect": "none",
        "scorer_notes": {},
    }
    defaults.update(overrides)
    return GoldenTestCase(**defaults)


def _score(
    dim: str,
    score: float = 1.0,
    passed: bool = True,
    skipped: bool = False,
) -> ScoreResult:
    return ScoreResult(score=score, passed=passed, reasoning="test", dimension=dim, skipped=skipped)  # type: ignore[arg-type]


def _all_scores(**overrides: ScoreResult) -> dict[str, ScoreResult]:
    base = {dim: _score(dim) for dim in _DIMENSIONS}
    base.update(overrides)
    return base


def _ans(**overrides: object) -> EarningsAnswer:
    defaults: dict[str, object] = {
        "answer": "iPhone revenue was $39.3 billion in Q3 2024.",
        "citation": "iPhone net sales were $39.3 billion for the third quarter of fiscal 2024",
        "is_refusal": False,
        "raw_response": "iPhone revenue was $39.3 billion in Q3 2024.",
        "prompt_version": "v1.0.0",
        "content_hash": "abc123",
        "model": "gpt-4o",
        "input_tokens": 100,
        "output_tokens": 50,
        "latency_ms": 500.0,
    }
    defaults.update(overrides)
    return EarningsAnswer(**defaults)


def _tcr(
    test_case_id: str = "TC_001",
    scores: dict[str, ScoreResult] | None = None,
    composite: float = 1.0,
    passed: bool = True,
    difficulty: str = "easy",
    category: str = "numerical_extraction",
) -> TestCaseResult:
    return TestCaseResult(
        test_case_id=test_case_id,
        question="test question",
        difficulty=difficulty,
        failure_mode_category=category,
        answer=_ans(),
        scores=scores if scores is not None else _all_scores(),
        composite_score=composite,
        passed=passed,
    )


@pytest.fixture
def mock_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _openai_response(content: str, input_tokens: int = 100, output_tokens: int = 50) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens},
    }


def _anthropic_response(text: str) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-haiku-4-5-20251001",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 10},
    }


def _make_dataset(tmp_path: Path, cases: list[GoldenTestCase] | None = None) -> Path:
    if cases is None:
        cases = [_tc("TC_001"), _tc("TC_002")]
    dataset = GoldenDataset(version="v1.0.0", created_at=datetime(2024, 1, 1), cases=cases)
    return save_dataset(dataset, tmp_path)


# ---------------------------------------------------------------------------
# _composite_score
# ---------------------------------------------------------------------------


class TestCompositeScore:
    def test_all_full_scores_returns_1(self):
        assert _composite_score(_all_scores()) == pytest.approx(1.0)

    def test_all_zero_scores_returns_0(self):
        scores = {dim: _score(dim, score=0.0, passed=False) for dim in _DIMENSIONS}
        assert _composite_score(scores) == pytest.approx(0.0)

    def test_skipped_dimension_excluded_from_composite(self):
        scores = _all_scores(
            temporal_precision=_score("temporal_precision", score=0.0, passed=False, skipped=True)
        )
        # All active dimensions score 1.0; weights renormalize to sum=1.0
        assert _composite_score(scores) == pytest.approx(1.0)

    def test_weights_renormalize_correctly(self):
        # Only numerical_accuracy (weight 0.30) active and scores 0.5; others skipped
        scores = {dim: _score(dim, score=0.0, passed=False, skipped=True) for dim in _DIMENSIONS}
        scores["numerical_accuracy"] = _score("numerical_accuracy", score=0.5)
        assert _composite_score(scores) == pytest.approx(0.5)

    def test_all_skipped_returns_zero(self):
        scores = {dim: _score(dim, score=1.0, skipped=True) for dim in _DIMENSIONS}
        assert _composite_score(scores) == 0.0

    def test_partial_scores_weighted_correctly(self):
        # numerical(0.30)*1.0 + refusal(0.20)*0.0 + faith(0.25)*1.0 + cit(0.15)*0.0 + temp(0.10)*1.0
        # = 0.30 + 0.00 + 0.25 + 0.00 + 0.10 = 0.65
        scores = {
            "numerical_accuracy": _score("numerical_accuracy", score=1.0),
            "refusal_correctness": _score("refusal_correctness", score=0.0, passed=False),
            "faithfulness": _score("faithfulness", score=1.0),
            "citation_accuracy": _score("citation_accuracy", score=0.0, passed=False),
            "temporal_precision": _score("temporal_precision", score=1.0),
        }
        assert _composite_score(scores) == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# _case_passed
# ---------------------------------------------------------------------------


class TestCasePassed:
    def test_passes_when_composite_high_and_refusal_correct(self):
        assert _case_passed(0.90, _all_scores()) is True

    def test_threshold_is_inclusive_at_085(self):
        assert _case_passed(0.85, _all_scores()) is True

    def test_fails_just_below_threshold(self):
        assert _case_passed(0.849, _all_scores()) is False

    def test_fails_when_composite_below_threshold(self):
        assert _case_passed(0.80, _all_scores()) is False

    def test_refusal_hard_gate_blocks_high_composite(self):
        scores = _all_scores(
            refusal_correctness=_score("refusal_correctness", score=0.0, passed=False)
        )
        assert _case_passed(0.95, scores) is False

    def test_skipped_refusal_treated_as_passing(self):
        scores = _all_scores(
            refusal_correctness=_score("refusal_correctness", score=0.0, passed=False, skipped=True)
        )
        assert _case_passed(0.90, scores) is True


# ---------------------------------------------------------------------------
# _dimension_pass_rate
# ---------------------------------------------------------------------------


class TestDimensionPassRate:
    def test_all_pass_returns_1(self):
        results = [_tcr(), _tcr("TC_002")]
        assert _dimension_pass_rate(results, "numerical_accuracy") == pytest.approx(1.0)

    def test_none_pass_returns_0(self):
        results = [
            _tcr(scores=_all_scores(numerical_accuracy=_score("numerical_accuracy", score=0.0, passed=False))),
        ]
        assert _dimension_pass_rate(results, "numerical_accuracy") == pytest.approx(0.0)

    def test_all_skipped_returns_none(self):
        results = [
            _tcr(scores=_all_scores(temporal_precision=_score("temporal_precision", skipped=True))),
            _tcr("TC_002", scores=_all_scores(temporal_precision=_score("temporal_precision", skipped=True))),
        ]
        assert _dimension_pass_rate(results, "temporal_precision") is None

    def test_skipped_excluded_from_denominator(self):
        # 1 skipped, 2 pass, 1 fail → 2/3 not 2/4
        results = [
            _tcr("TC_001", scores=_all_scores(numerical_accuracy=_score("numerical_accuracy", skipped=True))),
            _tcr("TC_002"),  # passes
            _tcr("TC_003"),  # passes
            _tcr("TC_004", scores=_all_scores(numerical_accuracy=_score("numerical_accuracy", score=0.0, passed=False))),
        ]
        assert _dimension_pass_rate(results, "numerical_accuracy") == pytest.approx(2 / 3)

    def test_empty_results_returns_none(self):
        assert _dimension_pass_rate([], "numerical_accuracy") is None


# ---------------------------------------------------------------------------
# run_eval — integration tests with mocked APIs
# ---------------------------------------------------------------------------


class TestRunEval:
    @respx.mock
    async def test_result_structure(self, tmp_path: Path, mock_api_keys: None) -> None:
        dataset_file = _make_dataset(tmp_path)
        answer_text = "iPhone revenue was $39.3 billion in Q3 2024. [CITATION: iPhone net sales were $39.3 billion for the third quarter of fiscal 2024]"
        judge_text = json.dumps({"score": 5, "reasoning": "Fully grounded."})

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(200, json=_openai_response(answer_text))
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response(judge_text))
        )

        result = await run_eval(dataset_file, prompt_version="v1.0.0", model="gpt-4o")

        assert isinstance(result, EvalRunResult)
        assert result.prompt_version == "v1.0.0"
        assert result.model == "gpt-4o"
        assert result.dataset_version == "v1.0.0"
        assert len(result.test_results) == 2
        assert result.run_id  # non-empty UUID string

    @respx.mock
    async def test_dataset_version_comes_from_file(self, tmp_path: Path, mock_api_keys: None) -> None:
        cases = [_tc("TC_001", dataset_version="v2.3.1")]
        dataset = GoldenDataset(version="v2.3.1", created_at=datetime(2024, 1, 1), cases=cases)
        dataset_file = save_dataset(dataset, tmp_path)
        answer_text = "iPhone revenue was $39.3 billion in Q3 2024."
        judge_text = json.dumps({"score": 4, "reasoning": "Grounded."})

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(200, json=_openai_response(answer_text))
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response(judge_text))
        )

        result = await run_eval(dataset_file)
        assert result.dataset_version == "v2.3.1"

    @respx.mock
    async def test_pass_rate_by_difficulty_only_contains_present_levels(
        self, tmp_path: Path, mock_api_keys: None
    ) -> None:
        dataset_file = _make_dataset(tmp_path, cases=[_tc("TC_001", difficulty="hard")])
        answer_text = "iPhone revenue was $39.3 billion in Q3 2024."
        judge_text = json.dumps({"score": 5, "reasoning": "Grounded."})

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(200, json=_openai_response(answer_text))
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response(judge_text))
        )

        result = await run_eval(dataset_file)
        assert "hard" in result.pass_rate_by_difficulty
        assert "easy" not in result.pass_rate_by_difficulty

    @respx.mock
    async def test_skipped_dimension_returns_none_in_pass_rate(
        self, tmp_path: Path, mock_api_keys: None
    ) -> None:
        skipped_tc = _tc("TC_001", scorer_notes={"temporal_precision": "skip — year implied"})
        dataset_file = _make_dataset(tmp_path, cases=[skipped_tc])
        answer_text = "iPhone revenue was $39.3 billion in Q3 2024."
        judge_text = json.dumps({"score": 5, "reasoning": "Grounded."})

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(200, json=_openai_response(answer_text))
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response(judge_text))
        )

        result = await run_eval(dataset_file)
        assert result.pass_rate_by_dimension["temporal_precision"] is None
        # All other dimensions have real rates
        for dim in ("numerical_accuracy", "refusal_correctness", "faithfulness", "citation_accuracy"):
            assert result.pass_rate_by_dimension[dim] is not None

    @respx.mock
    async def test_cost_and_latency_aggregated(self, tmp_path: Path, mock_api_keys: None) -> None:
        dataset_file = _make_dataset(tmp_path, cases=[_tc("TC_001")])
        answer_text = "iPhone revenue was $39.3 billion in Q3 2024."
        judge_text = json.dumps({"score": 5, "reasoning": "Grounded."})

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(200, json=_openai_response(answer_text, input_tokens=200, output_tokens=100))
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response(judge_text))
        )

        result = await run_eval(dataset_file)
        expected_cost = 200 * 0.0000025 + 100 * 0.000010
        assert result.total_cost_usd == pytest.approx(expected_cost)
        assert result.avg_latency_ms > 0

    @respx.mock
    async def test_all_scores_present_per_case(self, tmp_path: Path, mock_api_keys: None) -> None:
        dataset_file = _make_dataset(tmp_path, cases=[_tc("TC_001")])
        answer_text = "iPhone revenue was $39.3 billion in Q3 2024."
        judge_text = json.dumps({"score": 5, "reasoning": "Grounded."})

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(200, json=_openai_response(answer_text))
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response(judge_text))
        )

        result = await run_eval(dataset_file)
        case = result.test_results[0]
        assert set(case.scores.keys()) == set(_DIMENSIONS)

    @respx.mock
    async def test_refusal_hard_gate_fails_case(self, tmp_path: Path, mock_api_keys: None) -> None:
        refusal_tc = _tc("TC_001", expected_is_refusal=True, expected_answer_contains=[])
        dataset_file = _make_dataset(tmp_path, cases=[refusal_tc])
        # Answer when model should have refused
        answer_text = "iPhone revenue was $39.3 billion in Q3 2024."
        judge_text = json.dumps({"score": 5, "reasoning": "Grounded."})

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(200, json=_openai_response(answer_text))
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response(judge_text))
        )

        result = await run_eval(dataset_file)
        case = result.test_results[0]
        assert case.scores["refusal_correctness"].passed is False
        assert case.passed is False
