"""Tests for Phase 3 Step 1: multi-dimensional scorers."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
import respx
from anthropic import AsyncAnthropic
from httpx import Response

from src.models import EarningsAnswer, GoldenTestCase
from src.scorer import (
    _parse_judge_response,
    normalize_financial_text,
    score_citation_accuracy,
    score_faithfulness_llm_judge,
    score_numerical_accuracy,
    score_refusal_correctness,
    score_temporal_precision,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONTEXT = "iPhone net sales were $39.3 billion for the third quarter of fiscal 2024."


def _tc(**overrides: object) -> GoldenTestCase:
    defaults: dict[str, object] = {
        "id": "TC_001",
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
        "notes": "Basic revenue extraction",
        "known_tricky_aspect": "multiple revenue lines",
    }
    defaults.update(overrides)
    return GoldenTestCase(**defaults)


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


def _anthropic_response(text: str) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-haiku-4-5-20251001",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


@pytest.fixture
def anthropic_client() -> AsyncAnthropic:
    return AsyncAnthropic(api_key="test-key")


# ---------------------------------------------------------------------------
# normalize_financial_text
# ---------------------------------------------------------------------------


class TestScorerNotesValidation:
    def test_invalid_dimension_key_raises(self):
        with pytest.raises(Exception, match="invalid dimension keys"):
            _tc(scorer_notes={"not_a_dimension": "some note"})

    def test_valid_dimension_keys_accepted(self):
        for dim in (
            "numerical_accuracy", "refusal_correctness",
            "faithfulness", "citation_accuracy", "temporal_precision",
        ):
            tc = _tc(scorer_notes={dim: "skip"})
            assert dim in tc.scorer_notes

    def test_empty_scorer_notes_accepted(self):
        tc = _tc(scorer_notes={})
        assert tc.scorer_notes == {}


class TestScoreResultDefaults:
    async def test_skipped_defaults_to_false(self):
        result = await score_refusal_correctness(_tc(), _ans())
        assert result.skipped is False

    async def test_numerical_accuracy_not_skipped(self):
        result = await score_numerical_accuracy(_tc(), _ans())
        assert result.skipped is False


class TestNormalizeFinancialText:
    def test_removes_dollar_sign(self):
        assert "$39.3" not in normalize_financial_text("$39.3 billion")

    def test_removes_commas(self):
        assert "," not in normalize_financial_text("1,234")

    def test_normalizes_billion(self):
        assert "39.3b" in normalize_financial_text("39.3 billion")
        assert "39.3b" in normalize_financial_text("39.3 Billion")

    def test_normalizes_million(self):
        assert "500m" in normalize_financial_text("500 million")

    def test_lowercases(self):
        result = normalize_financial_text("Revenue")
        assert result == result.lower()

    def test_strips_whitespace(self):
        assert normalize_financial_text("  $1.2 billion  ") == normalize_financial_text(
            "$1.2 billion"
        )


# ---------------------------------------------------------------------------
# score_numerical_accuracy
# ---------------------------------------------------------------------------


class TestScoreNumericalAccuracy:
    async def test_all_phrases_found_returns_1(self):
        result = await score_numerical_accuracy(_tc(), _ans())
        assert result.score == 1.0
        assert result.passed is True
        assert result.dimension == "numerical_accuracy"

    async def test_no_phrase_found_returns_0(self):
        answer = _ans(answer="Revenue was significant this quarter.")
        result = await score_numerical_accuracy(_tc(), answer)
        assert result.score == 0.0
        assert result.passed is False

    async def test_partial_match_scores_proportionally(self):
        tc = _tc(expected_answer_contains=["39.3 billion", "fiscal 2024"])
        answer = _ans(answer="Revenue was 39.3 billion.")
        result = await score_numerical_accuracy(tc, answer)
        assert result.score == pytest.approx(0.5)
        assert result.passed is False

    async def test_unexpected_refusal_scores_zero(self):
        answer = _ans(answer="NOT_IN_DOCUMENT", is_refusal=True, citation=None)
        result = await score_numerical_accuracy(_tc(), answer)
        assert result.score == 0.0
        assert "refused" in result.reasoning.lower()

    async def test_expected_refusal_skips_scoring(self):
        tc = _tc(expected_is_refusal=True, expected_answer_contains=[])
        answer = _ans(answer="NOT_IN_DOCUMENT", is_refusal=True, citation=None)
        result = await score_numerical_accuracy(tc, answer)
        assert result.score == 1.0
        assert result.passed is True

    async def test_normalizes_billion_variants(self):
        tc = _tc(expected_answer_contains=["39.3 billion"])
        answer = _ans(answer="Revenue was $39,300,000,000 (39.3 Billion) in Q3 2024.")
        result = await score_numerical_accuracy(tc, answer)
        assert result.score == 1.0

    async def test_empty_expected_phrases_scores_full(self):
        tc = _tc(expected_is_refusal=False, expected_answer_contains=[])
        result = await score_numerical_accuracy(tc, _ans())
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# score_refusal_correctness
# ---------------------------------------------------------------------------


class TestScoreRefusalCorrectness:
    async def test_correct_refusal(self):
        tc = _tc(expected_is_refusal=True)
        answer = _ans(answer="NOT_IN_DOCUMENT", is_refusal=True, citation=None)
        result = await score_refusal_correctness(tc, answer)
        assert result.score == 1.0
        assert result.passed is True
        assert "correctly refused" in result.reasoning.lower()

    async def test_correct_answer(self):
        result = await score_refusal_correctness(_tc(), _ans())
        assert result.score == 1.0
        assert result.passed is True
        assert "correctly answered" in result.reasoning.lower()

    async def test_should_have_refused_but_answered(self):
        tc = _tc(expected_is_refusal=True)
        result = await score_refusal_correctness(tc, _ans())
        assert result.score == 0.0
        assert result.passed is False
        assert "should have refused" in result.reasoning.lower()

    async def test_should_have_answered_but_refused(self):
        answer = _ans(answer="NOT_IN_DOCUMENT", is_refusal=True, citation=None)
        result = await score_refusal_correctness(_tc(), answer)
        assert result.score == 0.0
        assert result.passed is False
        assert "should have answered" in result.reasoning.lower()

    async def test_dimension_label(self):
        result = await score_refusal_correctness(_tc(), _ans())
        assert result.dimension == "refusal_correctness"


# ---------------------------------------------------------------------------
# score_faithfulness_llm_judge
# ---------------------------------------------------------------------------


class TestParseJudgeResponse:
    def test_pure_json(self):
        score, reasoning = _parse_judge_response('{"score": 4, "reasoning": "Good."}')
        assert score == 4
        assert reasoning == "Good."

    def test_prose_wrapped_json(self):
        raw = 'Here is my evaluation:\n{"score": 3, "reasoning": "Mostly supported."}\nHope that helps.'
        score, reasoning = _parse_judge_response(raw)
        assert score == 3
        assert reasoning == "Mostly supported."

    def test_score_above_5_clamped(self):
        score, _ = _parse_judge_response('{"score": 7, "reasoning": "Perfect."}')
        assert score == 5

    def test_score_below_0_clamped(self):
        score, _ = _parse_judge_response('{"score": -2, "reasoning": "Bad."}')
        assert score == 0

    def test_float_score_rounded(self):
        score, _ = _parse_judge_response('{"score": 3.7, "reasoning": "OK."}')
        assert score == 4

    def test_string_score_coerced(self):
        score, _ = _parse_judge_response('{"score": "5", "reasoning": "Great."}')
        assert score == 5

    def test_missing_reasoning_returns_empty_string(self):
        _, reasoning = _parse_judge_response('{"score": 4}')
        assert reasoning == ""

    def test_reasoning_with_braces_parses_correctly(self):
        # [^{}]* regex breaks when reasoning itself contains braces
        raw = 'Here is my output:\n{"score": 4, "reasoning": "Uses {B} notation correctly."}'
        score, reasoning = _parse_judge_response(raw)
        assert score == 4
        assert "Uses {B} notation correctly." in reasoning

    def test_no_json_raises_value_error(self):
        with pytest.raises(ValueError, match="No valid JSON with score key"):
            _parse_judge_response("The answer is faithful and well-grounded.")


class TestScoreFaithfulnessLlmJudge:
    @respx.mock
    async def test_high_faithfulness_passes(self, anthropic_client):
        judge_output = json.dumps({"score": 5, "reasoning": "Every claim is supported."})
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response(judge_output))
        )
        result = await score_faithfulness_llm_judge(_tc(), _ans(), client=anthropic_client)
        assert result.score == pytest.approx(1.0)
        assert result.passed is True
        assert result.dimension == "faithfulness"

    @respx.mock
    async def test_low_faithfulness_fails(self, anthropic_client):
        judge_output = json.dumps({"score": 1, "reasoning": "Answer contradicts context."})
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response(judge_output))
        )
        result = await score_faithfulness_llm_judge(_tc(), _ans(), client=anthropic_client)
        assert result.score == pytest.approx(0.2)
        assert result.passed is False

    @respx.mock
    async def test_score_4_of_5_passes(self, anthropic_client):
        judge_output = json.dumps({"score": 4, "reasoning": "Minor paraphrase acceptable."})
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response(judge_output))
        )
        result = await score_faithfulness_llm_judge(_tc(), _ans(), client=anthropic_client)
        assert result.score == pytest.approx(0.8)
        assert result.passed is True

    async def test_refusal_skips_judge(self, anthropic_client):
        answer = _ans(answer="NOT_IN_DOCUMENT", is_refusal=True, citation=None)
        result = await score_faithfulness_llm_judge(_tc(), answer, client=anthropic_client)
        assert result.score == 1.0
        assert result.passed is True
        assert "not applicable" in result.reasoning.lower()

    @respx.mock
    async def test_prose_wrapped_json_parses_correctly(self, anthropic_client):
        judge_output = 'Sure! Here is the score:\n{"score": 5, "reasoning": "Fully grounded."}'
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response(judge_output))
        )
        result = await score_faithfulness_llm_judge(_tc(), _ans(), client=anthropic_client)
        assert result.score == pytest.approx(1.0)
        assert result.passed is True

    @respx.mock
    async def test_out_of_range_score_is_clamped(self, anthropic_client):
        judge_output = json.dumps({"score": 8, "reasoning": "Very good."})
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response(judge_output))
        )
        result = await score_faithfulness_llm_judge(_tc(), _ans(), client=anthropic_client)
        assert result.score == pytest.approx(1.0)
        assert result.passed is True

    @respx.mock
    async def test_unparseable_response_returns_sentinel(self, anthropic_client):
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("I cannot evaluate this."))
        )
        result = await score_faithfulness_llm_judge(_tc(), _ans(), client=anthropic_client)
        assert result.score == 0.0
        assert result.passed is False
        assert "parse error" in result.reasoning.lower()
        assert result.dimension == "faithfulness"

    @respx.mock
    async def test_reasoning_propagated(self, anthropic_client):
        reasoning = "All figures directly quoted from the context."
        judge_output = json.dumps({"score": 5, "reasoning": reasoning})
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response(judge_output))
        )
        result = await score_faithfulness_llm_judge(_tc(), _ans(), client=anthropic_client)
        assert result.reasoning == reasoning


# ---------------------------------------------------------------------------
# score_citation_accuracy
# ---------------------------------------------------------------------------


class TestScoreCitationAccuracy:
    async def test_grounded_and_relevant_citation(self):
        result = await score_citation_accuracy(_tc(), _ans())
        assert result.score == 1.0
        assert result.passed is True

    async def test_missing_citation_when_expected(self):
        answer = _ans(citation=None)
        result = await score_citation_accuracy(_tc(), answer)
        assert result.score == 0.0
        assert result.passed is False
        assert "none was produced" in result.reasoning.lower()

    async def test_hallucinated_citation(self):
        answer = _ans(citation="This fact was never in the document at all.")
        result = await score_citation_accuracy(_tc(), answer)
        assert result.score == 0.0
        assert "hallucinated" in result.reasoning.lower()

    async def test_grounded_but_wrong_phrase(self):
        answer = _ans(citation="iPhone net sales were")
        tc = _tc(expected_citation_contains="third quarter of fiscal 2024")
        result = await score_citation_accuracy(tc, answer)
        assert result.score == pytest.approx(0.5)
        assert result.passed is False
        assert "wrong phrase" in result.reasoning.lower()

    async def test_no_citation_expected_scores_full(self):
        tc = _tc(expected_citation_contains=None)
        result = await score_citation_accuracy(tc, _ans())
        assert result.score == 1.0
        assert result.passed is True

    async def test_refusal_skips_citation_check(self):
        answer = _ans(answer="NOT_IN_DOCUMENT", is_refusal=True, citation=None)
        result = await score_citation_accuracy(_tc(), answer)
        assert result.score == 1.0

    async def test_dimension_label(self):
        result = await score_citation_accuracy(_tc(), _ans())
        assert result.dimension == "citation_accuracy"


# ---------------------------------------------------------------------------
# score_temporal_precision
# ---------------------------------------------------------------------------


class TestScoreTemporalPrecision:
    async def test_skipped_when_scorer_notes_present(self):
        tc = _tc(scorer_notes={"temporal_precision": "year implied by filing context"})
        result = await score_temporal_precision(tc, _ans())
        assert result.skipped is True
        assert result.passed is False
        assert result.score == 0.0
        assert "year implied by filing context" in result.reasoning
        assert result.dimension == "temporal_precision"

    async def test_not_skipped_when_scorer_notes_empty(self):
        answer = _ans(answer="iPhone revenue was $39.3 billion in Q3 2024.")
        result = await score_temporal_precision(_tc(), answer)
        assert result.skipped is False

    async def test_scorer_notes_for_other_dimension_does_not_skip(self):
        tc = _tc(scorer_notes={"citation_accuracy": "some other note"})
        answer = _ans(answer="Revenue was $39.3 billion in Q3 2024.")
        result = await score_temporal_precision(tc, answer)
        assert result.skipped is False
        assert result.passed is True

    async def test_correct_period_found(self):
        answer = _ans(answer="iPhone revenue was $39.3 billion in Q3 2024.")
        result = await score_temporal_precision(_tc(), answer)
        assert result.score == 1.0
        assert result.passed is True
        assert "Q3 2024" in result.reasoning

    async def test_wrong_period_fails(self):
        answer = _ans(answer="iPhone revenue was $39.3 billion in Q2 2024.")
        result = await score_temporal_precision(_tc(), answer)
        assert result.score == 0.0
        assert result.passed is False
        assert "Q3 2024" in result.reasoning

    async def test_case_insensitive_match(self):
        answer = _ans(answer="iPhone revenue was $39.3 billion in q3 2024.")
        result = await score_temporal_precision(_tc(), answer)
        assert result.score == 1.0

    async def test_refusal_skips_temporal_check(self):
        answer = _ans(answer="NOT_IN_DOCUMENT", is_refusal=True, citation=None)
        result = await score_temporal_precision(_tc(), answer)
        assert result.score == 1.0
        assert result.passed is True
        assert "not applicable" in result.reasoning.lower()

    async def test_dimension_label(self):
        result = await score_temporal_precision(_tc(), _ans())
        assert result.dimension == "temporal_precision"
