"""
Multi-dimensional scorers for Phase 3 evaluation.
Each scorer takes a GoldenTestCase and an EarningsAnswer and returns a ScoreResult (0.0–1.0).
"""

from __future__ import annotations

import json
import re

from anthropic import AsyncAnthropic

from src.models import EarningsAnswer, GoldenTestCase, ScoreResult

_JUDGE_PROMPT = """\
You are evaluating whether an AI answer is faithfully grounded in the provided context.

Context: {context}

Question: {question}

Answer: {answer}

Score the faithfulness of this answer from 0 to 5:
5 = Every claim is directly supported by the context
4 = All main claims supported, minor paraphrase acceptable
3 = Most claims supported, one ungrounded inference
2 = Some claims supported but significant hallucination present
1 = Answer contradicts or ignores the context
0 = Complete hallucination

Respond with ONLY a JSON object: {{"score": <int>, "reasoning": "<one sentence>"}}\
"""


def normalize_financial_text(text: str) -> str:
    text = text.lower()
    text = text.replace("$", "").replace(",", "")
    text = re.sub(r"(\d+\.?\d*)\s*billion", r"\1b", text)
    text = re.sub(r"(\d+\.?\d*)\s*million", r"\1m", text)
    return text.strip()


async def score_numerical_accuracy(
    test_case: GoldenTestCase,
    answer: EarningsAnswer,
) -> ScoreResult:
    if answer.is_refusal and not test_case.expected_is_refusal:
        return ScoreResult(
            score=0.0,
            passed=False,
            reasoning="Model refused to answer but answer was expected",
            dimension="numerical_accuracy",
        )

    if not test_case.expected_is_refusal:
        hits = sum(
            1
            for phrase in test_case.expected_answer_contains
            if normalize_financial_text(phrase) in normalize_financial_text(answer.answer)
        )
        total = len(test_case.expected_answer_contains)
        score = hits / total if total else 1.0
        return ScoreResult(
            score=score,
            passed=score >= 0.9,
            reasoning=f"{hits}/{total} expected phrases found",
            dimension="numerical_accuracy",
        )

    return ScoreResult(
        score=1.0,
        passed=True,
        reasoning="Refusal case — not scored on numerical accuracy",
        dimension="numerical_accuracy",
    )


async def score_refusal_correctness(
    test_case: GoldenTestCase,
    answer: EarningsAnswer,
) -> ScoreResult:
    correct = answer.is_refusal == test_case.expected_is_refusal
    if correct and answer.is_refusal:
        reasoning = "Correctly refused"
    elif correct:
        reasoning = "Correctly answered"
    elif not answer.is_refusal:
        reasoning = "Should have refused but answered"
    else:
        reasoning = "Should have answered but refused"
    return ScoreResult(
        score=1.0 if correct else 0.0,
        passed=correct,
        reasoning=reasoning,
        dimension="refusal_correctness",
    )


def _parse_judge_response(raw_text: str) -> tuple[int, str]:
    """Extract (score, reasoning) from judge output, tolerating prose wrappers and bad ranges."""
    try:
        result = json.loads(raw_text.strip())
    except json.JSONDecodeError:
        # Regex like [^{}]* breaks when the reasoning value itself contains braces.
        # raw_decode() is the actual JSON parser — it handles nested braces and
        # quoted strings correctly. Scan every '{' and collect all valid objects
        # that contain a "score" key; take the last one (real payload is usually last).
        decoder = json.JSONDecoder()
        candidates: list[dict] = []
        idx = 0
        while idx < len(raw_text):
            start = raw_text.find("{", idx)
            if start == -1:
                break
            try:
                obj, end = decoder.raw_decode(raw_text, start)
                if isinstance(obj, dict) and "score" in obj:
                    candidates.append(obj)
                idx = end
            except json.JSONDecodeError:
                idx = start + 1
        if not candidates:
            raise ValueError(f"No valid JSON with score key: {raw_text[:200]!r}")
        result = candidates[-1]
    score = max(0, min(5, int(round(float(result["score"])))))
    reasoning = str(result.get("reasoning", ""))
    return score, reasoning


async def score_faithfulness_llm_judge(
    test_case: GoldenTestCase,
    answer: EarningsAnswer,
    client: AsyncAnthropic | None = None,
) -> ScoreResult:
    if answer.is_refusal:
        return ScoreResult(
            score=1.0,
            passed=True,
            reasoning="Refusal — faithfulness not applicable",
            dimension="faithfulness",
        )

    if client is None:
        client = AsyncAnthropic()

    prompt = _JUDGE_PROMPT.format(
        context=test_case.context_chunk,
        question=test_case.question,
        answer=answer.answer,
    )
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        raw_score, reasoning = _parse_judge_response(response.content[0].text)
    except (ValueError, KeyError, TypeError) as exc:
        return ScoreResult(
            score=0.0,
            passed=False,
            reasoning=f"Judge parse error: {exc}",
            dimension="faithfulness",
        )
    normalized_score = raw_score / 5.0
    return ScoreResult(
        score=normalized_score,
        passed=normalized_score >= 0.8,
        reasoning=reasoning,
        dimension="faithfulness",
    )


async def score_citation_accuracy(
    test_case: GoldenTestCase,
    answer: EarningsAnswer,
) -> ScoreResult:
    if answer.is_refusal or not test_case.expected_citation_contains:
        return ScoreResult(
            score=1.0,
            passed=True,
            reasoning="No citation expected",
            dimension="citation_accuracy",
        )

    if not answer.citation:
        return ScoreResult(
            score=0.0,
            passed=False,
            reasoning="Answer expected a citation but none was produced",
            dimension="citation_accuracy",
        )

    citation_grounded = answer.citation.lower() in test_case.context_chunk.lower()
    citation_relevant = test_case.expected_citation_contains.lower() in answer.citation.lower()

    if citation_grounded and citation_relevant:
        score, reasoning = 1.0, "Citation is grounded and contains expected phrase"
    elif citation_grounded:
        score, reasoning = 0.5, "Citation is grounded in context but cites wrong phrase"
    else:
        score, reasoning = 0.0, "Citation not found verbatim in context — hallucinated"

    return ScoreResult(
        score=score,
        passed=score >= 0.8,
        reasoning=reasoning,
        dimension="citation_accuracy",
    )


async def score_temporal_precision(
    test_case: GoldenTestCase,
    answer: EarningsAnswer,
) -> ScoreResult:
    if "temporal_precision" in test_case.scorer_notes:
        return ScoreResult(
            score=0.0,
            passed=False,
            skipped=True,
            reasoning=f"Skipped — {test_case.scorer_notes['temporal_precision']}",
            dimension="temporal_precision",
        )

    if answer.is_refusal:
        return ScoreResult(
            score=1.0,
            passed=True,
            reasoning="Refusal — temporal precision not applicable",
            dimension="temporal_precision",
        )

    if test_case.time_period.lower() in answer.answer.lower():
        return ScoreResult(
            score=1.0,
            passed=True,
            reasoning=f"Correct time period '{test_case.time_period}' found",
            dimension="temporal_precision",
        )

    return ScoreResult(
        score=0.0,
        passed=False,
        reasoning=f"Expected time period '{test_case.time_period}' not found in answer",
        dimension="temporal_precision",
    )
