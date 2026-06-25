"""Tests for Phase 1 Step 2: the earnings QA feature function."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import respx
from httpx import Response
from openai import AsyncOpenAI

from src.earnings_qa import _build_messages, _extract_citation, answer_earnings_question
from src.models import EarningsQuery
from src.prompt_loader import load_version

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

QUERY = EarningsQuery(
    question="What was iPhone revenue in Q3 2024?",
    context_chunk="iPhone net sales were $39.3 billion for the third quarter of fiscal 2024.",
    document_id="AAPL_10Q_Q3_2024",
    chunk_id="chunk_001",
    time_period="Q3 2024",
)

REFUSAL_QUERY = EarningsQuery(
    question="What was the CEO's bonus in Q3 2024?",
    context_chunk="iPhone net sales were $39.3 billion for the third quarter of fiscal 2024.",
    document_id="AAPL_10Q_Q3_2024",
    chunk_id="chunk_001",
    time_period="Q3 2024",
)


def _openai_response(content: str, input_tokens: int = 100, output_tokens: int = 50) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


@pytest.fixture
def openai_client():
    return AsyncOpenAI(api_key="test-key")


# ---------------------------------------------------------------------------
# _extract_citation
# ---------------------------------------------------------------------------


class TestExtractCitation:
    def test_extracts_single_line(self):
        raw = "Revenue was $39.3B [CITATION: iPhone net sales were $39.3 billion]"
        assert _extract_citation(raw) == "iPhone net sales were $39.3 billion"

    def test_strips_extra_spaces(self):
        assert _extract_citation("Answer [CITATION:   some phrase   ]") == "some phrase"

    def test_returns_none_when_absent(self):
        assert _extract_citation("NOT_IN_DOCUMENT") is None
        assert _extract_citation("Plain answer.") is None

    def test_multiline_citation(self):
        raw = "[CITATION: iPhone net sales\nwere $39.3 billion]"
        assert _extract_citation(raw) == "iPhone net sales\nwere $39.3 billion"

    def test_first_citation_wins(self):
        assert _extract_citation("[CITATION: first] text [CITATION: second]") == "first"


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_system_message_is_first(self):
        prompt = load_version("v1.0.0", PROMPTS_DIR)
        msgs = _build_messages(prompt.system_prompt, prompt.few_shot_examples, QUERY)
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == prompt.system_prompt

    def test_few_shot_pairs_interleaved(self):
        prompt = load_version("v1.0.0", PROMPTS_DIR)
        msgs = _build_messages(prompt.system_prompt, prompt.few_shot_examples, QUERY)
        for i, ex in enumerate(prompt.few_shot_examples):
            user_msg = msgs[1 + i * 2]
            asst_msg = msgs[2 + i * 2]
            assert user_msg["role"] == "user"
            assert ex.question in user_msg["content"]
            assert ex.context in user_msg["content"]
            assert asst_msg["role"] == "assistant"
            assert asst_msg["content"] == ex.answer

    def test_actual_query_is_last(self):
        prompt = load_version("v1.0.0", PROMPTS_DIR)
        msgs = _build_messages(prompt.system_prompt, prompt.few_shot_examples, QUERY)
        last = msgs[-1]
        assert last["role"] == "user"
        assert QUERY.question in last["content"]
        assert QUERY.context_chunk in last["content"]

    def test_metadata_fields_not_in_messages(self):
        """document_id and chunk_id are opaque identifiers — must never reach the LLM.
        time_period is not checked because it naturally appears in questions."""
        prompt = load_version("v1.0.0", PROMPTS_DIR)
        msgs = _build_messages(prompt.system_prompt, prompt.few_shot_examples, QUERY)
        all_content = " ".join(m["content"] for m in msgs)
        assert QUERY.document_id not in all_content
        assert QUERY.chunk_id not in all_content


# ---------------------------------------------------------------------------
# answer_earnings_question (OpenAI mocked via respx)
# ---------------------------------------------------------------------------


@respx.mock
async def test_citation_answer(openai_client):
    raw = (
        "iPhone revenue was $39.3 billion in Q3 2024 "
        "[CITATION: iPhone net sales were $39.3 billion for the third quarter of fiscal 2024]"
    )
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(200, json=_openai_response(raw))
    )

    result = await answer_earnings_question(QUERY, client=openai_client)

    assert result.is_refusal is False
    assert result.citation == "iPhone net sales were $39.3 billion for the third quarter of fiscal 2024"
    assert result.prompt_version == "v1.0.0"
    assert len(result.content_hash) == 64


@respx.mock
async def test_refusal_answer(openai_client):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(200, json=_openai_response("NOT_IN_DOCUMENT"))
    )

    result = await answer_earnings_question(REFUSAL_QUERY, client=openai_client)

    assert result.is_refusal is True
    assert result.citation is None


@respx.mock
async def test_token_counts_captured(openai_client):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(200, json=_openai_response("NOT_IN_DOCUMENT", input_tokens=123, output_tokens=7))
    )

    result = await answer_earnings_question(REFUSAL_QUERY, client=openai_client)

    assert result.input_tokens == 123
    assert result.output_tokens == 7


@respx.mock
async def test_latency_is_positive(openai_client):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(200, json=_openai_response("NOT_IN_DOCUMENT"))
    )

    result = await answer_earnings_question(QUERY, client=openai_client)

    assert result.latency_ms > 0


@respx.mock
async def test_temperature_zero_sent_to_api(openai_client):
    """temperature=0 is required for determinism across regression runs."""
    captured: list[dict] = []

    def handler(request):
        captured.append(json.loads(request.content))
        return Response(200, json=_openai_response("NOT_IN_DOCUMENT"))

    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=handler)
    await answer_earnings_question(QUERY, client=openai_client)

    assert captured[0]["temperature"] == 0.0


@respx.mock
async def test_model_override(openai_client):
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(200, json=_openai_response("NOT_IN_DOCUMENT"))
    )

    result = await answer_earnings_question(QUERY, model="gpt-4o-mini", client=openai_client)

    assert result.model == "gpt-4o-mini"
