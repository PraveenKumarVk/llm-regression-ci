"""
Core feature: answer a question grounded in an earnings report context chunk.
"""

from __future__ import annotations

import re
import time

from openai import AsyncOpenAI

from src.models import EarningsAnswer, EarningsQuery
from src.prompt_loader import load_version


def _build_messages(system_prompt: str, few_shot_examples: list, query: EarningsQuery) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    for ex in few_shot_examples:
        messages.append({
            "role": "user",
            "content": f"Context: {ex.context}\nQuestion: {ex.question}",
        })
        messages.append({"role": "assistant", "content": ex.answer})

    messages.append({
        "role": "user",
        "content": f"Context: {query.context_chunk}\nQuestion: {query.question}",
    })
    return messages


def _extract_citation(text: str) -> str | None:
    # Use regex so multi-line or extra-space citations parse correctly
    match = re.search(r"\[CITATION:\s*(.*?)\]", text, re.DOTALL)
    return match.group(1).strip() if match else None


async def answer_earnings_question(
    query: EarningsQuery,
    prompt_version: str = "v1.0.0",
    model: str = "gpt-4o",
    client: AsyncOpenAI | None = None,
) -> EarningsAnswer:
    prompt = load_version(prompt_version)
    messages = _build_messages(prompt.system_prompt, prompt.few_shot_examples, query)

    if client is None:
        client = AsyncOpenAI()

    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,  # determinism matters for regression testing
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    raw = response.choices[0].message.content or ""

    return EarningsAnswer(
        answer=raw,
        citation=_extract_citation(raw),
        is_refusal=raw.strip() == "NOT_IN_DOCUMENT",
        raw_response=raw,
        prompt_version=prompt_version,
        content_hash=prompt.content_hash,
        model=model,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        latency_ms=latency_ms,
    )
