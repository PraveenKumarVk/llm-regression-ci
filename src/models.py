"""
Single source of truth for all Pydantic v2 data models.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FewShotExample(BaseModel):
    question: str
    context: str
    answer: str


class PromptVersion(BaseModel):
    """
    A versioned prompt artifact loaded from prompts/<version>.yaml.

    content_hash is derived from system_prompt + few_shot_examples only —
    version and timestamp changes never affect caching or comparison.
    """

    version: str
    timestamp: str
    system_prompt: str
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)
    content_hash: str = Field(default="", init=False)

    @model_validator(mode="after")
    def _compute_hash(self) -> "PromptVersion":
        canonical = {
            "system_prompt": self.system_prompt,
            "few_shot_examples": [e.model_dump() for e in self.few_shot_examples],
        }
        self.content_hash = hashlib.sha256(
            json.dumps(canonical, sort_keys=True).encode()
        ).hexdigest()
        return self


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

SectionName = Literal[
    "financial_statements",
    "management_discussion",
    "notes_to_financial_statements",
    "quantitative_disclosures",
    "controls_and_procedures",
    "other",
]


class DocumentChunk(BaseModel):
    chunk_id: str               # e.g. "AAPL_10Q_Q3_2024_mda_003"
    document_id: str            # e.g. "AAPL_10Q_Q3_2024"
    ticker: str
    company_name: str
    filing_type: str            # "10-Q"
    period: str                 # "Q3 2024"
    period_end_date: str        # "2024-06-29"
    filing_date: str            # date SEC received the filing
    section: SectionName
    section_index: int          # chunk number within this section, 0-based
    text: str
    char_count: int
    source_url: str             # EDGAR URL of the source document


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------

class EarningsQuery(BaseModel):
    question: str
    context_chunk: str
    document_id: str            # e.g. "AAPL_10Q_Q3_2024"
    chunk_id: str               # stable ID for the chunk within the document
    time_period: str            # metadata only — not passed to the LLM


class EarningsAnswer(BaseModel):
    answer: str
    citation: str | None        # text extracted from [CITATION: ...], or None
    is_refusal: bool            # True when answer is exactly "NOT_IN_DOCUMENT"
    raw_response: str
    prompt_version: str
    content_hash: str           # hash of the prompt that produced this answer
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Golden dataset
# ---------------------------------------------------------------------------

FailureModeCategory = Literal[
    "numerical_extraction",     # exact number must be pulled from text
    "temporal_precision",       # answer depends on the correct time period
    "faithfulness",             # model must not hallucinate beyond the context
    "refusal_correctness",      # answer is NOT_IN_DOCUMENT — model must refuse
    "cross_document_conflict",  # context contains contradictory figures
    "multi_hop_calculation",    # answer requires arithmetic across multiple figures
]


class GoldenTestCase(BaseModel):
    # Identity
    id: str                             # stable, e.g. "TC_001"
    created_at: datetime
    dataset_version: str                # version of the dataset file itself

    # Input
    question: str
    context_chunk: str
    document_id: str
    chunk_id: str
    time_period: str

    # Expected output
    expected_answer_contains: list[str]     # key phrases that MUST appear in the answer
    expected_citation_contains: str | None  # key phrase the [CITATION: ...] must contain
    expected_is_refusal: bool               # True when correct answer is NOT_IN_DOCUMENT

    # Failure mode metadata
    difficulty: Literal["easy", "medium", "hard", "adversarial"]
    failure_mode_category: FailureModeCategory

    # Human-readable context
    notes: str                  # why this case was included
    known_tricky_aspect: str    # what specifically might trip the model

    # Per-scorer directives — key is ScorerDimension, presence means skip that scorer
    scorer_notes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_scorer_notes_keys(self) -> "GoldenTestCase":
        valid = {
            "numerical_accuracy", "refusal_correctness",
            "faithfulness", "citation_accuracy", "temporal_precision",
        }
        invalid = set(self.scorer_notes) - valid
        if invalid:
            raise ValueError(
                f"scorer_notes contains invalid dimension keys: {invalid}. "
                f"Valid keys: {valid}"
            )
        return self


class GoldenDataset(BaseModel):
    """Wrapper that ties a versioned list of test cases to a dataset version string."""

    version: str                        # e.g. "v1.0.0"
    created_at: datetime
    cases: list[GoldenTestCase]

    @property
    def size(self) -> int:
        return len(self.cases)

    def by_category(self) -> dict[str, list[GoldenTestCase]]:
        result: dict[str, list[GoldenTestCase]] = {}
        for case in self.cases:
            result.setdefault(case.failure_mode_category, []).append(case)
        return result

    def by_difficulty(self) -> dict[str, list[GoldenTestCase]]:
        result: dict[str, list[GoldenTestCase]] = {}
        for case in self.cases:
            result.setdefault(case.difficulty, []).append(case)
        return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

ScorerDimension = Literal[
    "numerical_accuracy",
    "refusal_correctness",
    "faithfulness",
    "citation_accuracy",
    "temporal_precision",
]


class ScoreResult(BaseModel):
    score: float
    passed: bool
    reasoning: str
    dimension: ScorerDimension
    skipped: bool = False


class TestCase(BaseModel):
    """A single QA test case drawn from a financial document."""

    id: str
    question: str
    context: str                        # excerpt from the earnings report
    expected_answer: str                # may be "NOT_IN_DOCUMENT"
    document_source: str                # e.g. "AAPL_10Q_Q3_2024"
    difficulty: Literal["easy", "medium", "hard", "edge"]
    notes: str = ""


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class EvalResult(BaseModel):
    run_id: str
    test_case_id: str
    prompt_version: str
    content_hash: str
    model_id: str
    raw_output: str
    citation_present: bool              # response contains [CITATION: ...]
    not_in_doc_correct: bool            # NOT_IN_DOCUMENT when it should be (or should not be)
    answer_score: float                 # 1–5 rubric from judge model
    latency_ms: float
    token_usage: TokenUsage
    cached: bool = False


class RunSummary(BaseModel):
    run_id: str
    prompt_version: str
    content_hash: str
    model_id: str
    judge_model_id: str
    dataset_version: str                # which golden dataset was used — tracked
    timestamp: datetime                 # separately from prompt_version so the
    total_cases: int                    # eval bar itself can be regression-checked
    citation_rate: float                # fraction where citation_present == True
    not_in_doc_accuracy: float          # accuracy on NOT_IN_DOCUMENT cases
    avg_answer_score: float
    status: Literal["pass", "warn", "critical"]
    baseline_run_id: str | None = None


class RegressionCase(BaseModel):
    test_case_id: str
    question_snippet: str               # first 120 chars of question
    baseline_output: str
    current_output: str
    expected_answer: str
    baseline_score: float
    current_score: float
    regression_type: Literal["score_drop", "citation_lost", "hallucinated_answer", "refusal_broken"]


class DiffReport(BaseModel):
    run_id: str
    baseline_run_id: str | None
    score_delta: float                  # avg_answer_score current − baseline
    citation_rate_delta: float
    not_in_doc_accuracy_delta: float
    regressions: list[RegressionCase]
    improvements: list[RegressionCase]
    status: Literal["pass", "warn", "critical"]


class DriftSnapshot(BaseModel):
    snapshot_id: str
    window_run_ids: list[str]
    moving_avg_score: float
    moving_avg_citation_rate: float
    moving_avg_not_in_doc_accuracy: float
    drift_flagged: bool
    flagged_reason: str | None = None
    timestamp: datetime
