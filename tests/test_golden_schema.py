"""Tests for Phase 2 Step 1: GoldenTestCase and GoldenDataset schema."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.models import FailureModeCategory, GoldenDataset, GoldenTestCase

NOW = datetime(2026, 6, 25, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Minimal valid fixture
# ---------------------------------------------------------------------------

VALID_KWARGS = dict(
    id="TC_001",
    created_at=NOW,
    dataset_version="v1.0.0",
    question="What was iPhone revenue in Q3 2024?",
    context_chunk="iPhone net sales were $39.3 billion for the third quarter of fiscal 2024.",
    document_id="AAPL_10Q_Q3_2024",
    chunk_id="AAPL_10Q_Q3_2024_mda_001",
    time_period="Q3 2024",
    expected_answer_contains=["$39.3 billion", "Q3 2024"],
    expected_citation_contains="iPhone net sales were $39.3 billion",
    expected_is_refusal=False,
    difficulty="easy",
    failure_mode_category="numerical_extraction",
    notes="Baseline numerical extraction from a clean context.",
    known_tricky_aspect="Model might omit the time period qualifier.",
)


# ---------------------------------------------------------------------------
# GoldenTestCase construction
# ---------------------------------------------------------------------------


class TestGoldenTestCase:
    def test_valid_case_constructs(self):
        tc = GoldenTestCase(**VALID_KWARGS)
        assert tc.id == "TC_001"

    def test_refusal_case_has_no_citation(self):
        kwargs = {**VALID_KWARGS, "expected_is_refusal": True, "expected_citation_contains": None}
        tc = GoldenTestCase(**kwargs)
        assert tc.expected_citation_contains is None
        assert tc.expected_is_refusal is True

    def test_expected_answer_contains_is_list(self):
        tc = GoldenTestCase(**VALID_KWARGS)
        assert isinstance(tc.expected_answer_contains, list)
        assert len(tc.expected_answer_contains) == 2

    def test_invalid_difficulty_raises(self):
        with pytest.raises(ValidationError):
            GoldenTestCase(**{**VALID_KWARGS, "difficulty": "trivial"})

    def test_invalid_failure_mode_raises(self):
        with pytest.raises(ValidationError):
            GoldenTestCase(**{**VALID_KWARGS, "failure_mode_category": "hallucination"})

    def test_all_difficulty_levels_accepted(self):
        for level in ("easy", "medium", "hard", "adversarial"):
            tc = GoldenTestCase(**{**VALID_KWARGS, "difficulty": level})
            assert tc.difficulty == level

    def test_all_failure_mode_categories_accepted(self):
        categories: list[FailureModeCategory] = [
            "numerical_extraction",
            "temporal_precision",
            "faithfulness",
            "refusal_correctness",
            "cross_document_conflict",
            "multi_hop_calculation",
        ]
        for cat in categories:
            tc = GoldenTestCase(**{**VALID_KWARGS, "failure_mode_category": cat})
            assert tc.failure_mode_category == cat

    def test_empty_expected_answer_contains_allowed(self):
        # Refusal cases legitimately have no expected phrases
        tc = GoldenTestCase(**{
            **VALID_KWARGS,
            "expected_answer_contains": [],
            "expected_is_refusal": True,
        })
        assert tc.expected_answer_contains == []

    def test_serialises_to_json(self):
        import json
        tc = GoldenTestCase(**VALID_KWARGS)
        dumped = json.dumps(tc.model_dump(mode="json"))
        reloaded = GoldenTestCase.model_validate_json(dumped)
        assert reloaded.id == tc.id
        assert reloaded.expected_answer_contains == tc.expected_answer_contains


# ---------------------------------------------------------------------------
# GoldenDataset
# ---------------------------------------------------------------------------


def _make_case(id: str, category: FailureModeCategory, difficulty: str) -> GoldenTestCase:
    return GoldenTestCase(**{
        **VALID_KWARGS,
        "id": id,
        "failure_mode_category": category,
        "difficulty": difficulty,
    })


class TestGoldenDataset:
    def test_size_property(self):
        ds = GoldenDataset(
            version="v1.0.0",
            created_at=NOW,
            cases=[GoldenTestCase(**VALID_KWARGS)],
        )
        assert ds.size == 1

    def test_empty_dataset(self):
        ds = GoldenDataset(version="v1.0.0", created_at=NOW, cases=[])
        assert ds.size == 0

    def test_by_category_groups_correctly(self):
        ds = GoldenDataset(
            version="v1.0.0",
            created_at=NOW,
            cases=[
                _make_case("TC_001", "numerical_extraction", "easy"),
                _make_case("TC_002", "numerical_extraction", "hard"),
                _make_case("TC_003", "refusal_correctness", "easy"),
            ],
        )
        grouped = ds.by_category()
        assert len(grouped["numerical_extraction"]) == 2
        assert len(grouped["refusal_correctness"]) == 1
        assert "temporal_precision" not in grouped

    def test_by_difficulty_groups_correctly(self):
        ds = GoldenDataset(
            version="v1.0.0",
            created_at=NOW,
            cases=[
                _make_case("TC_001", "numerical_extraction", "easy"),
                _make_case("TC_002", "faithfulness", "adversarial"),
                _make_case("TC_003", "refusal_correctness", "easy"),
            ],
        )
        grouped = ds.by_difficulty()
        assert len(grouped["easy"]) == 2
        assert len(grouped["adversarial"]) == 1

    def test_all_six_failure_modes_representable(self):
        categories: list[FailureModeCategory] = [
            "numerical_extraction",
            "temporal_precision",
            "faithfulness",
            "refusal_correctness",
            "cross_document_conflict",
            "multi_hop_calculation",
        ]
        cases = [_make_case(f"TC_{i:03d}", cat, "easy") for i, cat in enumerate(categories)]
        ds = GoldenDataset(version="v1.0.0", created_at=NOW, cases=cases)
        assert set(ds.by_category().keys()) == set(categories)
