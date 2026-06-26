"""Tests for src/golden_loader.py."""

from __future__ import annotations

import json
import re

import pytest

from src.golden_loader import (
    filter_by_category,
    filter_by_difficulty,
    filter_refusals,
    list_versions,
    load_dataset,
    load_latest,
    load_version,
    save_dataset,
)
from src.models import GoldenDataset


def normalize_financial_text(text: str) -> str:
    """Lowercase, strip $ and remove commas between digits for fuzzy number matching."""
    text = text.lower().replace("$", "")
    # Remove commas between digits repeatedly (handles 1,234,567 → 1234567)
    while re.search(r"\d,\d", text):
        text = re.sub(r"(\d),(\d)", r"\1\2", text)
    return text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = "2026-06-25T00:00:00+00:00"
_BASE = dict(
    id="TC_001",
    created_at=_NOW,
    dataset_version="v1.0.0",
    question="What was net income for Q3 2024?",
    context_chunk="Net income was $21.4 billion.",
    document_id="AAPL_10Q_Q3_2024",
    chunk_id="AAPL_10Q_Q3_2024_fs_000",
    time_period="Q3 2024",
    expected_answer_contains=["21.4"],
    expected_citation_contains="Net income was",
    expected_is_refusal=False,
    difficulty="easy",
    failure_mode_category="numerical_extraction",
    notes="baseline",
    known_tricky_aspect="none",
)


def _make_case(**overrides) -> dict:
    return {**_BASE, **overrides}


def _dataset_json(cases: list[dict]) -> str:
    return json.dumps(
        {
            "version": "v1.0.0",
            "created_at": _NOW,
            "cases": cases,
        }
    )


# ---------------------------------------------------------------------------
# load_dataset — explicit path (test fixtures only)
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_load_from_explicit_path(self, tmp_path):
        f = tmp_path / "ds.json"
        f.write_text(_dataset_json([_make_case()]))
        ds = load_dataset(f)
        assert ds.size == 1
        assert ds.cases[0].id == "TC_001"

    def test_load_round_trips_all_fields(self, tmp_path):
        f = tmp_path / "ds.json"
        f.write_text(_dataset_json([_make_case()]))
        ds = load_dataset(f)
        tc = ds.cases[0]
        assert tc.question == _BASE["question"]
        assert tc.expected_answer_contains == ["21.4"]
        assert tc.expected_is_refusal is False

    def test_load_multiple_cases(self, tmp_path):
        cases = [
            _make_case(id="TC_001"),
            _make_case(id="TC_002", failure_mode_category="temporal_precision"),
            _make_case(id="TC_003", expected_is_refusal=True, expected_citation_contains=None),
        ]
        f = tmp_path / "ds.json"
        f.write_text(_dataset_json(cases))
        ds = load_dataset(f)
        assert ds.size == 3

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.json")

    def test_invalid_json_raises(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json at all")
        with pytest.raises(Exception):
            load_dataset(f)

    def test_invalid_schema_raises(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text(json.dumps({"version": "v1.0.0", "cases": [{"bad": "shape"}]}))
        with pytest.raises(Exception):
            load_dataset(f)


# ---------------------------------------------------------------------------
# Versioned API: save_dataset / load_version / load_latest / list_versions
# ---------------------------------------------------------------------------


def _make_dataset(version: str = "v1.0.0", n: int = 2) -> GoldenDataset:
    return GoldenDataset.model_validate(
        {"version": version, "created_at": _NOW, "cases": [_make_case(id=f"TC_{i:03d}") for i in range(n)]}
    )


class TestVersionedDataset:
    def test_save_writes_versioned_filename(self, tmp_path):
        ds = _make_dataset("v1.0.0")
        p = save_dataset(ds, tmp_path)
        assert p.name == "golden_dataset_v1.0.0.json"
        assert p.exists()

    def test_save_filename_reflects_version(self, tmp_path):
        for v in ("v1.0.0", "v1.1.0", "v2.0.0"):
            save_dataset(_make_dataset(v), tmp_path)
        names = {p.name for p in tmp_path.glob("*.json")}
        assert names == {
            "golden_dataset_v1.0.0.json",
            "golden_dataset_v1.1.0.json",
            "golden_dataset_v2.0.0.json",
        }

    def test_list_versions_empty_dir(self, tmp_path):
        assert list_versions(tmp_path) == []

    def test_list_versions_sorted_by_semver(self, tmp_path):
        for v in ("v1.10.0", "v1.2.0", "v2.0.0", "v1.0.0"):
            save_dataset(_make_dataset(v), tmp_path)
        assert list_versions(tmp_path) == ["v1.0.0", "v1.2.0", "v1.10.0", "v2.0.0"]

    def test_list_versions_ignores_non_matching_files(self, tmp_path):
        (tmp_path / "golden_dataset.json").write_text("{}")
        (tmp_path / "golden_dataset_v1.0.json").write_text("{}")  # wrong: only 2 parts
        save_dataset(_make_dataset("v1.0.0"), tmp_path)
        assert list_versions(tmp_path) == ["v1.0.0"]

    def test_load_version_loads_correct_file(self, tmp_path):
        save_dataset(_make_dataset("v1.0.0", n=2), tmp_path)
        save_dataset(_make_dataset("v1.1.0", n=5), tmp_path)
        ds = load_version("v1.0.0", tmp_path)
        assert ds.version == "v1.0.0"
        assert ds.size == 2

    def test_load_version_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_version("v9.9.9", tmp_path)

    def test_load_latest_returns_highest_semver(self, tmp_path):
        save_dataset(_make_dataset("v1.0.0", n=1), tmp_path)
        save_dataset(_make_dataset("v1.10.0", n=10), tmp_path)  # 1.10.0 > 1.2.0
        save_dataset(_make_dataset("v1.2.0", n=2), tmp_path)
        ds = load_latest(tmp_path)
        assert ds.version == "v1.10.0"
        assert ds.size == 10

    def test_load_latest_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_latest(tmp_path)

    def test_save_load_roundtrip(self, tmp_path):
        ds = _make_dataset("v1.1.0", n=3)
        save_dataset(ds, tmp_path)
        reloaded = load_version("v1.1.0", tmp_path)
        assert reloaded.size == 3
        assert reloaded.version == "v1.1.0"

    def test_load_latest_real_data(self):
        ds = load_latest()
        assert ds.size >= 75
        assert ds.version == "v1.0.0"

    def test_list_versions_real_data(self):
        versions = list_versions()
        assert "v1.0.0" in versions




# ---------------------------------------------------------------------------
# filter helpers
# ---------------------------------------------------------------------------


class TestFilterHelpers:
    def _load_real(self) -> GoldenDataset:
        return load_latest()

    def test_filter_by_category_numerical(self):
        ds = self._load_real()
        results = filter_by_category(ds, "numerical_extraction")
        assert len(results) >= 20
        assert all(c.failure_mode_category == "numerical_extraction" for c in results)

    def test_filter_by_category_temporal(self):
        ds = self._load_real()
        results = filter_by_category(ds, "temporal_precision")
        assert len(results) >= 15
        assert all(c.failure_mode_category == "temporal_precision" for c in results)

    def test_filter_by_category_unknown_returns_empty(self):
        ds = self._load_real()
        results = filter_by_category(ds, "does_not_exist")
        assert results == []

    def test_filter_by_difficulty_hard(self):
        ds = self._load_real()
        results = filter_by_difficulty(ds, "hard")
        assert len(results) >= 10
        assert all(c.difficulty == "hard" for c in results)

    def test_filter_by_difficulty_adversarial(self):
        ds = self._load_real()
        results = filter_by_difficulty(ds, "adversarial")
        assert len(results) >= 5

    def test_filter_refusals(self):
        ds = self._load_real()
        results = filter_refusals(ds)
        assert len(results) >= 15
        assert all(c.expected_is_refusal for c in results)

    def test_filter_refusals_have_no_expected_answers(self):
        ds = self._load_real()
        for c in filter_refusals(ds):
            assert c.expected_answer_contains == []
            assert c.expected_citation_contains is None

    def test_all_cases_have_non_empty_context_chunk(self):
        ds = self._load_real()
        for c in ds.cases:
            assert len(c.context_chunk) > 200, f"{c.id} has suspiciously short context"

    def test_all_case_ids_are_unique(self):
        ds = self._load_real()
        ids = [c.id for c in ds.cases]
        assert len(ids) == len(set(ids))

    def test_dataset_covers_all_five_companies(self):
        ds = self._load_real()
        doc_ids = {c.document_id for c in ds.cases}
        expected = {
            "AAPL_10Q_Q3_2024",
            "JPM_10Q_Q2_2024",
            "JNJ_10Q_Q3_2024",
            "AMZN_10Q_Q2_2024",
            "XOM_10Q_Q3_2024",
        }
        assert expected.issubset(doc_ids)


# ---------------------------------------------------------------------------
# Truthfulness / integrity of dataset content
# ---------------------------------------------------------------------------


def _normalize_for_citation(text: str) -> str:
    """Collapse whitespace so PDF newline artifacts don't break substring checks."""
    return " ".join(text.lower().split())


def test_expected_citation_appears_verbatim_in_context():
    """
    The citation scorer checks this at runtime against LLM output.
    If ground truth citations don't exist in the context, every case
    will false-fail on citation_accuracy.
    Whitespace is normalised to tolerate PDF extraction newline artifacts.
    """
    ds = load_latest()
    failures = []
    for c in ds.cases:
        if c.expected_citation_contains is None:
            continue
        if _normalize_for_citation(c.expected_citation_contains) not in _normalize_for_citation(c.context_chunk):
            failures.append(
                f"{c.id}: expected_citation_contains "
                f"'{c.expected_citation_contains}' "
                f"not found in context_chunk (after whitespace normalisation)"
            )
    assert failures == [], "\n".join(failures)


def test_expected_answer_phrases_appear_in_context():
    """
    Every phrase in expected_answer_contains should be derivable from the
    context chunk. Multi-hop calculation cases are excluded because their
    answer is the *result* of arithmetic on context figures, not a verbatim
    phrase — the input numbers (which ARE in context) drive the answer.
    """
    ds = load_latest()
    failures = []
    for c in ds.cases:
        if c.expected_is_refusal:
            continue
        if c.failure_mode_category == "multi_hop_calculation":
            continue  # result is computed, not verbatim in context
        for phrase in c.expected_answer_contains:
            normalized_context = normalize_financial_text(c.context_chunk)
            normalized_phrase = normalize_financial_text(phrase)
            if normalized_phrase not in normalized_context:
                failures.append(
                    f"{c.id}: expected phrase '{phrase}' "
                    f"not found in context after normalization"
                )
    assert failures == [], "\n".join(failures)


def _time_period_in_context(time_period: str, context: str) -> bool:
    """
    Return True if *time_period* can be verified as present in *context*.

    Handles:
    - Direct substring (e.g. "Q3 2024" appears verbatim)
    - Quarter → "third quarter", "three months", "september 30" etc.
    - "Nine months" → also "fiscal nine months" (JNJ phrasing)
    - "H1/H2" → "six months" / "first half"
    - "… guidance" suffix → stripped before matching
    - Pure data tables (no 4-digit year in chunk at all) → exempt; temporal
      grounding comes from document provenance, not chunk text

    The year stated in time_period MUST appear in the chunk text. Cases where
    the year is implied but not stated are classified adversarial difficulty and
    exempted explicitly in the test, not here.
    """
    ctx = context.lower()
    tp_base = re.sub(r"\s*guidance\s*", "", time_period, flags=re.IGNORECASE).strip()

    # Direct match
    if tp_base.lower() in ctx:
        return True

    # Chunks with no 4-digit year are date-free data tables — exempt
    if not re.search(r"\b20\d{2}\b", context):
        return True

    year_match = re.search(r"\b(20\d{2})\b", tp_base)
    if not year_match:
        return True  # no year to verify
    year = year_match.group(1)

    # The stated year must appear verbatim — no fallback to prior year
    if year not in context:
        return False

    _QUARTER_VARIANTS: dict[str, list[str]] = {
        "Q1": ["first quarter", "q1", "1q", "march 31", "three months ended march"],
        "Q2": ["second quarter", "q2", "2q", "june 30", "three months ended june"],
        "Q3": ["third quarter", "q3", "3q", "september 30", "three months"],
        "Q4": ["fourth quarter", "q4", "4q", "december 31", "three months ended december"],
    }

    tp_upper = tp_base.upper()
    for q, variants in _QUARTER_VARIANTS.items():
        if tp_upper.startswith(q):
            return any(v in ctx for v in variants)

    tp_lower_base = tp_base.lower()
    if "nine months" in tp_lower_base:
        return any(v in ctx for v in ["nine months", "fiscal nine months"])
    if tp_upper.startswith("H1"):
        return any(v in ctx for v in ["six months", "first half"])
    if tp_upper.startswith("H2"):
        return any(v in ctx for v in ["six months", "second half"])
    if tp_upper.startswith("FY"):
        return True  # FY cases are refusals and excluded by caller

    return False


def test_time_period_appears_in_context_for_non_refusal_cases():
    """
    Validates that each case's time_period is grounded in its context_chunk.
    The temporal-precision scorer depends on this: if the period isn't
    represented in the chunk, the case is testing the wrong thing.

    Two categories are explicitly exempt:
    - Pure data-table chunks (no 4-digit year in text): temporal grounding
      comes from document provenance, not chunk text.
    - Adversarial cases where known_tricky_aspect documents "implied" year:
      these are intentionally hard cases where the LLM must infer the current
      period from filing context rather than an explicit year in the chunk.
      Example: AMZN_MDA025 Q3 2024 guidance section says "compared with third
      quarter 2023" — "2024" is never printed; the period is implied by the
      forward-looking nature of the sentence.
    """
    ds = load_latest()
    failures = []
    for c in ds.cases:
        if c.expected_is_refusal:
            continue
        # Intentionally hard: year is implied by document context, not stated
        # verbatim. Must be adversarial difficulty with "implied" documented
        # in known_tricky_aspect so there's an auditable record per case.
        if c.difficulty == "adversarial" and "implied" in c.known_tricky_aspect.lower():
            continue
        if not _time_period_in_context(c.time_period, c.context_chunk):
            failures.append(
                f"{c.id}: time_period '{c.time_period}' not found "
                f"in context_chunk in any recognized format"
            )
    assert failures == [], "\n".join(failures)


def test_refusal_cases_answer_not_derivable_from_context():
    """
    Flags refusal cases whose context contains financial figures — may mean
    the answer is actually present and the case is mislabeled.
    This is a soft warning, not a hard fail.
    """
    ds = load_latest()
    suspicious = []
    for c in ds.cases:
        if not c.expected_is_refusal:
            continue
        if re.search(
            r"\$[\d,]+\.?\d*\s*(billion|million|B|M)\b",
            c.context_chunk,
            re.IGNORECASE,
        ):
            suspicious.append(
                f"{c.id}: refusal case but context contains "
                f"financial figures — verify manually"
            )
    if suspicious:
        print("\nREVIEW THESE REFUSAL CASES:\n" + "\n".join(suspicious))
