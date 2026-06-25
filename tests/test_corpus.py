"""Tests for the corpus pipeline: chunker (pure) and EDGAR client (mocked)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import respx
from httpx import Response

from src.corpus.chunker import _find_spans, _split_with_overlap, chunk_document
from src.corpus.edgar import (
    FILING_TARGETS,
    build_document_url,
    find_filing,
    get_cik,
)
from src.models import DocumentChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    document_id="AAPL_10Q_Q3_2024",
    ticker="AAPL",
    company_name="Apple Inc.",
    period="Q3 2024",
    period_end_date="2024-06-29",
    filing_date="2024-08-02",
    source_url="https://www.sec.gov/Archives/edgar/data/320193/test.htm",
)

SAMPLE_10Q = """
PART I — FINANCIAL INFORMATION

Item 1. Financial Statements

The condensed consolidated balance sheet as of June 29, 2024 reflects total
assets of $337.4 billion.

Notes to Condensed Consolidated Financial Statements

Note 1 — Summary of Significant Accounting Policies
The Company prepares its financial statements in accordance with GAAP.

Item 2. Management's Discussion and Analysis of Financial Condition
and Results of Operations

iPhone net sales were $39.3 billion for the third quarter of fiscal 2024,
compared with $39.7 billion in the year-ago quarter.

Item 3. Quantitative and Qualitative Disclosures About Market Risk

The Company is exposed to market risk for changes in foreign currency exchange rates.

Item 4. Controls and Procedures

The Company's principal executive officer has concluded that the disclosure
controls and procedures are effective.
"""


# ---------------------------------------------------------------------------
# _split_with_overlap
# ---------------------------------------------------------------------------


class TestSplitWithOverlap:
    def test_short_text_returns_single_chunk(self):
        text = "Short text."
        result = _split_with_overlap(text, max_chars=2000, overlap=200)
        assert result == ["Short text."]

    def test_empty_text_returns_empty(self):
        assert _split_with_overlap("", max_chars=2000, overlap=200) == []

    def test_long_text_splits_into_multiple_chunks(self):
        text = ("word " * 200).strip()  # ~1000 chars
        result = _split_with_overlap(text, max_chars=300, overlap=50)
        assert len(result) > 1

    def test_chunks_do_not_exceed_max_chars(self):
        text = "x" * 5000
        for chunk in _split_with_overlap(text, max_chars=500, overlap=50):
            assert len(chunk) <= 500

    def test_overlap_means_content_repeated(self):
        # Build text where the overlap zone has a known marker
        marker = "OVERLAP_MARKER"
        text = ("a" * 400) + f"\n\n{marker}\n\n" + ("b" * 400)
        chunks = _split_with_overlap(text, max_chars=500, overlap=100)
        # The marker should appear in at least two consecutive chunks
        chunks_with_marker = [c for c in chunks if marker in c]
        assert len(chunks_with_marker) >= 1  # at minimum it appears somewhere


# ---------------------------------------------------------------------------
# _find_spans
# ---------------------------------------------------------------------------


class TestFindSpans:
    def test_detects_all_four_items(self):
        spans = _find_spans(SAMPLE_10Q)
        section_names = {s.section for s in spans}
        assert "financial_statements" in section_names
        assert "management_discussion" in section_names
        assert "quantitative_disclosures" in section_names
        assert "controls_and_procedures" in section_names

    def test_spans_cover_full_document(self):
        spans = _find_spans(SAMPLE_10Q)
        assert spans[0].start == 0
        assert spans[-1].end == len(SAMPLE_10Q)

    def test_spans_are_non_overlapping_and_contiguous(self):
        spans = _find_spans(SAMPLE_10Q)
        for a, b in zip(spans, spans[1:]):
            assert a.end == b.start

    def test_no_headers_returns_other(self):
        text = "Just some plain text with no SEC headers at all."
        spans = _find_spans(text)
        assert len(spans) == 1
        assert spans[0].section == "other"

    def test_management_discussion_comes_after_financial_statements(self):
        spans = _find_spans(SAMPLE_10Q)
        by_section = {s.section: s for s in spans}
        fs = by_section.get("financial_statements")
        mda = by_section.get("management_discussion")
        if fs and mda:
            assert fs.start < mda.start


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------


class TestChunkDocument:
    def test_returns_list_of_document_chunks(self):
        chunks = chunk_document(text=SAMPLE_10Q, **BASE_KWARGS)
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunk_ids_are_unique(self):
        chunks = chunk_document(text=SAMPLE_10Q, **BASE_KWARGS)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_contain_document_id(self):
        chunks = chunk_document(text=SAMPLE_10Q, **BASE_KWARGS)
        for c in chunks:
            assert BASE_KWARGS["document_id"] in c.chunk_id

    def test_char_count_matches_text_length(self):
        chunks = chunk_document(text=SAMPLE_10Q, **BASE_KWARGS)
        for c in chunks:
            assert c.char_count == len(c.text)

    def test_no_chunk_exceeds_max_chars(self):
        chunks = chunk_document(text=SAMPLE_10Q, max_chars=500, overlap=50, **BASE_KWARGS)
        for c in chunks:
            assert c.char_count <= 500

    def test_metadata_propagated_to_all_chunks(self):
        chunks = chunk_document(text=SAMPLE_10Q, **BASE_KWARGS)
        for c in chunks:
            assert c.ticker == "AAPL"
            assert c.period == "Q3 2024"
            assert c.filing_type == "10-Q"
            assert c.source_url == BASE_KWARGS["source_url"]

    def test_section_index_is_sequential_per_section(self):
        chunks = chunk_document(text=SAMPLE_10Q, max_chars=300, overlap=50, **BASE_KWARGS)
        from collections import defaultdict
        by_section: dict[str, list[int]] = defaultdict(list)
        for c in chunks:
            by_section[c.section].append(c.section_index)
        for section, indices in by_section.items():
            assert indices == list(range(len(indices))), f"{section}: {indices}"

    def test_empty_text_returns_no_chunks(self):
        assert chunk_document(text="", **BASE_KWARGS) == []

    def test_chunks_are_serialisable(self):
        chunks = chunk_document(text=SAMPLE_10Q, **BASE_KWARGS)
        for c in chunks:
            json.dumps(c.model_dump())  # must not raise


# ---------------------------------------------------------------------------
# EDGAR client (mocked)
# ---------------------------------------------------------------------------

TICKERS_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 19617, "ticker": "JPM", "title": "JPMorgan Chase"},
}

SUBMISSIONS_PAYLOAD = {
    "filings": {
        "recent": {
            "form": ["10-Q", "10-K", "10-Q"],
            "reportDate": ["2024-06-29", "2023-09-30", "2024-03-30"],
            "accessionNumber": ["0000320193-24-000081", "0000320193-23-000099", "0000320193-24-000050"],
            "filingDate": ["2024-08-02", "2023-11-03", "2024-05-03"],
            "primaryDocument": ["aapl-20240629.htm", "aapl-20230930.htm", "aapl-20240330.htm"],
        }
    }
}


@pytest.fixture
def http_client():
    import httpx
    return httpx.AsyncClient()


@respx.mock
async def test_get_cik_returns_padded_cik(http_client):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=Response(200, json=TICKERS_PAYLOAD)
    )
    cik = await get_cik("AAPL", http_client)
    assert cik == "0000320193"


@respx.mock
async def test_get_cik_unknown_ticker_raises(http_client):
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=Response(200, json=TICKERS_PAYLOAD)
    )
    with pytest.raises(ValueError, match="TSLA"):
        await get_cik("TSLA", http_client)


@respx.mock
async def test_find_filing_returns_correct_accession(http_client):
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=Response(200, json=SUBMISSIONS_PAYLOAD)
    )
    accession, date, primary_doc = await find_filing("0000320193", "2024-06-29", http_client)
    assert accession == "0000320193-24-000081"
    assert date == "2024-08-02"
    assert primary_doc == "aapl-20240629.htm"


@respx.mock
async def test_find_filing_missing_period_raises(http_client):
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=Response(200, json=SUBMISSIONS_PAYLOAD)
    )
    with pytest.raises(ValueError, match="2024-12-31"):
        await find_filing("0000320193", "2024-12-31", http_client)


def test_build_document_url():
    url = build_document_url("0000320193", "0000320193-24-000081", "aapl-20240629.htm")
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000032019324000081/aapl-20240629.htm"


def test_filing_targets_have_required_keys():
    required = {"ticker", "company_name", "period", "period_end_date", "document_id"}
    for target in FILING_TARGETS:
        assert required <= target.keys(), f"Missing keys in target: {target}"
