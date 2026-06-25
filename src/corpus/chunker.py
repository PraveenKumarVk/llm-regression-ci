"""
Split extracted 10-Q text into labelled, size-bounded chunks by SEC section.

Strategy:
  1. Scan the full text for known section-header patterns (Item 1, Item 2, …).
  2. Split the document at those boundaries; each span gets a SectionName label.
  3. Further divide long spans into overlapping sub-chunks of ~MAX_CHARS characters,
     breaking on paragraph boundaries where possible.

Overlap means consecutive chunks share ~OVERLAP chars — important so a question
that straddles a chunk boundary still has an answer in at least one chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.models import DocumentChunk, SectionName

MAX_CHARS = 2000
OVERLAP = 200

# Ordered list of (SectionName, regex) pairs.
# We scan for the first match of each pattern to find section boundaries.
# Patterns are designed for the plain-text output of PyMuPDF / BeautifulSoup,
# where formatting is lost but case is preserved from the original filing.
_SECTION_PATTERNS: list[tuple[SectionName, str]] = [
    (
        "financial_statements",
        r"(?im)^.*item\s+1[\.\s—-]+financial\s+statements",
    ),
    (
        "management_discussion",
        r"(?im)^.*item\s+2[\.\s—-]+management.s\s+discussion",
    ),
    (
        "quantitative_disclosures",
        r"(?im)^.*item\s+3[\.\s—-]+quantitative",
    ),
    (
        "controls_and_procedures",
        r"(?im)^.*item\s+4[\.\s—-]+controls",
    ),
    (
        "notes_to_financial_statements",
        r"(?im)^.*notes\s+to\s+(condensed\s+)?(consolidated\s+)?financial\s+statements",
    ),
]


@dataclass
class _Span:
    section: SectionName
    start: int
    end: int


def _find_spans(text: str) -> list[_Span]:
    """
    Locate section boundaries and return a list of non-overlapping spans.
    Sections not preceded by any recognised header are labelled 'other'.
    """
    hits: list[tuple[int, SectionName]] = []
    for section_name, pattern in _SECTION_PATTERNS:
        for m in re.finditer(pattern, text):
            hits.append((m.start(), section_name))

    if not hits:
        return [_Span(section="other", start=0, end=len(text))]

    hits.sort(key=lambda h: h[0])

    # Deduplicate: if the same section appears more than once (e.g. a TOC entry
    # followed by the real section), keep the last occurrence.
    seen: dict[SectionName, int] = {}
    for pos, name in hits:
        seen[name] = pos
    hits = sorted((pos, name) for name, pos in seen.items())

    spans: list[_Span] = []
    if hits[0][0] > 0:
        spans.append(_Span(section="other", start=0, end=hits[0][0]))

    for i, (start, name) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        spans.append(_Span(section=name, start=start, end=end))

    return spans


def _split_with_overlap(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Break text into chunks of at most max_chars, preferring paragraph breaks.
    Adjacent chunks share `overlap` characters so no sentence is stranded.
    """
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        # Try to break on a paragraph boundary within the last 20% of the window
        if end < len(text):
            search_start = pos + int(max_chars * 0.8)
            para_break = text.rfind("\n\n", search_start, end)
            if para_break != -1:
                end = para_break + 2
            else:
                # Fall back to a newline
                newline = text.rfind("\n", search_start, end)
                if newline != -1:
                    end = newline + 1

        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        pos = end - overlap  # back up by overlap for the next chunk

    return chunks


def chunk_document(
    text: str,
    document_id: str,
    ticker: str,
    company_name: str,
    period: str,
    period_end_date: str,
    filing_date: str,
    source_url: str,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP,
) -> list[DocumentChunk]:
    spans = _find_spans(text)

    chunks: list[DocumentChunk] = []
    for span in spans:
        span_text = text[span.start : span.end]
        sub_chunks = _split_with_overlap(span_text, max_chars, overlap)
        for idx, sub in enumerate(sub_chunks):
            section_abbrev = {
                "financial_statements": "fs",
                "management_discussion": "mda",
                "notes_to_financial_statements": "notes",
                "quantitative_disclosures": "qd",
                "controls_and_procedures": "cp",
                "other": "other",
            }[span.section]
            chunk_id = f"{document_id}_{section_abbrev}_{idx:03d}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    ticker=ticker,
                    company_name=company_name,
                    filing_type="10-Q",
                    period=period,
                    period_end_date=period_end_date,
                    filing_date=filing_date,
                    section=span.section,
                    section_index=idx,
                    text=sub,
                    char_count=len(sub),
                    source_url=source_url,
                )
            )

    return chunks
