"""
Convert a downloaded SEC filing (PDF or HTML) to clean plaintext.
"""

from __future__ import annotations

import re
from pathlib import Path


def extract_text_from_pdf(path: Path) -> str:
    import fitz  # PyMuPDF — import here so the module loads without it installed

    doc = fitz.open(str(path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def extract_text_from_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags common in EDGAR iXBRL filings
    for tag in soup(["script", "style", "ix:header", "ix:hidden", "head"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    return _normalize_whitespace(text)


def extract_text(path: Path) -> str:
    """Auto-detect format from file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix in {".htm", ".html"}:
        return extract_text_from_html(path)
    raise ValueError(f"Unsupported file format: {suffix}")


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of blank lines to a single blank line."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
