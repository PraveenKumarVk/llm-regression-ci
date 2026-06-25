"""
SEC EDGAR client: discovers and downloads 10-Q filings.

Rate limit: EDGAR asks for ≤10 req/s and a descriptive User-Agent.
We add a small delay between requests and always send the header.

Key discovery: EDGAR 10-Qs are HTML only (no PDF). The submissions JSON
already includes `primaryDocument`, so no separate filing-index fetch is needed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
TICKER_JSON = "https://www.sec.gov/files/company_tickers.json"

HEADERS = {
    "User-Agent": "model-regression-detection research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Five companies covering distinct financial-document terminology structures
FILING_TARGETS = [
    {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "period": "Q3 2024",
        "period_end_date": "2024-06-29",
        "document_id": "AAPL_10Q_Q3_2024",
    },
    {
        "ticker": "JPM",
        "company_name": "JPMorgan Chase & Co.",
        "period": "Q2 2024",
        "period_end_date": "2024-06-30",
        "document_id": "JPM_10Q_Q2_2024",
    },
    {
        "ticker": "JNJ",
        "company_name": "Johnson & Johnson",
        "period": "Q3 2024",
        "period_end_date": "2024-09-29",
        "document_id": "JNJ_10Q_Q3_2024",
    },
    {
        "ticker": "AMZN",
        "company_name": "Amazon.com Inc.",
        "period": "Q2 2024",
        "period_end_date": "2024-06-30",
        "document_id": "AMZN_10Q_Q2_2024",
    },
    {
        "ticker": "XOM",
        "company_name": "Exxon Mobil Corporation",
        "period": "Q3 2024",
        "period_end_date": "2024-09-30",
        "document_id": "XOM_10Q_Q3_2024",
    },
]


async def get_cik(ticker: str, client: httpx.AsyncClient) -> str:
    """Return the zero-padded 10-digit CIK for a ticker symbol."""
    resp = await client.get(TICKER_JSON, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"].upper() == ticker_upper:
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker '{ticker}' not found in EDGAR company list")


async def find_filing(
    cik: str,
    period_end_date: str,
    client: httpx.AsyncClient,
) -> tuple[str, str, str]:
    """
    Return (accession_number, filing_date, primary_document_filename) for the
    10-Q whose reportDate matches period_end_date exactly (YYYY-MM-DD).
    All three fields come from a single submissions JSON fetch.
    """
    resp = await client.get(f"{EDGAR_SUBMISSIONS}/CIK{cik}.json", headers=HEADERS)
    resp.raise_for_status()
    filings = resp.json()["filings"]["recent"]

    for form, report_date, accession, filing_date, primary_doc in zip(
        filings["form"],
        filings["reportDate"],
        filings["accessionNumber"],
        filings["filingDate"],
        filings["primaryDocument"],
    ):
        if form == "10-Q" and report_date == period_end_date:
            return accession, filing_date, primary_doc

    raise ValueError(
        f"No 10-Q found for CIK {cik} with period_end_date={period_end_date}"
    )


def build_document_url(cik: str, accession: str, primary_doc: str) -> str:
    """Construct the EDGAR archives URL for a filing's primary document."""
    acc_nodash = accession.replace("-", "")
    return f"{EDGAR_ARCHIVES}/{int(cik)}/{acc_nodash}/{primary_doc}"


async def download_filing(
    url: str,
    dest: Path,
    client: httpx.AsyncClient,
) -> Path:
    """Stream-download a filing to dest, return the path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with client.stream("GET", url, headers=HEADERS) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            async for chunk in resp.aiter_bytes(chunk_size=65536):
                f.write(chunk)
    await asyncio.sleep(0.15)  # stay well under EDGAR's 10 req/s guideline
    return dest
