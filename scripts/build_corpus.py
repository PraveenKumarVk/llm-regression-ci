"""
Download 10-Q filings from SEC EDGAR, extract text, chunk by section,
and write the chunks to data/chunks/<document_id>/.

Usage:
    python scripts/build_corpus.py
    python scripts/build_corpus.py --tickers AAPL JPM   # subset
    python scripts/build_corpus.py --raw-dir /tmp/raw   # custom paths
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import httpx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.corpus.chunker import chunk_document
from src.corpus.edgar import FILING_TARGETS, build_document_url, download_filing, find_filing, get_cik
from src.corpus.extractor import extract_text


async def process_target(
    target: dict,
    raw_dir: Path,
    chunks_dir: Path,
    client: httpx.AsyncClient,
) -> int:
    ticker = target["ticker"]
    document_id = target["document_id"]
    click.echo(f"[{ticker}] resolving CIK …")
    cik = await get_cik(ticker, client)

    click.echo(f"[{ticker}] finding 10-Q for period_end={target['period_end_date']} …")
    accession, filing_date, primary_doc = await find_filing(cik, target["period_end_date"], client)

    doc_url = build_document_url(cik, accession, primary_doc)
    suffix = primary_doc.rsplit(".", 1)[-1]
    dest = raw_dir / f"{document_id}.{suffix}"
    if dest.exists():
        click.echo(f"[{ticker}] already downloaded, skipping fetch")
    else:
        click.echo(f"[{ticker}] downloading from {doc_url} …")
        await download_filing(doc_url, dest, client)

    click.echo(f"[{ticker}] extracting text …")
    text = extract_text(dest)

    click.echo(f"[{ticker}] chunking …")
    chunks = chunk_document(
        text=text,
        document_id=document_id,
        ticker=ticker,
        company_name=target["company_name"],
        period=target["period"],
        period_end_date=target["period_end_date"],
        filing_date=filing_date,
        source_url=doc_url,
    )

    out_dir = chunks_dir / document_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        (out_dir / f"{chunk.chunk_id}.json").write_text(
            json.dumps(chunk.model_dump(), indent=2)
        )

    click.echo(f"[{ticker}] wrote {len(chunks)} chunks → {out_dir}")
    return len(chunks)


@click.command()
@click.option("--tickers", multiple=True, help="Limit to these tickers (default: all)")
@click.option("--raw-dir", default="data/raw", show_default=True, help="Directory for raw filings")
@click.option("--chunks-dir", default="data/chunks", show_default=True, help="Directory for chunk JSON files")
def main(tickers: tuple[str, ...], raw_dir: str, chunks_dir: str) -> None:
    targets = (
        [t for t in FILING_TARGETS if t["ticker"] in tickers]
        if tickers
        else FILING_TARGETS
    )
    if not targets:
        click.echo(f"No targets matched tickers: {tickers}", err=True)
        sys.exit(1)

    async def _run() -> None:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            total = 0
            for target in targets:
                try:
                    n = await process_target(
                        target,
                        Path(raw_dir),
                        Path(chunks_dir),
                        client,
                    )
                    total += n
                except Exception as exc:
                    click.echo(f"[{target['ticker']}] ERROR: {exc}", err=True)
            click.echo(f"\nDone. {total} total chunks written.")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
