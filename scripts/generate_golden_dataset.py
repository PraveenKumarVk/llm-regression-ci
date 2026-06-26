"""
Generate data/golden_dataset_v{version}.json from verified SEC 10-Q corpus chunks.

Every context_chunk is pulled verbatim from an actual chunk file.
Run: python scripts/generate_golden_dataset.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.golden_loader import save_dataset  # noqa: E402 — after sys.path fixup

CHUNKS_DIR = ROOT / "data" / "chunks"
DATASET_VERSION = "v1.0.0"
NOW = datetime(2026, 6, 25, tzinfo=timezone.utc).isoformat()


def chunk_text(document_id: str, chunk_id: str) -> str:
    p = CHUNKS_DIR / document_id / f"{chunk_id}.json"
    return json.loads(p.read_text())["text"]


# ---------------------------------------------------------------------------
# Load all source chunks once
# ---------------------------------------------------------------------------
AAPL_FS000  = chunk_text("AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_fs_000")
AAPL_MDA006 = chunk_text("AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_mda_006")
AAPL_MDA007 = chunk_text("AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_mda_007")
AAPL_MDA008 = chunk_text("AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_mda_008")

JPM_MDA001  = chunk_text("JPM_10Q_Q2_2024",  "JPM_10Q_Q2_2024_mda_001")
JPM_MDA003  = chunk_text("JPM_10Q_Q2_2024",  "JPM_10Q_Q2_2024_mda_003")
JPM_MDA009  = chunk_text("JPM_10Q_Q2_2024",  "JPM_10Q_Q2_2024_mda_009")
JPM_MDA010  = chunk_text("JPM_10Q_Q2_2024",  "JPM_10Q_Q2_2024_mda_010")
JPM_MDA013  = chunk_text("JPM_10Q_Q2_2024",  "JPM_10Q_Q2_2024_mda_013")

JNJ_MDA000  = chunk_text("JNJ_10Q_Q3_2024",  "JNJ_10Q_Q3_2024_mda_000")
JNJ_MDA001  = chunk_text("JNJ_10Q_Q3_2024",  "JNJ_10Q_Q3_2024_mda_001")
JNJ_MDA012  = chunk_text("JNJ_10Q_Q3_2024",  "JNJ_10Q_Q3_2024_mda_012")
JNJ_MDA013  = chunk_text("JNJ_10Q_Q3_2024",  "JNJ_10Q_Q3_2024_mda_013")
JNJ_MDA014  = chunk_text("JNJ_10Q_Q3_2024",  "JNJ_10Q_Q3_2024_mda_014")
JNJ_FS029   = chunk_text("JNJ_10Q_Q3_2024",  "JNJ_10Q_Q3_2024_fs_029")

AMZN_MDA006 = chunk_text("AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_006")
AMZN_MDA010 = chunk_text("AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_010")
AMZN_MDA011 = chunk_text("AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_011")
AMZN_MDA012 = chunk_text("AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_012")
AMZN_MDA025 = chunk_text("AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_025")
AMZN_FS000  = chunk_text("AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_fs_000")

XOM_MDA006  = chunk_text("XOM_10Q_Q3_2024",  "XOM_10Q_Q3_2024_mda_006")
XOM_MDA008  = chunk_text("XOM_10Q_Q3_2024",  "XOM_10Q_Q3_2024_mda_008")
XOM_FS000   = chunk_text("XOM_10Q_Q3_2024",  "XOM_10Q_Q3_2024_fs_000")


def tc(
    id: str,
    question: str,
    context_chunk: str,
    document_id: str,
    chunk_id: str,
    time_period: str,
    expected_answer_contains: list[str],
    expected_citation_contains: str | None,
    expected_is_refusal: bool,
    difficulty: str,
    failure_mode_category: str,
    notes: str,
    known_tricky_aspect: str,
) -> dict:
    return {
        "id": id,
        "created_at": NOW,
        "dataset_version": DATASET_VERSION,
        "question": question,
        "context_chunk": context_chunk,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "time_period": time_period,
        "expected_answer_contains": expected_answer_contains,
        "expected_citation_contains": expected_citation_contains,
        "expected_is_refusal": expected_is_refusal,
        "difficulty": difficulty,
        "failure_mode_category": failure_mode_category,
        "notes": notes,
        "known_tricky_aspect": known_tricky_aspect,
    }


cases = [

    # =========================================================================
    # NUMERICAL EXTRACTION (20 cases)
    # =========================================================================

    tc("TC_001",
       "What were Apple's total net sales for Q3 2024?",
       AAPL_FS000, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_fs_000", "Q3 2024",
       ["85,777"],
       "Total net sales",
       False, "easy", "numerical_extraction",
       "Baseline extraction. Q3 2023 figure of 81,797 appears in same table row.",
       "Prior-year comparison figure in same row — model may return 81,797."),

    tc("TC_002",
       "What were Apple's Services net sales for the three months ended June 29, 2024?",
       AAPL_FS000, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_fs_000", "Q3 2024",
       ["24,213"],
       "Services",
       False, "medium", "numerical_extraction",
       "Services appears three times: Q3'24 (24,213), Q3'23 (21,213), 9M'24 (71,197).",
       "Nine-month Services figure 71,197 dwarfs the quarterly one — model may grab it."),

    tc("TC_003",
       "What was Apple's net income for Q3 2024?",
       AAPL_FS000, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_fs_000", "Q3 2024",
       ["21,448"],
       "Net income",
       False, "easy", "numerical_extraction",
       "Four net income figures span two periods and two years.",
       "Nine-month 2024 figure 79,000 is more than 3× larger and could be grabbed instead."),

    tc("TC_004",
       "What was Apple's selling, general and administrative expense for Q3 2024?",
       AAPL_MDA007, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_mda_007", "Q3 2024",
       ["6,320"],
       "Selling, general and administrative",
       False, "easy", "numerical_extraction",
       "SGA for four periods appear: Q3'24 (6,320), Q3'23 (5,973), 9M'24 (19,574), 9M'23 (18,781).",
       "Nine-month figure 19,574 is 3× larger — a model anchoring on 'SGA' may return it."),

    tc("TC_005",
       "What was Apple's Services gross margin percentage for Q3 2024?",
       AAPL_MDA006, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_mda_006", "Q3 2024",
       ["74.0"],
       "Services",
       False, "medium", "numerical_extraction",
       "Four Services GM% values in one row across two periods and two years.",
       "Nine-month figure 73.8% is adjacent and looks similar — temporal confusion likely."),

    tc("TC_006",
       "What was JPMorgan Chase's net income for Q2 2024?",
       JPM_MDA001, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_001", "Q2 2024",
       ["18,149"],
       "Net income",
       False, "medium", "numerical_extraction",
       "Five quarterly net income figures plus two semi-annual totals in one row.",
       "Column order is reverse-chronological (Q2'24 first); six-month total 31,568 is a big distractor."),

    tc("TC_007",
       "What was JPMorgan Chase's diluted earnings per share for Q2 2024?",
       JPM_MDA001, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_001", "Q2 2024",
       ["6.12"],
       "Diluted",
       False, "medium", "numerical_extraction",
       "Seven EPS figures across five quarters and two semi-annual periods.",
       "Basic EPS is 6.13 (rounds to same) but diluted is 6.12 — model may conflate basic/diluted."),

    tc("TC_008",
       "What was JPMorgan's net interest income for the second quarter of 2024?",
       JPM_MDA009, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_009", "Q2 2024",
       ["22.7"],
       "Net interest income",
       False, "medium", "numerical_extraction",
       "NII (22.7B) and total net revenue (50.2B) both appear; also NII ex-Markets (22.9B).",
       "22.9B (NII ex-Markets) is stated immediately after 22.7B and could be confused with it."),

    tc("TC_009",
       "What were Johnson & Johnson's worldwide sales for the fiscal nine months of 2024?",
       JNJ_MDA000, "JNJ_10Q_Q3_2024", "JNJ_10Q_Q3_2024_mda_000", "Nine months 2024",
       ["66.3"],
       "worldwide sales were $66.3 billion",
       False, "easy", "numerical_extraction",
       "Prior-year nine-month figure 63.8B appears in same sentence.",
       "Growth percentage (4.0%, 5.6%) may distract; model could return 63.8B (prior year)."),

    tc("TC_010",
       "What were Johnson & Johnson's worldwide sales for the fiscal third quarter of 2024?",
       JNJ_MDA001, "JNJ_10Q_Q3_2024", "JNJ_10Q_Q3_2024_mda_001", "Q3 2024",
       ["22.5"],
       "worldwide sales were $22.5 billion",
       False, "easy", "numerical_extraction",
       "Q3 2024 (22.5B) vs Q3 2023 (21.4B) in same sentence.",
       "Prior-year figure 21.4B appears immediately after the 2024 figure."),

    tc("TC_011",
       "What was J&J's consolidated earnings before provision for taxes for the fiscal nine months of 2024?",
       JNJ_MDA012, "JNJ_10Q_Q3_2024", "JNJ_10Q_Q3_2024_mda_012", "Nine months 2024",
       ["12.8"],
       "12.8 billion",
       False, "medium", "numerical_extraction",
       "Four pre-tax figures: Q3'24 (3.3B), Q3'23 (5.2B), 9M'24 (12.8B), 9M'23 (10.2B).",
       "Q3 quarterly figure 3.3B is far smaller than nine-month; may be returned for 'nine months'."),

    tc("TC_012",
       "What were Amazon's AWS net sales for Q2 2024?",
       AMZN_MDA010, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_010", "Q2 2024",
       ["26,281"],
       "AWS",
       False, "medium", "numerical_extraction",
       "Six AWS figures: Q2'23 (22,140), Q2'24 (26,281), H1'23 (43,494), H1'24 (51,318) plus growth %.",
       "H1 2024 figure 51,318 is almost exactly double Q2 and looks like it could be annual."),

    tc("TC_013",
       "What were Amazon's North America net sales for Q2 2024?",
       AMZN_MDA010, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_010", "Q2 2024",
       ["90,033"],
       "North America",
       False, "easy", "numerical_extraction",
       "North America Q2'23 (82,546) vs Q2'24 (90,033) in same row.",
       "Prior-year 82,546 appears first in the row; model may return it as the 2024 figure."),

    tc("TC_014",
       "What was Amazon's AWS operating income for Q2 2024?",
       AMZN_MDA012, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_012", "Q2 2024",
       ["9,334"],
       "AWS",
       False, "medium", "numerical_extraction",
       "AWS operating income: Q2'23 (5,365), Q2'24 (9,334), H1'23 (10,488), H1'24 (18,755).",
       "H1 2024 figure 18,755 is 2× the Q2 figure and may be mistaken for a quarterly value."),

    tc("TC_015",
       "What were ExxonMobil's earnings for the third quarter of 2024?",
       XOM_MDA006, "XOM_10Q_Q3_2024", "XOM_10Q_Q3_2024_mda_006", "Q3 2024",
       ["8.6"],
       "third quarter 2024 earnings were $8.6 billion",
       False, "easy", "numerical_extraction",
       "Q3'24 (8.6B) vs Q3'23 (9.1B) in adjacent sentences; also 9M figures in next sentences.",
       "Nine-month 2024 earnings 26.1B appear two sentences later; also 9M'23 28.4B."),

    tc("TC_016",
       "What were ExxonMobil's capital and exploration expenditures for the first nine months of 2024?",
       XOM_MDA006, "XOM_10Q_Q3_2024", "XOM_10Q_Q3_2024_mda_006", "Nine months 2024",
       ["20.0"],
       "Capital and exploration expenditures were $20.0 billion",
       False, "medium", "numerical_extraction",
       "Two capex figures: Q3'24 (7.2B) and nine-month 2024 (20.0B). Also YoY changes (1.1B, 1.5B).",
       "Q3 capex 7.2B appears first and is the salient recent-quarter number."),

    tc("TC_017",
       "What were ExxonMobil's sales and other operating revenue for Q3 2024?",
       XOM_FS000, "XOM_10Q_Q3_2024", "XOM_10Q_Q3_2024_fs_000", "Q3 2024",
       ["87,792"],
       "Sales and other operating revenue",
       False, "medium", "numerical_extraction",
       "Sales (87,792) vs total revenues incl. equity affiliates (90,016) in same table.",
       "Total revenues (90,016) is 2.5B larger and also Q3 2024; model may return it as 'revenue'."),

    tc("TC_018",
       "What was ExxonMobil's US crude oil production in Q3 2024 in thousands of barrels per day?",
       XOM_MDA008, "XOM_10Q_Q3_2024", "XOM_10Q_Q3_2024_mda_008", "Q3 2024",
       ["1,444"],
       "United States",
       False, "medium", "numerical_extraction",
       "US crude: Q3'24 (1,444), Q3'23 (756), 9M'24 (1,174), 9M'23 (787).",
       "Q3'23 (756) is nearly half of Q3'24 due to Pioneer acquisition — large YoY step makes prior-year figure seem implausibly small."),

    tc("TC_019",
       "What was JPMorgan Chase's total allowance for credit losses at June 30, 2024?",
       JPM_MDA010, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_010", "Q2 2024",
       ["25.5"],
       "allowance for credit losses",
       False, "easy", "numerical_extraction",
       "Clear statement: 'total allowance for credit losses was $25.5 billion at June 30, 2024'.",
       "1.81% coverage ratio and 8.4B nonperforming assets are nearby credit metrics that could be confused."),

    tc("TC_020",
       "What was Amazon's consolidated net sales for Q2 2024?",
       AMZN_MDA010, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_010", "Q2 2024",
       ["147,977"],
       "Consolidated",
       False, "easy", "numerical_extraction",
       "Consolidated Q2'23 (134,383), Q2'24 (147,977), H1'23 (261,741), H1'24 (291,290).",
       "H1 2024 figure 291,290 is roughly double the quarterly figure; prior-year Q2 134,383 is close."),

    # =========================================================================
    # TEMPORAL PRECISION (15 cases)
    # =========================================================================

    tc("TC_021",
       "What was Apple's net income for the nine months ended June 29, 2024?",
       AAPL_FS000, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_fs_000", "Nine months 2024",
       ["79,000"],
       "Net income",
       False, "medium", "temporal_precision",
       "Four net income figures in one row: Q3'24 (21,448), Q3'23 (19,881), 9M'24 (79,000), 9M'23 (74,039).",
       "Q3 quarterly net income (21,448) is the most salient single-period figure — model defaults to it."),

    tc("TC_022",
       "What was Apple's Services gross margin percentage for the nine months ended June 29, 2024?",
       AAPL_MDA006, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_mda_006", "Nine months 2024",
       ["73.8"],
       "Services",
       False, "hard", "temporal_precision",
       "Four Services GM% values: Q3'24 (74.0%), Q3'23 (70.5%), 9M'24 (73.8%), 9M'23 (70.8%). Column headers not in this chunk.",
       "74.0% (Q3 quarterly) is the more prominent recent figure — the nine-month answer 73.8% is easily confused with it."),

    tc("TC_023",
       "What was JPMorgan Chase's diluted EPS for Q1 2024?",
       JPM_MDA001, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_001", "Q1 2024",
       ["4.44"],
       "Diluted",
       False, "hard", "temporal_precision",
       "Seven EPS values in one row ordered Q2'24, Q1'24, Q4'23, Q3'23, Q2'23, H1'24, H1'23.",
       "Q2 2024 diluted EPS (6.12) is the most recent and prominent — Q1 2024 (4.44) is the second column."),

    tc("TC_024",
       "What was JPMorgan Chase's net income for Q2 2023?",
       JPM_MDA001, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_001", "Q2 2023",
       ["14,472"],
       "Net income",
       False, "hard", "temporal_precision",
       "Five quarterly net incomes plus two semi-annual. Q2 2023 (14,472) is the last quarterly column.",
       "Q2 2024 net income (18,149) is the first and most prominent; Q1 2024 (13,419) is close in value to the correct answer."),

    tc("TC_025",
       "What was JPMorgan Chase's net income for Q4 2023?",
       JPM_MDA001, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_001", "Q4 2023",
       ["9,307"],
       "Net income",
       False, "hard", "temporal_precision",
       "Q4 2023 (9,307) is distinctly lower than surrounding quarters — includes large one-time charges.",
       "The value 9,307 stands out as anomalously low; model may assume it's wrong and return an adjacent value."),

    tc("TC_026",
       "What were J&J's worldwide sales for the fiscal third quarter of 2023?",
       JNJ_MDA001, "JNJ_10Q_Q3_2024", "JNJ_10Q_Q3_2024_mda_001", "Q3 2023",
       ["21.4"],
       "2023 fiscal third quarter sales of $21.4 billion",
       False, "medium", "temporal_precision",
       "Q3 2024 (22.5B) and Q3 2023 (21.4B) in the same sentence; nine-month figures in the prior chunk.",
       "Q3 2024 appears first (22.5B); Q3 2023 (21.4B) appears as a comparison at the end."),

    tc("TC_027",
       "What was J&J's consolidated earnings before provision for taxes for the fiscal nine months of 2023?",
       JNJ_MDA012, "JNJ_10Q_Q3_2024", "JNJ_10Q_Q3_2024_mda_012", "Nine months 2023",
       ["10.2"],
       "10.2 billion in the fiscal nine months of 2023",
       False, "hard", "temporal_precision",
       "Four pre-tax figures. The 2023 nine-month value (10.2B) is lower than the 2024 nine-month (12.8B), reversing the Q3 trend.",
       "Q3 2023 pre-tax (5.2B) is higher than Q3 2024 (3.3B) — the nine-month trend is opposite, confusing."),

    tc("TC_028",
       "What were Amazon's AWS net sales for the six months ended June 30, 2024?",
       AMZN_MDA010, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_010", "H1 2024",
       ["51,318"],
       "AWS",
       False, "medium", "temporal_precision",
       "AWS Q2'24 (26,281) and H1'24 (51,318) both appear; H1'23 (43,494) is also present.",
       "Q2 2024 standalone (26,281) is the natural answer to 'AWS sales Q2 2024' — this asks for the six-month total."),

    tc("TC_029",
       "What was Amazon's AWS operating income for Q2 2023?",
       AMZN_MDA012, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_012", "Q2 2023",
       ["5,365"],
       "AWS",
       False, "hard", "temporal_precision",
       "Four AWS operating income figures across two periods and two years.",
       "Q2 2024 AWS operating income (9,334) is far larger and more prominent — prior year (5,365) is 42% less."),

    tc("TC_030",
       "What were Amazon's consolidated net sales for the six months ended June 30, 2023?",
       AMZN_MDA010, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_010", "H1 2023",
       ["261,741"],
       "Consolidated",
       False, "medium", "temporal_precision",
       "Consolidated: Q2'23 (134,383), Q2'24 (147,977), H1'23 (261,741), H1'24 (291,290).",
       "H1 2024 (291,290) is the largest figure and may be returned regardless of year asked."),

    tc("TC_031",
       "What were ExxonMobil's earnings for the third quarter of 2023?",
       XOM_MDA006, "XOM_10Q_Q3_2024", "XOM_10Q_Q3_2024_mda_006", "Q3 2023",
       ["9.1"],
       "compared to $9.1 billion a year earlier",
       False, "medium", "temporal_precision",
       "Q3 2024 (8.6B) appears first; Q3 2023 (9.1B) is the comparison value.",
       "Model anchors to the first earnings figure mentioned (8.6B = Q3 2024); prior-year value is secondary."),

    tc("TC_032",
       "What were ExxonMobil's earnings for the nine months ended September 30, 2023?",
       XOM_MDA006, "XOM_10Q_Q3_2024", "XOM_10Q_Q3_2024_mda_006", "Nine months 2023",
       ["28.4"],
       "compared to $28.4 billion a year earlier",
       False, "hard", "temporal_precision",
       "Nine-month 2024 (26.1B) vs nine-month 2023 (28.4B). Also Q3 figures in surrounding sentences.",
       "Nine-month 2024 (26.1B) is in the same sentence; model may return it instead of the prior-year 28.4B."),

    tc("TC_033",
       "What was ExxonMobil's US crude oil and NGL production in Q3 2023 in thousands of barrels per day?",
       XOM_MDA008, "XOM_10Q_Q3_2024", "XOM_10Q_Q3_2024_mda_008", "Q3 2023",
       ["756"],
       "United States",
       False, "hard", "temporal_precision",
       "US crude Q3'24 (1,444), Q3'23 (756), 9M'24 (1,174), 9M'23 (787).",
       "Q3 2024 (1,444) is nearly double Q3 2023 (756) due to Pioneer acquisition — the prior-year figure looks implausibly small."),

    tc("TC_034",
       "What was Apple's Products net sales for the three months ended July 1, 2023?",
       AAPL_FS000, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_fs_000", "Q3 2023",
       ["60,584"],
       "Products",
       False, "medium", "temporal_precision",
       "Products: Q3'24 (61,564) and Q3'23 (60,584) plus nine-month figures in same table.",
       "Q3 2024 figure (61,564) appears before the Q3 2023 figure (60,584); values are close."),

    tc("TC_035",
       "What was JPMorgan Chase's net income for Q3 2023?",
       JPM_MDA001, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_001", "Q3 2023",
       ["13,151"],
       "Net income",
       False, "hard", "temporal_precision",
       "Five quarterly net incomes. Q3 2023 (13,151) is the fourth column in reverse-chronological order.",
       "Q3 2023 (13,151) is sandwiched between Q4 2023 (9,307) and Q2 2023 (14,472) — easy to mis-anchor."),

    # =========================================================================
    # REFUSAL CORRECTNESS (15 cases)
    # =========================================================================

    tc("TC_036",
       "What was Apple's free cash flow for Q3 2024?",
       AAPL_MDA008, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_mda_008", "Q3 2024",
       [], None, True,
       "medium", "refusal_correctness",
       "Chunk mentions large cash figures ($38.4B obligations, $26.0B buybacks) but no operating cash flow or capex needed to compute FCF.",
       "Plausible cash-related figures abound; model may hallucinate FCF from buyback/obligation figures."),

    tc("TC_037",
       "How many iPhones did Apple sell in Q3 2024?",
       AAPL_FS000, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_fs_000", "Q3 2024",
       [], None, True,
       "easy", "refusal_correctness",
       "The income statement reports iPhone revenue in dollars only. Unit sales are not disclosed.",
       "iPhone revenue ($39.3B, found in mda_004) might tempt model to convert to unit estimate."),

    tc("TC_038",
       "What was Apple's total gross profit for the nine months ended June 29, 2024?",
       AAPL_MDA007, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_mda_007", "Nine months 2024",
       [], None, True,
       "hard", "refusal_correctness",
       "mda_007 contains operating expenses and SGA but NOT the gross margin figures (those are in mda_006).",
       "SGA and R&D figures present; model may attempt to reconstruct gross profit from partial data."),

    tc("TC_039",
       "What was JPMorgan Chase's net charge-off rate for Q2 2024?",
       JPM_MDA010, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_010", "Q2 2024",
       [], None, True,
       "hard", "refusal_correctness",
       "Chunk discusses credit quality (allowance 25.5B, NPA 8.4B, coverage ratio 1.81%) but not the net charge-off rate.",
       "1.81% coverage ratio looks like a charge-off rate; model may cite it as the answer."),

    tc("TC_040",
       "What was JPMorgan Chase's adjusted expense for full-year 2024?",
       JPM_MDA013, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_013", "FY 2024",
       [], None, True,
       "medium", "refusal_correctness",
       "mda_013 contains context around the 2024 outlook but the specific ~$92 billion guidance figure is in a later chunk.",
       "Model may hallucinate the $92B figure from memory of JPM disclosures or from the $91B NII figure in the same filing."),

    tc("TC_041",
       "What was J&J's MedTech segment worldwide sales for Q3 2024?",
       JNJ_MDA001, "JNJ_10Q_Q3_2024", "JNJ_10Q_Q3_2024_mda_001", "Q3 2024",
       [], None, True,
       "medium", "refusal_correctness",
       "mda_001 gives only total worldwide sales (22.5B); segment breakdown (Innovative Medicine vs MedTech) is not in this chunk.",
       "Total Q3 sales (22.5B) is present — model may assign it to MedTech or invent a segment split."),

    tc("TC_042",
       "What were J&J's R&D expenses for Q3 2024?",
       JNJ_MDA013, "JNJ_10Q_Q3_2024", "JNJ_10Q_Q3_2024_mda_013", "Q3 2024",
       [], None, True,
       "hard", "refusal_correctness",
       "mda_013 discusses R&D expense qualitatively (phasing, MedTech drivers) but the dollar amounts are in a different chunk.",
       "Yellow Jersey acquisition cost ($1.25B) appears and could be mistaken for Q3 total R&D."),

    tc("TC_043",
       "What was Amazon's operating income for Q3 2024?",
       AMZN_MDA025, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_025", "Q3 2024",
       [], None, True,
       "medium", "refusal_correctness",
       "mda_025 contains Q3 2024 guidance ($11.5-15.0B range), not actual Q3 2024 results (this is a Q2 2024 filing).",
       "The guidance range $11.5-15.0B looks like an actual result; 'compared with $11.2 billion in Q3 2023' adds apparent authority."),

    tc("TC_044",
       "What was Amazon's AWS gross margin for Q2 2024?",
       AMZN_MDA012, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_012", "Q2 2024",
       [], None, True,
       "hard", "refusal_correctness",
       "mda_012 reports AWS operating income (9,334M) but not AWS gross profit or gross margin percentage.",
       "AWS operating income is prominent; model may confuse it with gross margin or attempt to divide by revenue from a different chunk."),

    tc("TC_045",
       "What was Amazon's net sales guidance growth rate for Q3 2024?",
       AMZN_MDA025, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_025", "Q3 2024",
       [], None, True,
       "adversarial", "refusal_correctness",
       "The chunk provides a range: '8% and 11% compared with third quarter 2023'. This is guidance not a single reported figure; the question asks for 'the growth rate' (singular) which doesn't exist.",
       "A range (8%-11%) is stated; model may return one endpoint or compute midpoint (9.5%) as a definitive answer."),

    tc("TC_046",
       "What was ExxonMobil's Energy Products gross margin for Q3 2024?",
       XOM_MDA006, "XOM_10Q_Q3_2024", "XOM_10Q_Q3_2024_mda_006", "Q3 2024",
       [], None, True,
       "hard", "refusal_correctness",
       "mda_006 has total company earnings (8.6B) but not Energy Products segment gross margin.",
       "Total company earnings may be cited as proxy for gross margin; or model may hallucinate a segment figure."),

    tc("TC_047",
       "How many barrels of natural gas did ExxonMobil produce in Q3 2024?",
       XOM_MDA008, "XOM_10Q_Q3_2024", "XOM_10Q_Q3_2024_mda_008", "Q3 2024",
       [], None, True,
       "medium", "refusal_correctness",
       "mda_008 shows crude oil production (barrels). Natural gas is measured in cubic feet, not barrels, and the gas figure is not in this chunk.",
       "Crude oil production (1,444 kbpd) is present; model may return it as the answer to a gas production question."),

    tc("TC_048",
       "What was JPMorgan's noninterest expense for Q2 2023?",
       JPM_MDA009, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_009", "Q2 2023",
       [], None, True,
       "hard", "refusal_correctness",
       "mda_009 states Q2 2024 noninterest expense (23.7B, up 14%) but not the Q2 2023 absolute figure.",
       "The 14% increase figure implies the prior year but doesn't state it; model may calculate 23.7 / 1.14 = 20.8B from the partial information."),

    tc("TC_049",
       "What was Apple's cash balance at the end of Q3 2024?",
       AAPL_MDA008, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_mda_008", "Q3 2024",
       [], None, True,
       "medium", "refusal_correctness",
       "mda_008 mentions $38.4B in purchase obligations and $26.0B in buybacks but not the cash balance.",
       "Large dollar figures ($38.4B, $38.3B, $26.0B) look like balance sheet figures; model may cite one as cash."),

    tc("TC_050",
       "What was J&J's interest income for the fiscal third quarter of 2024?",
       JNJ_MDA014, "JNJ_10Q_Q3_2024", "JNJ_10Q_Q3_2024_mda_014", "Q3 2024",
       [], None, True,
       "hard", "refusal_correctness",
       "mda_014 states nine-month 2024 interest income ($433M) and nine-month 2023 ($277M), but not the Q3 quarterly figure.",
       "Nine-month figure $433M is explicit; model may return it as the quarterly answer."),

    # =========================================================================
    # MULTI-HOP CALCULATION (15 cases)
    # =========================================================================

    tc("TC_051",
       "By how many dollars did Apple's selling, general and administrative expense increase in Q3 2024 compared to Q3 2023?",
       AAPL_MDA007, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_mda_007", "Q3 2024",
       ["347"],
       "increased $347 million during the third quarter of 2024",
       False, "easy", "multi_hop_calculation",
       "The dollar increase is stated directly ($347M for Q3, $793M for nine months).",
       "Nine-month increase ($793M) appears in the same sentence and is more than double — easy to grab the wrong figure."),

    tc("TC_052",
       "What was Apple's Products gross profit for Q3 2024?",
       AAPL_FS000, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_fs_000", "Q3 2024",
       ["21,761"],
       "Products",
       False, "hard", "multi_hop_calculation",
       "Products revenue (61,564) and Products cost of sales (39,803) are both in the table. 61,564 - 39,803 = 21,761.",
       "Gross margin total (39,678 including Services) is stated explicitly; model may return it instead of computing Products-only gross profit."),

    tc("TC_053",
       "By how many dollars did Apple's total net sales increase from Q3 2023 to Q3 2024?",
       AAPL_FS000, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_fs_000", "Q3 2024",
       ["3,980"],
       "Total net sales",
       False, "medium", "multi_hop_calculation",
       "85,777 - 81,797 = 3,980. Both figures appear in the same row.",
       "The percentage change is not stated here; model must calculate. Nine-month revenues nearby may distract."),

    tc("TC_054",
       "What was JPMorgan Chase's six-month net income increase from H1 2023 to H1 2024?",
       JPM_MDA001, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_001", "H1 2024",
       ["4,474"],
       "Net income",
       False, "hard", "multi_hop_calculation",
       "31,568 (H1 2024) - 27,094 (H1 2023) = 4,474. Both appear in the semi-annual columns.",
       "Five quarterly figures are more prominent than the semi-annual columns; model may calculate a quarterly difference instead."),

    tc("TC_055",
       "By how many dollars did J&J's consolidated pre-tax earnings improve from the fiscal nine months of 2023 to the fiscal nine months of 2024?",
       JNJ_MDA012, "JNJ_10Q_Q3_2024", "JNJ_10Q_Q3_2024_mda_012", "Nine months 2024",
       ["2.6"],
       "12.8 billion",
       False, "hard", "multi_hop_calculation",
       "12.8B (9M'24) - 10.2B (9M'23) = 2.6B improvement. But Q3 trend is opposite (3.3 vs 5.2 = decline).",
       "Q3 comparison shows a $1.9B decline — model may report a decline instead of the nine-month improvement."),

    tc("TC_056",
       "By how many dollars did Amazon's AWS operating income grow from Q2 2023 to Q2 2024?",
       AMZN_MDA012, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_012", "Q2 2024",
       ["3,969"],
       "AWS",
       False, "medium", "multi_hop_calculation",
       "9,334 - 5,365 = 3,969. H1 delta (18,755 - 10,488 = 8,267) is in same table.",
       "H1 growth of 8,267 is more than double the quarterly growth — model may return it."),

    tc("TC_057",
       "What were Amazon's total capital outlays (capex plus acquisitions) for the six months ended June 30, 2024?",
       AMZN_MDA006, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_006", "H1 2024",
       ["34.2", "34"],
       "30.3 billion",
       False, "hard", "multi_hop_calculation",
       "Capex H1 2024 (30.3B) + acquisitions H1 2024 (3.9B) = 34.2B. Both figures in same paragraph.",
       "30.3B is the explicit 'cash capital expenditures' figure — most models stop there and ignore the 3.9B acquisitions."),

    tc("TC_058",
       "By how many dollars did Amazon's International segment operating result improve from Q2 2023 to Q2 2024?",
       AMZN_MDA012, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_012", "Q2 2024",
       ["1,168"],
       "International",
       False, "hard", "multi_hop_calculation",
       "273 - (-895) = 1,168. Requires recognizing the Q2 2023 figure is a loss (-895).",
       "The sign change (loss to profit) makes the arithmetic non-obvious; model may subtract absolute values (273 - 895 = -622)."),

    tc("TC_059",
       "By how many dollars did ExxonMobil's Q3 earnings decline from 2023 to 2024?",
       XOM_MDA006, "XOM_10Q_Q3_2024", "XOM_10Q_Q3_2024_mda_006", "Q3 2024",
       ["0.5", "500"],
       "third quarter 2024 earnings were $8.6 billion, compared to $9.1 billion",
       False, "medium", "multi_hop_calculation",
       "9.1 - 8.6 = 0.5B. Nine-month decline (28.4 - 26.1 = 2.3B) is in adjacent sentences.",
       "Nine-month decline (2.3B) is 4.6× the quarterly decline (0.5B) and appears nearby — model may return it."),

    tc("TC_060",
       "What was ExxonMobil's net income margin (net income as a % of sales and other operating revenue) for Q3 2024?",
       XOM_FS000, "XOM_10Q_Q3_2024", "XOM_10Q_Q3_2024_fs_000", "Q3 2024",
       ["9.8", "9.7"],
       "Sales and other operating revenue",
       False, "hard", "multi_hop_calculation",
       "8,610 / 87,792 = 9.8%. But total revenues including equity affiliates = 90,016; 8,610/90,016 = 9.6%.",
       "Using total revenues (90,016) vs operating revenues (87,792) changes the answer — the distinction matters."),

    tc("TC_061",
       "By how many thousands of barrels per day did ExxonMobil's US crude production increase from Q3 2023 to Q3 2024?",
       XOM_MDA008, "XOM_10Q_Q3_2024", "XOM_10Q_Q3_2024_mda_008", "Q3 2024",
       ["688"],
       "United States",
       False, "hard", "multi_hop_calculation",
       "1,444 - 756 = 688 kbpd increase. Text notes this is driven by Pioneer acquisition.",
       "The near-doubling (88% increase) may seem implausible; model may question or adjust the figures."),

    tc("TC_062",
       "What was Apple's total operating expense for the nine months ended June 29, 2024?",
       AAPL_FS000, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_fs_000", "Nine months 2024",
       ["43,179"],
       "Total operating expenses",
       False, "medium", "multi_hop_calculation",
       "43,179 is directly stated. But the question could also mean total cost of sales + opex = 159,301 + 43,179 = 202,480.",
       "Ambiguity between 'operating expenses' (R&D + SGA = 43,179) vs total expenses including COGS — both are valid interpretations."),

    tc("TC_063",
       "What was JPMorgan's total net revenue increase in dollars from Q2 2023 to Q2 2024?",
       JPM_MDA009, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_009", "Q2 2024",
       ["8.5", "8,500"],
       " was $50.2 billion, up 22%",
       False, "hard", "multi_hop_calculation",
       "50.2B - 41.3B = 8.9B (mda_006 has the prior-year figure). This chunk only has 50.2B and the 22% growth rate.",
       "With only 50.2B and 22%, model must compute backward: 50.2/1.22 = 41.1B → delta = 9.1B. Or it may just state the percentage."),

    tc("TC_064",
       "What was Amazon's consolidated operating income for the six months ended June 30, 2024?",
       AMZN_MDA012, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_012", "H1 2024",
       ["29,979"],
       "Consolidated",
       False, "medium", "multi_hop_calculation",
       "29,979 is directly in the H1 2024 column. But Q2 2024 alone (14,672) is also prominent in the prose.",
       "The prose says 'increased from $12.5B to $30.0B' using rounded figures — exact table value 29,979 differs slightly."),

    tc("TC_065",
       "What was J&J's year-over-year change in worldwide sales for Q3 in absolute dollar terms?",
       JNJ_MDA001, "JNJ_10Q_Q3_2024", "JNJ_10Q_Q3_2024_mda_001", "Q3 2024",
       ["1.1"],
       "worldwide sales were $22.5 billion",
       False, "medium", "multi_hop_calculation",
       "22.5 - 21.4 = 1.1B. The percentage (5.2%) is stated; dollar change requires subtraction.",
       "Model may return 5.2% (stated) instead of computing the $1.1B absolute change."),

    # =========================================================================
    # ADVERSARIAL (10 cases)
    # =========================================================================

    tc("TC_066",
       "By how much did Apple's SG&A increase in the first nine months of 2024?",
       AAPL_MDA007, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_mda_007", "Nine months 2024",
       ["793"],
       "793 million during the first nine months of 2024",
       False, "adversarial", "temporal_precision",
       "Both Q3 ($347M) and nine-month ($793M) increases are in the same sentence. Question asks for nine months.",
       "Q3 figure $347M appears first in the sentence — question asks for nine months but model returns the Q3 figure."),

    tc("TC_067",
       "What were Apple's manufacturing purchase obligations payable within 12 months as of June 29, 2024?",
       AAPL_MDA008, "AAPL_10Q_Q3_2024", "AAPL_10Q_Q3_2024_mda_008", "Q3 2024",
       ["38.3"],
       "38.3 billion payable within 12 months",
       False, "adversarial", "numerical_extraction",
       "Total obligations ($38.4B) vs payable within 12 months ($38.3B). The question specifies 12-month subset.",
       "38.4B (total) appears first; 38.3B (12-month subset) differs by only $0.1B — model returns the total."),

    tc("TC_068",
       "What was JPMorgan Chase's reported net income for Q2 2024, and does this include the Visa gain?",
       JPM_MDA003, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_003", "Q2 2024",
       ["7.9", "Visa"],
       "7.9 billion net gain related to Visa shares",
       False, "adversarial", "faithfulness",
       "The chunk discloses that Q2 2024 net revenue included a $7.9B Visa gain. The reported net income (18.1B) includes it.",
       "Model may either (a) not mention the Visa gain when discussing net income, or (b) try to compute an ex-Visa figure without sufficient data."),

    tc("TC_069",
       "What is JPMorgan's expected full-year 2024 net interest income?",
       JPM_MDA013, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_013", "FY 2024",
       [], None, True,
       "adversarial", "refusal_correctness",
       "mda_013 provides forward-looking context but the specific $91B NII guidance figure is in a different chunk.",
       "Model may recall or hallucinate the $91B figure from JPM public disclosures rather than citing NOT_IN_DOCUMENT."),

    tc("TC_070",
       "Did J&J's pre-tax income improve or decline year-over-year for the fiscal nine months of 2024?",
       JNJ_MDA012, "JNJ_10Q_Q3_2024", "JNJ_10Q_Q3_2024_mda_012", "Nine months 2024",
       ["12.8", "10.2"],
       "12.8 billion",
       False, "adversarial", "temporal_precision",
       "Nine-month 2024 pre-tax (12.8B) > nine-month 2023 (10.2B) = improvement. But Q3 alone shows a decline (3.3 vs 5.2B).",
       "Q3 trend is a decline (3.3 < 5.2); nine-month trend is an improvement. Model reading Q3 first answers 'declined'."),

    tc("TC_071",
       "What was Amazon's Q3 2024 operating income guidance in this filing, and is this an actual reported result?",
       AMZN_MDA025, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_025", "Q3 2024 guidance",
       ["11.5", "15.0", "guidance", "expected"],
       "expected to be between $11.5 billion and $15.0 billion",
       False, "adversarial", "faithfulness",
       "Tests whether model correctly labels a range as forward-looking guidance vs an actual reported result.",
       "The range $11.5-15.0B sounds factual; model may present it as an actual Q3 2024 result. "
       "Intentionally hard: year '2024' is implied by filing context (this is a Q2 2024 10-Q), "
       "not stated verbatim in the chunk — chunk only prints '2023' as the comparison year."),

    tc("TC_072",
       "What were Amazon's total capital expenditures for Q2 2024?",
       AMZN_MDA006, "AMZN_10Q_Q2_2024", "AMZN_10Q_Q2_2024_mda_006", "Q2 2024",
       ["16.4"],
       "16.4 billion during Q2 2023 and Q2 2024",
       False, "adversarial", "temporal_precision",
       "Capex Q2'23 (10.4B) and Q2'24 (16.4B) appear in the same clause: 'during Q2 2023 and Q2 2024'. H1'24 capex (30.3B) is also present.",
       "H1 2024 capex (30.3B) ≈ 2× Q2 figure; model may return it. Also the 10.4B (Q2 2023) appears first."),

    tc("TC_073",
       "What was ExxonMobil's total revenues for Q3 2024?",
       XOM_FS000, "XOM_10Q_Q3_2024", "XOM_10Q_Q3_2024_fs_000", "Q3 2024",
       ["90,016"],
       "Total revenues and other income",
       False, "adversarial", "numerical_extraction",
       "Two revenue figures: 'Sales and other operating revenue' (87,792) vs 'Total revenues and other income' (90,016).",
       "'Revenue' is ambiguous — 87,792 (operating revenue) vs 90,016 (total including equity affiliates and other income)."),

    tc("TC_074",
       "What was JPMorgan's book value per share at Q2 2024?",
       JPM_MDA001, "JPM_10Q_Q2_2024", "JPM_10Q_Q2_2024_mda_001", "Q2 2024",
       ["111.29"],
       "Book value per share",
       False, "adversarial", "numerical_extraction",
       "Book value per share (111.29) vs Tangible book value per share TBVPS (92.77) at Q2 2024.",
       "TBVPS (92.77) is a commonly cited alternative metric; model may return it instead of GAAP book value."),

    tc("TC_075",
       "How much did ExxonMobil's Q3 2024 US crude production increase due to the Pioneer acquisition?",
       XOM_MDA008, "XOM_10Q_Q3_2024", "XOM_10Q_Q3_2024_mda_008", "Q3 2024",
       [], None, True,
       "adversarial", "refusal_correctness",
       "The chunk mentions Pioneer drove higher production but does not isolate the Pioneer-specific contribution vs heritage Permian growth.",
       "Total US crude increase (688 kbpd) is computable; model may attribute all of it to Pioneer, which overstates Pioneer's standalone impact."),
]

from src.models import GoldenDataset, GoldenTestCase as _GTC  # noqa: E402

dataset = GoldenDataset.model_validate({
    "version": DATASET_VERSION,
    "created_at": NOW,
    "cases": cases,
})

out_path = save_dataset(dataset)
print(f"Wrote {len(cases)} test cases to {out_path}")

# Verify distribution
from collections import Counter
cats = Counter(c["failure_mode_category"] for c in cases)
diffs = Counter(c["difficulty"] for c in cases)
refusals = sum(1 for c in cases if c["expected_is_refusal"])
print("\nBy category:", dict(cats))
print("By difficulty:", dict(diffs))
print(f"Refusal cases: {refusals}")
