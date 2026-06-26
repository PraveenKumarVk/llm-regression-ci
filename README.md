# Financial Earnings Q&A — Regression Eval Harness

## Why I built this

AI teams ship prompt changes by editing a string and eyeballing a few outputs. There's no
systematic way to know if quality degraded across the full distribution of inputs.

I built this after observing that financial Q&A systems have three failure modes that are
easy to miss without structured testing:

1. **Temporal confusion** — models confuse Q3 2024 with Q3 2023 when both figures appear
   in the same chunk. Humans spot this on easy cases but miss it on complex ones.

2. **Silent hallucination** — models answer confidently when they should refuse. A
   `NOT_IN_DOCUMENT` failure is more dangerous than a wrong answer because it's invisible
   to users.

3. **Slow drift** — prompts that work well degrade gradually over repeated model API
   updates. No single run looks bad but cumulative degradation is significant.

The system catches all three. It runs in CI, costs ~$0.12 per full eval run, and has
blocked 4 prompt changes that would have degraded production quality in testing.

## What this does

Runs on every PR touching `/prompts`. Detects when prompt or model changes
degrade answer quality on our golden dataset of 75 financial Q&A test cases.
Posts a result summary to the PR and blocks merge on critical regressions.

## Severity levels

- **Clean**: overall pass rate dropped less than 3 pp vs baseline, and fewer than 2 regressions
- **Warning**: dropped 3–8 pp, OR 2+ regressions on any difficulty → PR comment posted
- **Critical**: dropped more than 8 pp, OR 3+ regressions on hard/adversarial cases → merge blocked

Severity is computed in `src/diff_runner.py:_severity()`. Both conditions are OR'd — a run
with a small overall delta can still be critical if it broke 3 hard cases.

## Adding test cases to the golden dataset

Test cases live in `data/golden_dataset_v*.json`. To add cases:

1. Find a relevant chunk in `data/chunks/<TICKER_10Q_PERIOD>/` that contains the answer
2. Write the case by hand in the JSON — **do not generate cases with an LLM**
3. Run the dataset integrity suite to catch mistakes before committing:
   ```
   pytest tests/test_golden_loader.py -v -p no:deepeval
   ```
4. Bump the version in the filename (`v1.0.0` → `v1.1.0`). The loader picks up the
   highest semver automatically — no config file to update.
5. If regenerating from scratch: `python scripts/generate_golden_dataset.py`

Edge cases and adversarial examples are more valuable than easy ones.
Aim for 30%+ of cases in hard/adversarial difficulty.

## Running locally

```bash
# Install
pip install -e .

# Run all tests (deepeval plugin causes slow teardown — always suppress it)
pytest -p no:deepeval

# Run a single eval against the golden dataset
python scripts/run_eval_ci.py \
  --dataset data/golden_dataset_v1.0.0.json \
  --output-dir .eval_results/ \
  --history-dir .eval_history/

# Rebuild the corpus (one ticker at a time)
python scripts/build_corpus.py --tickers AAPL
```

## Architecture decisions

**Why temperature=0.0 for the feature under test:**
Regression testing requires deterministic outputs. With temperature > 0 you cannot
distinguish a real regression from output variance.

**Why Claude Haiku for the faithfulness judge:**
The judge runs on every test case on every eval run. Using GPT-4o as judge would cost
~$2 per run vs ~$0.08 with Haiku. Haiku's faithfulness scoring correlates >0.85 with
GPT-4o on our calibration set.

**Why `refusal_correctness` is a hard gate:**
A system that answers when it should refuse is more dangerous than one that gives a
slightly imprecise answer. We treat `NOT_IN_DOCUMENT` failures as blocking regardless
of composite score — the composite can be 0.90 and the case still fails.

**Why we track slow drift separately:**
A prompt that degrades 0.5 pp per run looks fine run-to-run but after 10 runs represents
5 pp quality loss. The 7-run moving average catches this. The window size is the
`window_size` parameter in `detect_slow_drift()` in `src/drift_detector.py`.

**Why dataset version is independent of prompt version:**
`RunSummary.dataset_version` is stored separately from `prompt_version`. Silently
removing hard cases from the dataset to improve scores is a regression in the eval
bar — tracked the same way as a prompt regression.

**Why `scorer_notes` skips dimensions rather than zeroing them:**
Some test cases are structurally incompatible with a scorer dimension (e.g., an
adversarial case where the year is implied by filing context — `temporal_precision`
can't fairly score it). Zeroing would drag down the pass rate unfairly; skipping
re-weights the composite over the active dimensions only.
