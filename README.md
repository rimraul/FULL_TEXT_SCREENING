# FULL_TEXT_SCREENING

Full-text eligibility screening for systematic and scoping literature reviews:
PDF parsing (Docling), criterion-by-criterion LLM adjudication with page-level
evidence spans, aggregation into an auditable eligibility decision, and an
evaluation harness against a frozen gold evidence set.

Part of the toolkit accompanying the chapter on automating systematic
literature reviews (see the `slr-automation` repository).

> Python package name is `ft_screening` (import `ft_screening`); the repository
> is named FULL_TEXT_SCREENING.

## Install

```bash
git clone https://github.com/<owner>/FULL_TEXT_SCREENING.git
cd FULL_TEXT_SCREENING
pip install -e ".[dev]"
pytest            # fast, offline tests (slow/real-PDF/real-LLM tests are opt-in)
```

Using a local clone as a working directory is the intended workflow. You can
edit the code directly, or let a coding agent do it (see `AGENTS.md`).

## Contents

| Path | What |
|---|---|
| `src/ft_screening/` | the library |
| `tests/` | unit, contract and acceptance tests; synthetic PDF fixtures |
| `gold_set/` | frozen gold set: annotations, criteria, protocol hashes (short quotes from the source papers, for evaluation only; no full texts) |
| `outputs/` | research artifacts kept for transparency/reproducibility (checkpoints, results, incident note) |
| `scripts/` | gold-set construction workflow and evaluation runners |

## Real LLM calls

Nothing calls a paid API by default. Real calls need `OPENAI_API_KEY` **and**
`FT_SCREENING_RUN_REAL_LLM_TEST=1`. Never commit keys.

## Licence

Code: MIT (`LICENSE`). Third-party dependencies keep their own licences; note
in particular that **PyMuPDF is AGPL-3.0** (see `THIRD_PARTY.md`).
