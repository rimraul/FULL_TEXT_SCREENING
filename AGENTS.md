# Notes for coding agents

This repository is meant to be cloned and used as a local working directory.
Personalize freely: edit criteria, prompts, adjudication components, scripts.

- Install: `pip install -e ".[dev]"` (Python >= 3.11).
- Test: `pytest` runs the fast offline suite. `-m slow` adds real Docling
  conversions; `real_pdf` needs `FT_SCREENING_REAL_PDF_DIR`; `real_llm` needs
  `OPENAI_API_KEY` and `FT_SCREENING_RUN_REAL_LLM_TEST=1` and spends money.
- Never print, log or commit API keys. Read them from environment variables.
- Do not modify `gold_set/frozen/` or `outputs/` artifacts: they are hash-linked
  records; changing them breaks verification tests. Create new versions instead.
- Package layout: `src/ft_screening/` (`document.py` parsing, `adjudication.py`
  LLM transport and criterion adjudication, `aggregation.py` decision
  building, `gold*.py` and `evaluation_harness.py` evaluation).
- Corporate TLS interception: point `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` at your
  CA bundle via environment variables; never commit `.pem` files.
