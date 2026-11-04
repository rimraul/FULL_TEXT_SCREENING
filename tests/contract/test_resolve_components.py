"""Tests for orchestrator.resolve_components() -- the real-component
factory/resolver. All fast: resolve_components() only validates identity
strings/versions and CONSTRUCTS real component objects; it never parses a
PDF or makes a provider call, so none of these tests need the `slow`/
`real_llm` markers, real Docling artifacts, or a real OPENAI_API_KEY.

`OPENAI_API_KEY` is set to an obviously-fake string (never a real key) for
the "successful wiring" tests -- the `openai` SDK's own client
constructor does not make a network call or validate key format, so this
proves construction-only, zero-network wiring without touching a real
credential.
"""

from __future__ import annotations

import os

import pytest

from ft_screening.document import DoclingDocumentParser
from ft_screening.errors import ComponentResolutionError
from ft_screening.retrieval import LexicalEvidenceRetriever
from ft_screening.schema import ScreeningConfig


def _config(**overrides) -> ScreeningConfig:
    defaults = dict(
        parser_name="docling", parser_version=DoclingDocumentParser.installed_parser_version(),
        chunker_version=DoclingDocumentParser.CHUNKER_VERSION,
        parser_options=DoclingDocumentParser.recommended_parser_options(),
        retriever_name="lexical_bm25", retriever_config_version=LexicalEvidenceRetriever.RETRIEVER_CONFIG_VERSION,
        retriever_options={},
        adjudicator_provider="openai", adjudicator_model="gpt-5.6-terra",
        model_config_id="cfg-1", model_options={},
        prompt_template_version="adjudication-prompt-6", output_schema_version="adjudication-schema-2",
        aggregation_policy_version="1",
    )
    defaults.update(overrides)
    return ScreeningConfig(**defaults)


@pytest.fixture(autouse=True)
def _clean_openai_env(monkeypatch):
    """Every test starts with NO OPENAI_API_KEY -- tests that need
    "credentials present" set an obviously-fake one explicitly."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _with_fake_credentials(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-obviously-not-a-real-key")


# ---------------------------------------------------------------------------
# Successful wiring + output-directory binding.
# ---------------------------------------------------------------------------


def test_resolve_components_successful_wiring(tmp_path, monkeypatch):
    from ft_screening.adjudication import LLMCriterionAdjudicator, OpenAIChatCompletionTransport
    from ft_screening.orchestrator import resolve_components
    from ft_screening.store import FileScreeningStore

    _with_fake_credentials(monkeypatch)
    components = resolve_components(_config(), output_dir=tmp_path)

    assert isinstance(components.parser, DoclingDocumentParser)
    assert isinstance(components.retriever, LexicalEvidenceRetriever)
    assert isinstance(components.adjudicator, LLMCriterionAdjudicator)
    assert isinstance(components.adjudicator._transport, OpenAIChatCompletionTransport)
    assert components.adjudicator.model == "gpt-5.6-terra"
    assert isinstance(components.store, FileScreeningStore)


def test_resolve_components_binds_store_to_explicit_output_dir(tmp_path, monkeypatch):
    from ft_screening.orchestrator import resolve_components

    _with_fake_credentials(monkeypatch)
    target_dir = tmp_path / "some" / "specific" / "output_dir"
    components = resolve_components(_config(), output_dir=target_dir)
    assert components.store.output_dir.resolve() == target_dir.resolve()


# ---------------------------------------------------------------------------
# Unsupported identities.
# ---------------------------------------------------------------------------


def test_resolve_components_rejects_unsupported_parser_name(tmp_path, monkeypatch):
    from ft_screening.orchestrator import resolve_components

    _with_fake_credentials(monkeypatch)
    with pytest.raises(ComponentResolutionError, match="unsupported parser_name"):
        resolve_components(_config(parser_name="pdfminer"), output_dir=tmp_path)


def test_resolve_components_rejects_unsupported_retriever_name(tmp_path, monkeypatch):
    from ft_screening.orchestrator import resolve_components

    _with_fake_credentials(monkeypatch)
    with pytest.raises(ComponentResolutionError, match="unsupported retriever_name"):
        resolve_components(_config(retriever_name="dense_vector"), output_dir=tmp_path)


def test_resolve_components_rejects_unsupported_adjudicator_provider(tmp_path, monkeypatch):
    from ft_screening.orchestrator import resolve_components

    _with_fake_credentials(monkeypatch)
    with pytest.raises(ComponentResolutionError, match="unsupported adjudicator_provider"):
        resolve_components(_config(adjudicator_provider="anthropic"), output_dir=tmp_path)


def test_resolve_components_never_resolves_fake_identity(tmp_path, monkeypatch):
    """"fake" is a test-only identity (FakeDocumentParser/etc.) -- must
    never be resolvable through the public factory."""
    from ft_screening.orchestrator import resolve_components

    _with_fake_credentials(monkeypatch)
    with pytest.raises(ComponentResolutionError, match="unsupported parser_name"):
        resolve_components(_config(parser_name="fake"), output_dir=tmp_path)


# ---------------------------------------------------------------------------
# Version drift -- declared config must match the LIVE implementation.
# ---------------------------------------------------------------------------


def test_resolve_components_rejects_parser_version_drift(tmp_path, monkeypatch):
    from ft_screening.orchestrator import resolve_components

    _with_fake_credentials(monkeypatch)
    with pytest.raises(ComponentResolutionError, match="parser_version"):
        resolve_components(_config(parser_version="0.0.0"), output_dir=tmp_path)


def test_resolve_components_rejects_chunker_version_drift(tmp_path, monkeypatch):
    from ft_screening.orchestrator import resolve_components

    _with_fake_credentials(monkeypatch)
    with pytest.raises(ComponentResolutionError, match="chunker_version"):
        resolve_components(_config(chunker_version="999"), output_dir=tmp_path)


def test_resolve_components_rejects_retriever_config_version_drift(tmp_path, monkeypatch):
    from ft_screening.orchestrator import resolve_components

    _with_fake_credentials(monkeypatch)
    with pytest.raises(ComponentResolutionError, match="retriever_config_version"):
        resolve_components(_config(retriever_config_version="999"), output_dir=tmp_path)


def test_resolve_components_rejects_prompt_template_version_drift(tmp_path, monkeypatch):
    from ft_screening.orchestrator import resolve_components

    _with_fake_credentials(monkeypatch)
    with pytest.raises(ComponentResolutionError, match="prompt_template_version"):
        resolve_components(_config(prompt_template_version="adjudication-prompt-999"), output_dir=tmp_path)


def test_resolve_components_rejects_output_schema_version_drift(tmp_path, monkeypatch):
    from ft_screening.orchestrator import resolve_components

    _with_fake_credentials(monkeypatch)
    with pytest.raises(ComponentResolutionError, match="output_schema_version"):
        resolve_components(_config(output_schema_version="adjudication-schema-999"), output_dir=tmp_path)


# ---------------------------------------------------------------------------
# Incompatible options.
# ---------------------------------------------------------------------------


def test_resolve_components_rejects_incompatible_parser_options(tmp_path, monkeypatch):
    from ft_screening.orchestrator import resolve_components

    _with_fake_credentials(monkeypatch)
    with pytest.raises(ComponentResolutionError, match="parser_options"):
        resolve_components(_config(parser_options={}), output_dir=tmp_path)  # missing mandatory artifacts_manifest_digest


def test_resolve_components_rejects_incompatible_retriever_options(tmp_path, monkeypatch):
    from ft_screening.orchestrator import resolve_components

    _with_fake_credentials(monkeypatch)
    with pytest.raises(ComponentResolutionError, match="retriever_options"):
        resolve_components(_config(retriever_options={"unknown_option": True}), output_dir=tmp_path)


def test_resolve_components_rejects_incompatible_model_options(tmp_path, monkeypatch):
    from ft_screening.orchestrator import resolve_components

    _with_fake_credentials(monkeypatch)
    with pytest.raises(ComponentResolutionError, match="model_options"):
        resolve_components(_config(model_options={"unknown_option": True}), output_dir=tmp_path)


# ---------------------------------------------------------------------------
# Aggregation-policy-version validation (codex_code_review_2be6cfa.md
# finding 1): the live aggregation implementation is one hard-coded,
# unversioned state machine, but its version is hashed into
# execution_fingerprint -- an arbitrary/typo'd/future value must be
# rejected, not silently resolved and hashed.
# ---------------------------------------------------------------------------


def test_resolve_components_rejects_unsupported_aggregation_policy_version(tmp_path):
    from ft_screening.orchestrator import resolve_components

    # Deliberately no fake credentials and a non-existent output_dir --
    # this must fail before adjudicator resolution (which needs
    # credentials) or store construction ever run.
    assert "OPENAI_API_KEY" not in os.environ
    with pytest.raises(ComponentResolutionError, match="aggregation_policy_version"):
        resolve_components(_config(aggregation_policy_version="definitely-not-the-live-policy"), output_dir=tmp_path / "unused")


def test_resolve_components_rejects_aggregation_policy_version_before_any_other_work(tmp_path, monkeypatch):
    """An unsupported aggregation_policy_version must fail BEFORE
    parser/retriever/adjudicator resolution or store construction --
    proven by: no OPENAI_API_KEY needed, ft_screening.document (docling's
    lazy-import boundary) stays unimported even though it isn't already
    loaded from an earlier test in this process (unlike `openai`, which
    other tests in this file legitimately import and leave in
    sys.modules, so it isn't a reliable signal here), and the output_dir
    is never created on disk."""
    import sys

    from ft_screening.orchestrator import resolve_components

    sys.modules.pop("ft_screening.document", None)
    output_dir = tmp_path / "never_created"
    with pytest.raises(ComponentResolutionError, match="aggregation_policy_version"):
        resolve_components(_config(aggregation_policy_version="bogus", parser_name="also-bogus"), output_dir=output_dir)
    assert "ft_screening.document" not in sys.modules
    assert not output_dir.exists()


def test_resolve_components_accepts_live_aggregation_policy_version(tmp_path, monkeypatch):
    from ft_screening.aggregation import AGGREGATION_POLICY_VERSION
    from ft_screening.orchestrator import resolve_components

    _with_fake_credentials(monkeypatch)
    components = resolve_components(_config(aggregation_policy_version=AGGREGATION_POLICY_VERSION), output_dir=tmp_path)
    assert components is not None


# ---------------------------------------------------------------------------
# Missing/broken docling installation escapes as ComponentResolutionError,
# not a raw importlib.metadata.PackageNotFoundError (codex_code_review_2be6cfa.md
# finding 3).
# ---------------------------------------------------------------------------


def test_resolve_components_wraps_missing_docling_package_metadata(tmp_path, monkeypatch):
    import importlib.metadata

    from ft_screening.document import DoclingDocumentParser
    from ft_screening.orchestrator import resolve_components

    _with_fake_credentials(monkeypatch)

    def _raise_package_not_found() -> str:
        raise importlib.metadata.PackageNotFoundError("docling")

    monkeypatch.setattr(DoclingDocumentParser, "installed_parser_version", staticmethod(_raise_package_not_found))
    with pytest.raises(ComponentResolutionError, match="docling"):
        resolve_components(_config(), output_dir=tmp_path)


# ---------------------------------------------------------------------------
# Missing credentials, at the right boundary.
# ---------------------------------------------------------------------------


def test_resolve_components_rejects_missing_credentials(tmp_path):
    from ft_screening.orchestrator import resolve_components

    assert "OPENAI_API_KEY" not in os.environ
    with pytest.raises(ComponentResolutionError, match="credentials"):
        resolve_components(_config(), output_dir=tmp_path)


def test_resolve_components_never_imports_my_api_keys(tmp_path, monkeypatch):
    """resolve_components must never attempt to load credentials itself
    -- proven by it FAILING with no OPENAI_API_KEY set, rather than
    somehow succeeding via a side-channel import."""
    import sys

    from ft_screening.orchestrator import resolve_components

    assert "my_api_keys" not in sys.modules
    with pytest.raises(ComponentResolutionError):
        resolve_components(_config(), output_dir=tmp_path)
    assert "my_api_keys" not in sys.modules


# ---------------------------------------------------------------------------
# Lazy imports: resolving one identity must never construct/import the
# machinery for a DIFFERENT one.
# ---------------------------------------------------------------------------


def test_resolve_components_unsupported_parser_never_touches_docling_module(tmp_path, monkeypatch):
    import sys

    _with_fake_credentials(monkeypatch)
    from ft_screening.orchestrator import resolve_components

    sys.modules.pop("ft_screening.document", None)
    with pytest.raises(ComponentResolutionError):
        resolve_components(_config(parser_name="bogus"), output_dir=tmp_path)
    assert "ft_screening.document" not in sys.modules


def test_resolve_components_unsupported_adjudicator_never_touches_openai_module(tmp_path, monkeypatch):
    import sys

    _with_fake_credentials(monkeypatch)
    from ft_screening.orchestrator import resolve_components

    sys.modules.pop("openai", None)
    with pytest.raises(ComponentResolutionError):
        resolve_components(_config(adjudicator_provider="bogus"), output_dir=tmp_path)
    assert "openai" not in sys.modules


# ---------------------------------------------------------------------------
# The public run_ft_screening() path: resolve_components is called with
# request.output_dir specifically (not some other path).
# ---------------------------------------------------------------------------


def test_run_ft_screening_calls_resolve_components_with_request_output_dir(tmp_path, monkeypatch):
    from ft_screening import orchestrator
    from ft_screening.components import Components
    from ft_screening.fakes import FakeCriterionAdjudicator, FakeDocumentParser, FakeEvidenceRetriever
    from ft_screening.schema import CriteriaSet, CriterionSpec, ScreeningItem, ScreeningRequest
    from ft_screening.store import FileScreeningStore

    pdf_path = tmp_path / "d1.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content for wiring test only")

    criteria_set = CriteriaSet(criteria_set_id="cs-1", version="1", criteria=[
        CriterionSpec(
            criterion_id="population", version="1", eligibility_requirement="adult humans",
            missing_evidence_policy="review", exclusion_reason_code="wrong_population",
            exclusion_reason_label="Wrong population", priority=1,
        ),
    ])
    config = _config(parser_name="fake", parser_version="1", chunker_version="1", retriever_name="fake", retriever_config_version="1", adjudicator_provider="fake")
    request = ScreeningRequest(
        items=[ScreeningItem(document_id="d1", report_id="r1", pdf_path=pdf_path)],
        criteria_set=criteria_set, config=config, output_dir=tmp_path / "run_output",
    )

    calls = []

    def _fake_resolve_components(cfg, *, output_dir):
        calls.append({"config": cfg, "output_dir": output_dir})
        return Components(
            parser=FakeDocumentParser(), retriever=FakeEvidenceRetriever(), adjudicator=FakeCriterionAdjudicator(),
            store=FileScreeningStore(output_dir=output_dir),
        )

    monkeypatch.setattr(orchestrator, "resolve_components", _fake_resolve_components)
    orchestrator.run_ft_screening(request)

    assert len(calls) == 1
    assert calls[0]["output_dir"] == request.output_dir
    assert calls[0]["config"] is request.config


# ---------------------------------------------------------------------------
# Public-surface test (codex_code_review_2be6cfa.md finding 2): calls
# resolve_components exactly as api_contract_v3.md correction 8 documents
# it -- positional config, keyword-only output_dir, imported from the
# package-level `ft_screening` surface (not the internal orchestrator
# module) -- so the documented signature and code example cannot drift
# from the real one again without a test failing.
# ---------------------------------------------------------------------------


def test_resolve_components_matches_api_contract_v3_documented_call(tmp_path, monkeypatch):
    import ft_screening
    from ft_screening.components import Components

    _with_fake_credentials(monkeypatch)
    components = ft_screening.resolve_components(_config(), output_dir=tmp_path)
    assert isinstance(components, Components)
    assert components.store.output_dir.resolve() == tmp_path.resolve()
