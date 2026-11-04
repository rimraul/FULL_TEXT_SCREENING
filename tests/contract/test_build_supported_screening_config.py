"""Tests for orchestrator.build_supported_screening_config() -- the
supported-config builder (api_contract_v4.md correction 10). All fast and
offline: this function only reads package metadata + the committed
Docling artifact manifest and validates option dicts in memory; it never
constructs a real component, never needs credentials, and never touches
the network.
"""

from __future__ import annotations

import os

import pytest

from ft_screening.aggregation import AGGREGATION_POLICY_VERSION
from ft_screening.document import DoclingDocumentParser
from ft_screening.errors import ComponentResolutionError
from ft_screening.retrieval import LexicalEvidenceRetriever
from ft_screening.schema import ScreeningConfig


def _build(**overrides):
    from ft_screening.orchestrator import build_supported_screening_config

    defaults = dict(
        adjudicator_model="gpt-5.6-terra",
        model_config_id="cfg-1",
        model_options={"max_completion_tokens": 600, "reasoning_effort": None},
    )
    defaults.update(overrides)
    return build_supported_screening_config(**defaults)


# ---------------------------------------------------------------------------
# Exact live identity binding.
# ---------------------------------------------------------------------------


def test_binds_exact_live_identities():
    from ft_screening.adjudication import LLMCriterionAdjudicator

    config = _build()
    assert config.parser_name == "docling"
    assert config.parser_version == DoclingDocumentParser.installed_parser_version()
    assert config.chunker_version == DoclingDocumentParser.CHUNKER_VERSION
    assert config.retriever_name == "lexical_bm25"
    assert config.retriever_config_version == LexicalEvidenceRetriever.RETRIEVER_CONFIG_VERSION
    assert config.adjudicator_provider == "openai"
    assert config.prompt_template_version == LLMCriterionAdjudicator.PROMPT_TEMPLATE_VERSION
    assert config.output_schema_version == LLMCriterionAdjudicator.OUTPUT_SCHEMA_VERSION
    assert config.aggregation_policy_version == AGGREGATION_POLICY_VERSION


def test_reads_live_constants_rather_than_duplicating_literals(monkeypatch):
    """Monkeypatch every live constant this function binds and confirm
    the built config reflects the NEW value -- proving it reads the
    implementation's own identity rather than hardcoding a copy of it."""
    from ft_screening.adjudication import LLMCriterionAdjudicator

    monkeypatch.setattr(DoclingDocumentParser, "CHUNKER_VERSION", "chunker-999")
    monkeypatch.setattr(LexicalEvidenceRetriever, "RETRIEVER_CONFIG_VERSION", "retriever-999")
    monkeypatch.setattr(LLMCriterionAdjudicator, "PROMPT_TEMPLATE_VERSION", "prompt-999")
    monkeypatch.setattr(LLMCriterionAdjudicator, "OUTPUT_SCHEMA_VERSION", "schema-999")
    monkeypatch.setattr("ft_screening.orchestrator.AGGREGATION_POLICY_VERSION", "policy-999")

    config = _build()
    assert config.chunker_version == "chunker-999"
    assert config.retriever_config_version == "retriever-999"
    assert config.prompt_template_version == "prompt-999"
    assert config.output_schema_version == "schema-999"
    assert config.aggregation_policy_version == "policy-999"


# ---------------------------------------------------------------------------
# Identity NAMES (parser_name/retriever_name/adjudicator_provider) are read
# from resolve_components()'s own _SUPPORTED_* registries, not duplicated
# literals (codex_code_review_build_supported_screening_config.md finding 3).
# ---------------------------------------------------------------------------


def test_reads_identity_names_from_resolver_registry_and_config_still_resolves(tmp_path, monkeypatch):
    """Monkeypatch all three _SUPPORTED_* registries to a renamed
    singleton each, confirm the built config picks up the NEW name (not
    a hardcoded "docling"/"lexical_bm25"/"openai" literal), and confirm
    resolve_components() -- reading the SAME patched registries --
    accepts that renamed identity. This is the round-trip proof that the
    builder and the resolver share one source of truth."""
    from ft_screening import orchestrator

    monkeypatch.setattr(orchestrator, "_SUPPORTED_PARSER_NAMES", frozenset({"docling-renamed"}))
    monkeypatch.setattr(orchestrator, "_SUPPORTED_RETRIEVER_NAMES", frozenset({"lexical-renamed"}))
    monkeypatch.setattr(orchestrator, "_SUPPORTED_ADJUDICATOR_PROVIDERS", frozenset({"openai-renamed"}))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-obviously-not-a-real-key")

    config = _build()
    assert config.parser_name == "docling-renamed"
    assert config.retriever_name == "lexical-renamed"
    assert config.adjudicator_provider == "openai-renamed"

    components = orchestrator.resolve_components(config, output_dir=tmp_path)
    assert components is not None


@pytest.mark.parametrize(
    "registry_name",
    ["_SUPPORTED_PARSER_NAMES", "_SUPPORTED_RETRIEVER_NAMES", "_SUPPORTED_ADJUDICATOR_PROVIDERS"],
)
@pytest.mark.parametrize("bad_registry", [frozenset(), frozenset({"a", "b"})], ids=["empty", "multi-entry"])
def test_fails_loudly_when_supported_identity_registry_is_not_a_singleton(monkeypatch, registry_name, bad_registry):
    """This builder intentionally supports exactly ONE stack -- an empty
    or multi-entry _SUPPORTED_* registry is a configuration bug, and must
    fail loudly rather than arbitrarily picking one identity or crashing
    unhelpfully."""
    from ft_screening import orchestrator

    monkeypatch.setattr(orchestrator, registry_name, bad_registry)
    with pytest.raises(ComponentResolutionError):
        _build()


def test_parser_version_reads_installed_docling_package(monkeypatch):
    monkeypatch.setattr(DoclingDocumentParser, "installed_parser_version", staticmethod(lambda: "1.2.3-fake"))
    config = _build()
    assert config.parser_version == "1.2.3-fake"


def test_artifacts_manifest_digest_reads_committed_manifest():
    from ft_screening import docling_artifacts

    config = _build()
    assert config.parser_options["artifacts_manifest_digest"] == docling_artifacts.load_manifest()["expected_tree_digest"]


# ---------------------------------------------------------------------------
# Fully materialized options -- no hidden defaults.
# ---------------------------------------------------------------------------


def test_parser_options_fully_materialized():
    config = _build()
    assert config.parser_options == {
        "do_ocr": True,
        "do_table_structure": True,
        "artifacts_manifest_digest": config.parser_options["artifacts_manifest_digest"],
    }


def test_parser_options_respects_ocr_table_customization():
    config = _build(parser_do_ocr=False, parser_do_table_structure=False)
    assert config.parser_options["do_ocr"] is False
    assert config.parser_options["do_table_structure"] is False


def test_retriever_options_fully_materialized_with_no_override():
    config = _build()
    assert config.retriever_options == {
        "max_results": 5,
        "min_results": 1,
        "min_score": 0.0,
        "section_match_bonus": 2.0,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "full_document_fallback_max_chars": 20000,
        "max_candidates": 100,
        "max_evidence_chars": 20000,
    }


def test_retriever_options_customization_merges_with_materialized_defaults():
    config = _build(retriever_options={"max_results": 10})
    assert config.retriever_options["max_results"] == 10
    assert config.retriever_options["min_results"] == 1  # untouched default, still materialized


def test_model_options_fully_materialized_beyond_mandatory_fields():
    config = _build(model_options={"max_completion_tokens": 600, "reasoning_effort": None})
    assert config.model_options == {
        "timeout_seconds": 60.0,
        "max_retries": 2,
        "initial_retry_delay_seconds": 1.0,
        "max_retry_delay_seconds": 20.0,
        "retry_backoff_multiplier": 2.0,
        "max_rationale_chars": 2000,
        "max_completion_tokens": 600,
        "reasoning_effort": None,
    }


def test_model_options_customization_merges_with_materialized_defaults():
    config = _build(model_options={"max_completion_tokens": 4096, "reasoning_effort": "medium"})
    assert config.model_options["reasoning_effort"] == "medium"
    assert config.model_options["max_completion_tokens"] == 4096
    assert config.model_options["timeout_seconds"] == 60.0  # untouched default, still materialized


# ---------------------------------------------------------------------------
# Required model_options.max_completion_tokens / reasoning_effort -- never
# a silently chosen token budget or reasoning effort
# (codex_code_review_build_supported_screening_config.md finding 1).
# ---------------------------------------------------------------------------


def test_rejects_missing_max_completion_tokens():
    with pytest.raises(ComponentResolutionError, match="max_completion_tokens"):
        _build(model_options={"reasoning_effort": None})


def test_rejects_missing_max_completion_tokens_even_with_other_model_options_set():
    with pytest.raises(ComponentResolutionError, match="max_completion_tokens"):
        _build(model_options={"reasoning_effort": "medium"})


def test_rejects_missing_reasoning_effort_even_with_max_completion_tokens_set():
    """The exact repro from finding 1: omitting reasoning_effort entirely
    must NOT silently fall through to _ModelOptionsModel's own
    reasoning_effort=None default."""
    with pytest.raises(ComponentResolutionError, match="reasoning_effort"):
        _build(model_options={"max_completion_tokens": 600})


def test_accepts_explicit_reasoning_effort_none_for_non_reasoning_model():
    """None is a legitimate EXPLICIT choice (non-reasoning model) --
    distinct from OMITTING the key, which is rejected above."""
    config = _build(model_options={"max_completion_tokens": 600, "reasoning_effort": None})
    assert config.model_options["reasoning_effort"] is None


def test_accepts_explicit_reasoning_effort_medium():
    config = _build(model_options={"max_completion_tokens": 4096, "reasoning_effort": "medium"})
    assert config.model_options["reasoning_effort"] == "medium"


# ---------------------------------------------------------------------------
# model_options / retriever_options type validation -- non-dict values
# must be sanitized ComponentResolutionErrors, never a raw TypeError, and
# a falsy-but-real value must never be silently treated as {}
# (codex_code_review_build_supported_screening_config.md finding 2).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_model_options", [None, [], False, "", "not-a-dict", 0])
def test_rejects_non_dict_model_options_without_raw_typeerror(bad_model_options):
    with pytest.raises(ComponentResolutionError, match="model_options"):
        _build(model_options=bad_model_options)


@pytest.mark.parametrize("bad_retriever_options", [[], False, "", "not-a-dict", 0, ["unexpected", "list"]])
def test_rejects_falsy_or_non_dict_retriever_options_rather_than_silently_defaulting(bad_retriever_options):
    """`retriever_options or {}` would silently accept every falsy value
    here as "use all defaults" -- must be rejected instead, since none of
    these are actually `None`."""
    with pytest.raises(ComponentResolutionError, match="retriever_options"):
        _build(retriever_options=bad_retriever_options)


def test_retriever_options_none_still_means_use_all_defaults():
    """`None` (the documented sentinel) is the ONE falsy value that's
    still accepted -- distinguishing this from the rejected cases above
    proves the fix checks `is None` specifically, not truthiness."""
    config = _build(retriever_options=None)
    assert config.retriever_options["max_results"] == 5


# ---------------------------------------------------------------------------
# Invalid-option sanitization.
# ---------------------------------------------------------------------------


def test_rejects_unknown_model_option_key_sanitized():
    """Sanitized means the error reports the exception TYPE
    (`ValidationError`, same convention as resolve_components' own
    _resolve_* wrappers) but never the raw offending key/value."""
    with pytest.raises(ComponentResolutionError, match="model_options") as excinfo:
        _build(model_options={"max_completion_tokens": 600, "reasoning_effort": None, "unknown_option": True})
    assert "unknown_option" not in str(excinfo.value)


def test_rejects_unknown_retriever_option_key_sanitized():
    with pytest.raises(ComponentResolutionError, match="retriever_options") as excinfo:
        _build(retriever_options={"unknown_option": True})
    assert "unknown_option" not in str(excinfo.value)


def test_rejects_out_of_range_model_option_value():
    with pytest.raises(ComponentResolutionError, match="model_options"):
        _build(model_options={"max_completion_tokens": 999999, "reasoning_effort": None})


def test_error_never_leaks_a_local_filesystem_path():
    """Sanitized errors report only the exception TYPE name, never
    str(exc) -- which for a pydantic ValidationError could embed
    caller-supplied values verbatim."""
    with pytest.raises(ComponentResolutionError) as excinfo:
        _build(model_options={
            "max_completion_tokens": 600, "reasoning_effort": None,
            "secret_path_like_value": "C:/Users/rimraul/secret",
        })
    assert "C:/Users/rimraul" not in str(excinfo.value)
    assert "secret_path_like_value" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Rejection of identity/artifact overrides -- structural, proven by
# Python's own keyword-argument enforcement (the signature has no
# parameter for any of these).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_kwarg",
    [
        "parser_name",
        "parser_version",
        "chunker_version",
        "parser_options",
        "retriever_name",
        "retriever_config_version",
        "adjudicator_provider",
        "prompt_template_version",
        "output_schema_version",
        "aggregation_policy_version",
    ],
)
def test_rejects_identity_override_kwargs_structurally(bad_kwarg):
    from ft_screening.orchestrator import build_supported_screening_config

    with pytest.raises(TypeError):
        build_supported_screening_config(
            adjudicator_model="gpt-5.6-terra",
            model_config_id="cfg-1",
            model_options={"max_completion_tokens": 600},
            **{bad_kwarg: "attempted-override"},
        )


# ---------------------------------------------------------------------------
# Required parameters -- no defaults.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["adjudicator_model", "model_config_id", "model_options"])
def test_required_parameters_have_no_default(missing):
    from ft_screening.orchestrator import build_supported_screening_config

    kwargs = dict(
        adjudicator_model="gpt-5.6-terra",
        model_config_id="cfg-1",
        model_options={"max_completion_tokens": 600},
    )
    del kwargs[missing]
    with pytest.raises(TypeError):
        build_supported_screening_config(**kwargs)


# ---------------------------------------------------------------------------
# Runtime-policy customization.
# ---------------------------------------------------------------------------


def test_runtime_policy_customization():
    config = _build(
        cache_mode="refresh", duplicate_content_policy="reuse", item_error_policy="fail_fast", max_workers=4,
    )
    assert config.cache_mode == "refresh"
    assert config.duplicate_content_policy == "reuse"
    assert config.item_error_policy == "fail_fast"
    assert config.max_workers == 4


# ---------------------------------------------------------------------------
# Determinism and serialization round-trip.
# ---------------------------------------------------------------------------


def test_deterministic_across_calls():
    a = _build()
    b = _build()
    assert a.model_dump(mode="json") == b.model_dump(mode="json")


def test_returned_object_is_an_ordinary_frozen_screening_config():
    config = _build()
    assert isinstance(config, ScreeningConfig)
    with pytest.raises(Exception):
        config.adjudicator_model = "different"  # frozen -- assignment must fail


def test_serialization_round_trip():
    config = _build()
    dumped = config.model_dump(mode="json")
    reloaded = ScreeningConfig.model_validate(dumped)
    assert reloaded.model_dump(mode="json") == dumped


# ---------------------------------------------------------------------------
# No side effects: no credentials, network, OpenAI client, model loading,
# or filesystem writes.
# ---------------------------------------------------------------------------


def test_requires_no_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "OPENAI_API_KEY" not in os.environ
    config = _build()  # must not raise
    assert config is not None


def test_never_imports_my_api_keys():
    import sys

    assert "my_api_keys" not in sys.modules
    _build()
    assert "my_api_keys" not in sys.modules


def test_never_constructs_an_openai_client_or_imports_openai():
    import sys

    sys.modules.pop("openai", None)
    _build()
    assert "openai" not in sys.modules


def test_never_touches_docling_conversion_module():
    """document.py's own top-level imports are already docling-free
    (confirmed in resolve_components' own lazy-import tests); this
    function's import of ft_screening.document must not change that."""
    import sys

    sys.modules.pop("docling", None)
    _build()
    assert "docling" not in sys.modules


def test_creates_no_files_or_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _build()
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Public import paths.
# ---------------------------------------------------------------------------


def test_importable_from_ft_screening_api():
    from ft_screening.api import ComponentResolutionError as ApiError
    from ft_screening.api import build_supported_screening_config as api_builder

    assert api_builder is not None
    assert ApiError is ComponentResolutionError


def test_importable_from_ft_screening_top_level():
    import ft_screening

    assert ft_screening.build_supported_screening_config is not None
    assert ft_screening.ComponentResolutionError is ComponentResolutionError
    config = ft_screening.build_supported_screening_config(
        adjudicator_model="gpt-5.6-terra", model_config_id="cfg-1",
        model_options={"max_completion_tokens": 600, "reasoning_effort": None},
    )
    assert isinstance(config, ScreeningConfig)


# ---------------------------------------------------------------------------
# The returned config actually works: resolve_components() accepts it
# under a fake (construction-only) credential.
# ---------------------------------------------------------------------------


def test_returned_config_passes_resolve_components_with_fake_credentials(tmp_path, monkeypatch):
    from ft_screening.orchestrator import resolve_components

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-obviously-not-a-real-key")
    config = _build()
    components = resolve_components(config, output_dir=tmp_path)
    assert components is not None
