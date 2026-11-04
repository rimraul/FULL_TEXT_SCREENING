"""Exceptions the call itself cannot be meaningfully executed under
(api_contract_v2.md "Exceptions and continuation policy"). Malformed
requests raise pydantic.ValidationError directly at ScreeningRequest
construction -- that already serves the contract's "RequestValidationError"
role, no separate class needed for it.
"""

from __future__ import annotations


class RunSetupError(Exception):
    """Output directory/ledger cannot be initialized, or a fresh run_id
    collides with an existing run."""


class ResumeConflictError(Exception):
    """Requested run cannot be safely resumed (doesn't exist, is locked by
    another process, or its criteria/execution fingerprint no longer
    matches the current request)."""


class ComponentResolutionError(Exception):
    """`resolve_components()` cannot build a real `Components` for the
    given `ScreeningConfig`: an unsupported parser/retriever/adjudicator
    identity name, a declared version that doesn't match the live
    implementation actually installed/available, incompatible
    parser_options/retriever_options/model_options, or credentials the
    caller's environment did not actually provide."""
