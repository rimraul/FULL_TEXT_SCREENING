"""Public integration entry point (codex_code_review_44d9e90.md finding
10 -- the frozen package layout names this module as the consumer-facing
boundary; it didn't exist yet). Downstream projects (e.g. `slr_pipeline`)
should import from here, never from `ft_screening.orchestrator` directly
-- `_run_ft_screening` and other internals stay private.
"""

from __future__ import annotations

from ft_screening.errors import ComponentResolutionError
from ft_screening.orchestrator import build_supported_screening_config, resolve_components, run_ft_screening
from ft_screening.schema import (
    CriteriaSet,
    CriterionAssessment,
    CriterionSpec,
    DocumentCoverage,
    EligibilityResult,
    EvidenceSpan,
    ExecutionProvenance,
    HumanAdjudication,
    ProcessingError,
    ProgressEvent,
    RunArtifacts,
    RunSummary,
    ScreeningConfig,
    ScreeningItem,
    ScreeningItemResult,
    ScreeningRequest,
    ScreeningRunResult,
)

__all__ = [
    "run_ft_screening",
    "resolve_components",
    "build_supported_screening_config",
    "ComponentResolutionError",
    "CriteriaSet",
    "CriterionSpec",
    "ScreeningItem",
    "ScreeningConfig",
    "ScreeningRequest",
    "CriterionAssessment",
    "EvidenceSpan",
    "HumanAdjudication",
    "EligibilityResult",
    "ProcessingError",
    "ExecutionProvenance",
    "DocumentCoverage",
    "ProgressEvent",
    "ScreeningItemResult",
    "RunSummary",
    "RunArtifacts",
    "ScreeningRunResult",
]
