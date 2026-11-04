"""Full-text eligibility screening for systematic/scoping literature reviews.

Public contract: see api_contract_v4.md in the project root (adds
build_supported_screening_config()); api_contract_v3.md amends v2's
resolve_components() signature and adds aggregation-policy-version
validation; v2 remains authoritative for everything else. Import the
consumer-facing surface from `ft_screening.api` (or directly from
`ft_screening`, re-exported here) -- `_run_ft_screening` and other
orchestrator internals are private.
"""

from ft_screening.api import (  # noqa: F401
    ComponentResolutionError,
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
    build_supported_screening_config,
    resolve_components,
    run_ft_screening,
)
