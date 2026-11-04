# Arm 2b v1 incident note: document_sha256 provenance gap, repaired offline as v2

## What happened

Arm 2b's redacted `EvidenceSpan`s carry the real PDF hash
(`gold_set.document_pdf_sha256[document_id]`) rather than the historical
`"0"*64` placeholder every other `gold_evidence`-family arm's evidence
uses (Arm 2, Arm 2 prompt-6). The first real `--execute` run completed
all 60 paid adjudication calls successfully (checkpointed individually),
then crashed in the post-loop scoring step:
`evaluation_harness.compute_arm_gold_evidence_document_decisions()`
hardcoded `document_sha256="0"*64` unconditionally when building each
document's `EligibilityResult`, so `aggregation.build_eligibility_result()`'s
own evidence-identity check (`span.document_sha256 != document_sha256`)
raised `AggregationError` on every document.

The run was unblocked with a runner-only, ephemeral, in-memory
normalization (`_normalized_for_document_decisions()` in
`run_arm2b_gold_evidence_set.py`) that zeroed the evidence's
`document_sha256` only for the duration of that one call, never touching
any checkpoint. This worked and produced a correct scoring result, but
left a real provenance gap: **the successful rerun's manifest recorded
`git_commit=ff68b7d`, but the fix that actually produced that result was
uncommitted at the time** (`source_identity.py`'s `RELEVANT_SOURCE_FILES`
covers `adjudication.py`/`components.py`/`hashing.py` only, not the
runner script itself, so `git_dirty_relevant_files=False` did not catch
the mismatch).

Separately, `v1`'s frozen result recorded `total_wall_seconds=0.0`. This
is not data loss from the incident -- checkpoint files
(`GoldEvidenceCriterionCheckpoint`) never recorded per-call wall time to
begin with, and every one of the 60 calls in the run that actually
produced `v1`'s frozen artifacts was loaded from an existing checkpoint
(the resume path), which has never contributed to
`wall_by_doc`/`wall_time_summary_gold_evidence` in this harness. Real,
but a pre-existing harness limitation surfaced by this incident, not
caused by it.

## Disposition

- **`arm2b_gold_evidence_set_result_v1.json` / `arm2b_gold_evidence_set_manifest_v1.json`
  are preserved UNCHANGED** as the historical incident record. Not to be
  cited as the canonical Arm 2b result going forward -- its provenance
  chain (uncommitted runner fix) is impure, even though its *numbers*
  (agreement, token usage) are correct and byte-identical to v2's.
- **The fix was moved into the shared, always-imported function itself**
  (`evaluation_harness._document_sha256_for_decision()`, committed at
  `a2a6780`): derives `document_sha256` from the assessments' own
  evidence (correct for both Arm 2b's real-hash convention and Arm 2's
  pre-existing `"0"*64` convention, no arm-specific branching), falls
  back to `gold_set.document_pdf_sha256` only when a document cites no
  evidence at all, and raises if a document's assessments ever disagree
  on the hash. The runner-only normalization was deleted entirely.
- **`arm2b_gold_evidence_set_result_v2.json` / `arm2b_gold_evidence_set_manifest_v2.json`
  are the canonical Arm 2b artifacts.** Produced by
  `run_arm2b_gold_evidence_set.py --finalize-from-checkpoints`, entirely
  OFFLINE from the same 60 already-completed checkpoints -- **zero new
  provider calls**, no adjudicator/transport constructed, no credentials
  loaded.

## v2 provenance, recorded in `arm2b_gold_evidence_set_manifest_v2.json`'s `config_summary`

- `v1_result_content_hash` / `v1_manifest_content_hash` -- binds v2 back
  to the exact historical v1 pair it supersedes.
- `checkpoint_execution_identity_hash` -- the single execution identity
  all 60 checkpoints share (verified, not assumed).
- `new_provider_calls: 0`.
- `runner_file_sha256` / `evaluation_harness_file_sha256` -- the exact
  committed file bytes this finalization ran under (commit `a2a6780`).
- `wall_time_unavailable_after_checkpoint_resume: true` -- explicit,
  honest flag rather than a silently-zero number with no explanation.
- `source_identity` -- git commit + dirty-state of the RELEVANT_SOURCE_FILES
  set (`adjudication.py`/`components.py`/`hashing.py`), same mechanism
  every other arm's manifest uses.

## Hashes for reference

| Artifact | content_hash |
|---|---|
| gold_evidence_set_v1.json | `eb7ac00b2fd1d6456b8bdf2ac58d67f58af01ae65ceaff42bd7c53c2512fc2db` |
| arm2b_evidence_projection_v1.json | `c827b8aa01e11953ed5ff51f3c4744d5f6212c3220361d3125f1cb15102e4163` |
| arm2b_gold_evidence_set_result_v1.json | `349dba726019db58c674c6c4e263354bb22a49dd82ee3ef692f58fe101b8ce76` |
| arm2b_gold_evidence_set_manifest_v1.json | `a85ee48ced8515625e5433d1926d891319f3d6bb139c1dd38960037a3e58dbcc` |
| arm2b_gold_evidence_set_result_v2.json | `349dba726019db58c674c6c4e263354bb22a49dd82ee3ef692f58fe101b8ce76` (identical to v1 -- document_sha256 is an identity binding, not a decision input) |
| arm2b_gold_evidence_set_manifest_v2.json | `c8908f1641a725a621c7189dba320fe60113891d254c6e283a15520c865bf47d` |

## What did NOT change

Arm 2b's agreement and token findings (criterion_status_agreement=0.8167,
final_decision_agreement=0.6500, billable tokens=142,102, etc. -- see the
main report) were never wrong and did not need paid recomputation. Only
their source attribution (which commit actually produced them) and the
already-honest `total_wall_seconds=0.0` label needed repair.
