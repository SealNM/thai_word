# Pathumma data-enrichment experiment — 2026-09-18

Status: prompt-v2 deterministic smoke exposed evidence-handling regressions; 300-pair scaling is paused. Prompt v3 evidence-aware semantic calibration is prepared for the same 10-pair sample.

## Goal

Evaluate whether `nectec/pathumma-llm-4b-think-4.0.0` can improve Thai Words data quality as a development-time data judge/enricher, without replacing V2.5 retrieval and without changing any production ranking path.

The first experiment calibrates Pathumma against already-consumed historical human labels. It does not auto-promote model output to Gold.

## First experiment

Pathumma receives only:

- query word / sense / definition / intended meaning / category;
- candidate word / sense / definition;
- stable `pair_id`.

It does not receive:

- human annotations;
- V2.5 score;
- V2.5 rank;
- V2.5 relation hints;
- learned/neural reranker output.

Candidate order is deterministically shuffled per query.

The model predicts:

- Writer Relevance utility 0..3;
- schema-v3 semantic relation;
- schema-v3 style tags;
- confidence;
- short reason;
- optional exploratory mood/usage tags.

The experiment reports agreement with human Gold:

- valid-output coverage;
- utility accuracy / MAE;
- relation accuracy;
- useful/high-utility binary agreement;
- severe-error detection.

## Leakage policy

Only historical/consumed labeled data may be used for this development experiment.

The runner rejects Phase-5 fresh/acceptance holdout paths. The fresh Phase-5 holdout remains reserved for the Production-v1 one-shot acceptance process in `plans/phase5-production-v1-2026-09-16.md`.

## Gold / Silver policy

Pathumma output is experimental/synthetic evidence only.

- Gold: human-approved labels.
- Silver: may be created later only after a separate acceptance rule is defined and validated.
- Pathumma predictions from this experiment must never overwrite human annotations.

## Colab smoke test

Notebook:

`notebooks/pathumma_writer_enrichment_colab.ipynb`

Runner:

`scripts/pathumma_writer_enrichment.py`

The first smoke test uses:

- model: `nectec/pathumma-llm-4b-think-4.0.0`;
- 4-bit bitsandbytes loading on CUDA;
- 2 historical queries;
- 5 candidates per query;
- blind candidate order;
- deterministic greedy decoding by default (sampled decoding is opt-in);
- final JSON parsed after `</think>`;
- resumable output by `pair_id`.

Start small before scaling because the model card warns that reasoning traces can run long.

## Current artifact availability

GitHub API and path-history checks confirmed that these approved JSONL sources are not committed anywhere in this repository history:

- `evaluation/writer_relevance_50_annotations.approved.jsonl`
- `evaluation/writer_relevance_phase4_holdout_annotations.approved.jsonl`
- `evaluation/writer_relevance_phase4_holdout_annotations.jsonl`

Their manifests/approved label overlay remain present. The Colab notebook therefore supports manual upload of an approved historical JSONL so this experiment does not block Phase-5 Wave A artifact recovery.

## Implementation checkpoint — 2026-09-18

Added:

- `scripts/pathumma_writer_enrichment.py`
- `tests/test_pathumma_writer_enrichment.py`
- `notebooks/pathumma_writer_enrichment_colab.ipynb`

Safeguards implemented:

- blind prompt excludes Gold labels and V2.5 scoring/ranking hints;
- deterministic candidate shuffle;
- schema-v3 relation/style validation;
- severe-error relations cannot carry positive utility;
- malformed output is recorded instead of silently corrected;
- fresh Phase-5 holdout filename guard;
- Pathumma output is written to separate experiment artifacts;
- reasoning traces are not persisted unless explicitly requested.

Repository state at this checkpoint:

- branch is ahead of its Phase-5 Wave-A base;
- no PR opened;
- no GitHub workflow is configured/running automatically for this branch;
- first Colab GPU/model compatibility smoke run passed successfully.

## Acceptance for the smoke test

Do not scale to the full historical pool until:

- the model loads on the selected Colab GPU;
- final JSON can be parsed reliably;
- schema-rule violations are visible rather than silently corrected;
- latency is practical enough for batch enrichment;
- Gold labels remain isolated from the model prompt.

After the first report, evaluate Pathumma separately for:

1. relation/utility pre-labeling;
2. style/register enrichment;
3. hard-negative discovery;
4. candidate-gap discovery;
5. conflict/QA triage.

No PR should be opened without explicit user approval.

## First GPU smoke result — 2026-09-18

Input:

- `writer_relevance_phase4_holdout_annotations.approved.jsonl`
- SHA256: `7d719f22bf7834ab24bfacd91b3535871f05aa5db1e9273579e0e76a8f7c204c`
- 2 queries × 5 candidates = 10 pairs
- 4-bit Pathumma load succeeded; 10/10 predictions parsed; no failed batch/fatal error

Observed metrics from prompt v1 / sampled decoding:

- utility exact accuracy: 0.80
- utility MAE: 0.20
- utility within ±1: 1.00
- useful binary precision/recall/F1: 1.00 / 0.80 / 0.889
- high-utility binary precision/recall/F1: 1.00 / 1.00 / 1.00
- exact relation accuracy: 0.30
- severe-error precision/recall/F1: 0.50 / 1.00 / 0.667

Interpretation:

- utility / useful-vs-noise triage is promising enough to continue calibration;
- relation taxonomy is not reliable enough for automatic Gold labeling;
- severe-error recall was high but the model over-called severe relations;
- prediction inspection showed prompt anchoring: some outputs copied the old reason placeholder and zero confidence from the concrete JSON example.

## Calibration revision — prompt v2

Before scaling, the runner was revised to:

- remove concrete output values/placeholders from the prompt schema;
- define and contrast `weak_related`, `scene_context`, `effect_state`, `manner_action`, `sense_mismatch`, `unrelated`, and `unclear` explicitly;
- state that utility 0 does not imply a severe relation;
- make deterministic greedy decoding the default; sampled decoding is opt-in with `--sample`;
- reject empty/old-placeholder reasons;
- persist query/candidate text with predictions;
- report relation and utility confusion matrices;
- report style exact/Jaccard agreement;
- report severe-error false positives with words and reasons directly.

Notebook v5 keeps the 10-pair smoke test and adds a separate calibration run:

- 10 historical queries;
- 30 candidates per query;
- 300 pairs total;
- 10 candidates per prompt;
- separate `pilot300` prediction/raw/report artifacts;
- no Phase-5 fresh acceptance data.

The 300-pair run has not yet been executed.

## Prompt v2 deterministic smoke result — 2026-09-18

Same historical input / same selected 10 pairs as prompt v1:

- valid prediction coverage: 1.00
- utility exact accuracy: 0.40
- utility MAE: 1.20
- utility within ±1: 0.70
- exact relation accuracy: 0.50
- useful binary precision/recall/F1: 0.60 / 0.60 / 0.60
- high-utility binary precision/recall/F1: 0.50 / 0.50 / 0.50
- severe-error precision/recall/F1: 0.60 / 1.00 / 0.75
- style exact/Jaccard: 0.00 / 0.05

Interpretation:

- relation exact accuracy improved from 0.30 → 0.50, but utility quality regressed sharply from 0.80 → 0.40;
- because prompt and decoding changed together, the utility regression cannot be attributed to deterministic decoding alone;
- the pair-level predictions exposed two source-evidence failure modes that are more important than the aggregate score:
  1. multi-gloss dictionary entries can be misread by over-weighting a later historical example; `แห้ง → บก` was Gold direct/utility 3 but predicted unrelated/0 even though the definition begins `แห้ง, พร่อง, ลดลง`;
  2. cross-reference-only definitions can trigger hallucinated semantics; `หัวด้วน` and `หัวบัว` had definitions only like `ดูใน หัว ๑`, yet the model invented flower meanings and predicted direct/utility 3;
- `เทียนดอก` was Gold weak-related/utility 1 but predicted sense-mismatch/0, showing that the model still tends to collapse weak association into severe mismatch;
- style/register output is not useful in the combined semantic prompt and is removed from the next calibration.

Decision: **do not run the 300-pair pilot yet.**

## Evidence-aware semantic calibration — prompt v3

The next 10-pair calibration changes the task structure rather than merely adding more wording:

- semantic judge only; style/register/mood/usage are removed from the prompt;
- raw dictionary definition is normalized and HTML is stripped for evidence checks;
- `definition_head` extracts the gloss before `เช่น` so a later historical example cannot erase an explicit head gloss;
- `definition_status` marks `substantive`, `cross_reference_only`, or `missing`;
- cross-reference-only/missing definitions are constrained to `unclear` + utility 0;
- the model must return `evidence_status` and an `evidence_quote` copied from the candidate definition;
- output validation rejects evidence quotes not present in the source definition;
- deterministic greedy decoding remains the default for reproducibility;
- the 300-pair notebook step is removed until this evidence-aware 10-pair calibration improves enough to justify scaling.

The goal of prompt v3 is not to maximize relation labels by instruction alone; it is to make every semantic decision auditable against source text and prevent unsupported dictionary completion.
