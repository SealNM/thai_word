# Thai Words — Phase 5 Production-v1 Plan

Date: 2026-09-16  
Status: **In progress — Wave A source recovery / historical-pool build**  
Base commit: `3dbe3a0bc8b58a7361c75f99546d71d527767271`  
Planning branch: `plan/phase5-production-v1-2026-09-16`

## Goal

Create the first separately named, reproducible production writer-reranker artifact:

```text
bge-reranker-v2-m3-production-v1
```

without selecting or tuning it on the consumed Phase-4 holdout.

Phase 5 must:

1. convert all already-consumed labels into a historical development/training pool;
2. perform model/recipe selection only on that historical pool;
3. freeze one Production-v1 recipe before opening a new holdout;
4. train the final Production-v1 artifact on the full historical pool;
5. evaluate it exactly once on a fresh untouched acceptance holdout;
6. package the accepted artifact with complete hashes/provenance;
7. preserve V2.5 fallback and the existing explicit-search deployment posture.

---

# Starting state carried forward from Phase 4

Phase 4 is closed.

Current facts:

- V2.5 remains the retrieval layer.
- Production runtime contract uses `category_mode=omit`.
- rerank pool is currently top-30 V2.5 candidates.
- WriterSearch, learned scorer, neural scorer, fallback modes, CLI, and runtime profiling exist.
- the consumed Phase-4 holdout contains 20 query senses × 30 candidates = 600 labeled pairs.
- the older approved Writer Relevance dataset contains 50 labeled query senses.
- the 50-query dataset and consumed 20-query Phase-4 holdout may now be used as historical development/training data.
- Phase-4 holdout must **not** be reused as an unbiased acceptance set.
- neural-only looked stronger than the locked 0.5/0.5 hybrid in the Phase-4 reproduction run, but that observation must not itself select Production-v1.
- the recovered Phase-4 neural checkpoint is a **reproduction**, not a verified byte-identical copy of the original Phase-3 locked checkpoint.
- a private Kaggle artifact dataset now preserves the reproduced neural checkpoint, learned ranker, evaluation files, and provenance from the completed Phase-4 run.

Artifact policy already frozen in:

```text
evaluation/writer_relevance_artifact_registry.json
```

Production artifact naming must follow:

```text
bge-reranker-v2-m3-production-v{N}
```

and every new unbiased production quality claim requires a fresh holdout.

---

# Non-negotiable leakage rules

## Historical pool

After Phase 4 is closed, the following may be used for development/training:

- original approved 50-query Writer Relevance dataset;
- consumed Phase-4 20-query holdout.

They become a **70-query historical pool** for Phase 5.

They may be used for:

- grouped cross-validation;
- model/recipe comparison;
- alpha comparison;
- training-schedule comparison;
- feature ablations;
- final Production-v1 training.

They must no longer be described as unbiased holdouts for Production-v1.

## Fresh Phase-5 acceptance holdout

The new Phase-5 holdout must not be used for:

- choosing neural-only vs hybrid;
- alpha search;
- epoch selection;
- learning-rate selection;
- severe-penalty selection;
- architecture selection;
- candidate-pool-size selection;
- feature selection;
- model compression/distillation decisions.

The Production-v1 recipe must be frozen before model output is inspected on this holdout.

---

# Wave A — Build and freeze the 70-query historical development pool

## A1. Materialize the two historical sources

Source 1:

```text
evaluation/writer_relevance_50_annotations.approved.jsonl
```

Expected known SHA-256:

```text
6e767583302a6df75c9b76d86fc73cbda150c98fbbe7c95b912a650a04a1a515
```

Source 2:

```text
evaluation/writer_relevance_phase4_holdout_annotations.approved.jsonl
```

Expected known SHA-256:

```text
7d719f22bf7834ab24bfacd91b3535871f05aa5db1e9273579e0e76a8f7c204c
```

The Phase-4 approved JSONL may be materialized from the frozen source + label overlay.

## A2. Normalize and combine

Create:

```text
evaluation/writer_relevance_phase5_historical_70.approved.jsonl
evaluation/writer_relevance_phase5_historical_70_manifest.json
```

Requirements:

- exactly 70 unique `query_id` values;
- preserve original query/candidate/annotation data;
- preserve source provenance per row;
- reject duplicate query IDs across the two source sets;
- validate Writer Relevance schema v3;
- severe-error relations still require utility 0;
- record pair count and per-source counts;
- freeze SHA-256.

## A3. Historical pool policy

After the combined SHA is frozen:

- all Phase-5 model selection uses this historical pool only;
- no fresh Phase-5 holdout is opened yet.

Acceptance for Wave A:

- source hashes verified;
- 70 unique queries;
- zero accidental duplicate query IDs;
- schema validation passes;
- combined dataset SHA recorded.

---

# Wave B — Grouped cross-validation harness

Model selection must split by `query_id`, never by individual candidate row.

Reason:

> candidates from one query share the same query context and must never appear across both train and validation folds.

## B1. Deterministic folds

Preferred initial design:

```text
5-fold GroupKFold by query_id
```

with a frozen fold-assignment file:

```text
evaluation/writer_relevance_phase5_cv_folds.json
```

Requirements:

- deterministic query grouping;
- every query appears in validation exactly once;
- no query overlap between train and validation within a fold;
- fold file SHA recorded;
- all recipes use the exact same folds.

If stratification by semantic family is practical without introducing subjective tuning, store semantic-family metadata separately and use it only to audit coverage, not to move queries after results are known.

## B2. CV metrics

At minimum record:

- Useful@10;
- HighUtility@10;
- Noise@10;
- SevereError@10;
- NDCG@10;
- MRR high utility;
- relation diversity;
- per-query NDCG;
- win/tie/loss vs V2.5.

Report:

- pooled out-of-fold metrics;
- fold mean;
- fold spread/standard deviation;
- per-query deltas.

---

# Wave C — Select the Production-v1 recipe on historical CV only

The Phase-4 result suggests neural-only deserves explicit evaluation, but does not select it.

## C1. Required candidate recipes

Compare at minimum:

1. **V2.5 baseline**
2. **learned-only**
3. **neural-only**
4. **locked hybrid alpha 0.5**

Optional hybrid alpha candidates may be included **only now**, before the fresh holdout:

```text
alpha = 0.25
alpha = 0.50
alpha = 0.75
```

Do not create a dense alpha sweep.

## C2. Frozen initial neural training family

Initial neural family remains:

```text
base model: BAAI/bge-reranker-v2-m3
category mode: omit
max length: 384
seed: 42
```

The first CV pass should keep the Phase-3 training recipe as the reference:

```text
epochs: 1
batch size: 2
learning rate: 1e-5
warmup ratio: 0.1
AMP: enabled
```

If training-schedule variants are explored, keep the search deliberately small and document every candidate **before** running CV.

Recommended maximum for Production-v1 preparation:

- epochs: 1 or 2;
- learning rate: 1e-5 or 2e-5;
- no architecture/model-family sweep in this phase unless the plan is amended before fresh-holdout creation.

## C3. Selection objective

Primary metric:

```text
NDCG@10
```

Safety constraints:

- SevereError@10 must not regress materially vs V2.5;
- Noise@10 must not regress materially vs V2.5;
- Useful@10 and HighUtility@10 should not trade large safety losses for small NDCG gains.

Tie-break order for recipes whose NDCG is effectively similar:

1. lower SevereError@10;
2. lower Noise@10;
3. higher HighUtility@10;
4. higher Useful@10;
5. higher relation diversity;
6. lower runtime/resource cost;
7. simpler production architecture.

The exact selection result and all CV results must be archived before Wave D.

---

# Wave D — Freeze the Production-v1 recipe before the new holdout

Create:

```text
evaluation/writer_relevance_phase5_production_v1_recipe.json
```

It must freeze:

- architecture: neural-only / hybrid / other selected historical-CV winner;
- base model;
- category mode;
- input text contract;
- historical dataset SHA;
- fold-assignment SHA;
- training hyperparameters;
- seed(s);
- rerank pool;
- alpha if applicable;
- rank normalization if applicable;
- tie-break policy;
- learned scorer configuration if applicable;
- exact code commit used to train;
- intended artifact name:
  `bge-reranker-v2-m3-production-v1`.

After this recipe file is frozen:

> do not change the recipe because of Phase-5 holdout results.

If a recipe change is needed later, create Production-v2 or a new candidate and a new holdout cycle.

---

# Wave E — Train and freeze the final Production-v1 artifact

Train on the **full frozen 70-query historical pool** using the frozen Wave-D recipe.

Output:

```text
artifacts/production/bge-reranker-v2-m3-production-v1/
```

Never overwrite:

```text
artifacts/phase3/bge-reranker-v2-m3-locked/
```

and never reuse the reproduction identity.

## E1. Required artifact provenance

Create:

```text
artifacts/production/bge-reranker-v2-m3-production-v1/PRODUCTION_MANIFEST.json
```

Record:

- artifact ID;
- role = `production_candidate`;
- training-data SHA;
- query count / pair count;
- recipe SHA;
- base model;
- all hyperparameters;
- seed;
- tokenizer/config identity;
- git commit;
- environment/package versions;
- CUDA / GPU identity when available;
- training timestamps;
- every model/config/tokenizer file size;
- SHA-256 for every output file.

Also create a deterministic directory-manifest hash.

This prevents a repeat of the Phase-3 situation where the exact binary identity could not later be proven.

## E2. Persist immediately

Before any acceptance evaluation:

- upload Production-v1 artifact + manifest to a private persistent store/Kaggle Dataset;
- verify uploaded files/hashes;
- do not rely on the active notebook session as the only copy.

---

# Wave F — Create and freeze a fresh Phase-5 acceptance holdout

Target size:

```text
30 new query senses × 30 V2.5 candidates = 900 labeled pairs
```

## F1. Coverage

Recommended balanced target families:

- 6 concrete/nature nouns;
- 6 people/object/place nouns;
- 6 actions/verbs;
- 6 states/adjectives;
- 6 emotions/abstract concepts.

## F2. Zero-overlap rule

Before candidate export:

- zero `query_id` overlap with all historical 70;
- preferably zero headword overlap unless a clearly different sense is intentionally justified;
- inspect and pin every query sense before export;
- freeze target list SHA.

## F3. Candidate export

Candidates must come from V2.5 only:

```text
top-30 per frozen query sense
```

No Production-v1 / learned / neural / hybrid score may influence:

- target selection;
- sense selection;
- candidate export;
- annotation ordering.

Freeze the exact 900-pair candidate source SHA.

## F4. Annotation

Use Writer Relevance schema v3:

- utility 0..3;
- semantic relation;
- style tags;
- concise note only when useful.

Safeguards:

- randomize candidate presentation order during annotation;
- annotate from query meaning + candidate meaning + writer usefulness only;
- severe relations
  `opposite_misleading`, `sense_mismatch`, `unrelated`
  require utility 0;
- no model output may be shown during annotation.

After 900/900 are complete:

- validate;
- freeze labels;
- freeze approved materialized JSONL SHA;
- create an immutable label-boundary commit.

---

# Wave G — One-shot Production-v1 acceptance evaluation

Only after Wave F is frozen may the acceptance gate open.

Evaluate exactly once on the fresh Phase-5 holdout.

Required systems:

1. V2.5;
2. Phase-4 reproduction/current writer contract for historical comparison;
3. Production-v1 frozen candidate.

Do not tune from these results.

## G1. Pre-register acceptance criteria before opening the holdout

Primary quality gate:

- Production-v1 NDCG@10 improvement vs V2.5: target **>= +0.05 absolute**.

Safety gates:

- SevereError@10 must be **<= V2.5**;
- Noise@10 must be **<= V2.5**;
- MRR high utility must not regress materially;
- no evidence of a broad systematic regression across semantic families.

Per-query regression guard:

- record every query where Production-v1 loses to V2.5;
- flag any query with NDCG delta <= -0.15 as a severe regression for manual review;
- acceptance must not silently ignore severe regressions even if aggregate metrics pass.

Secondary metrics:

- Useful@10;
- HighUtility@10;
- relation diversity;
- win/tie/loss counts;
- semantic-family breakdown.

If the candidate fails the pre-registered gate:

> do not tune it against the acceptance holdout.

Close the holdout as consumed, return to historical development data, create a new candidate/version, then create a new fresh holdout for the next unbiased acceptance claim.

---

# Wave H — Production packaging and rollout readiness

Only after Wave G acceptance passes.

## H1. Registry promotion

Update:

```text
evaluation/writer_relevance_artifact_registry.json
```

to:

```json
{
  "production_policy": {
    "status": "created",
    "active_artifact_name": "bge-reranker-v2-m3-production-v1"
  }
}
```

while preserving historical benchmark/reproduction records.

## H2. Runtime packaging

Production behavior:

- V2.5 retrieval remains unchanged;
- Production-v1 applies only after explicit submitted search unless a later latency study justifies live typing;
- lazy model load;
- warm worker reuse;
- optional fallback to exact V2.5;
- no automatic Hugging Face download in normal production;
- model path supplied explicitly or via production environment variable.

## H3. Performance gate

Profile the **exact accepted Production-v1 binary**.

Reference metrics:

- V2.5 search;
- cold load;
- warm 30-pair neural/rerank time;
- end-to-end explicit search;
- peak RSS;
- peak VRAM;
- CPU fallback behavior.

The current Phase-4 reproduction measured roughly:

- warm neural top-30: ~1.384 s;
- warm full writer search: ~1.620 s;
- peak RSS: ~4619 MiB;
- CUDA allocated: ~3353 MiB.

Production-v1 must not be enabled for live typing by default. Explicit-search deployment remains the initial product posture.

Engineering-only performance changes such as batching, process reuse, caching, safe dtype, and queueing are allowed after quality freeze as long as they do not alter ranking.

Model quantization/compression/distillation that can change ranking constitutes a new model candidate and requires a new quality-validation cycle.

## H4. Final release gate

Before declaring Production-v1 ready:

- artifact registry valid;
- Production-v1 file hashes verified from persistent storage;
- focused writer-runtime tests pass;
- artifact-policy tests pass;
- historical-CV report archived;
- fresh holdout manifest archived;
- one-shot acceptance report archived;
- performance report archived;
- V2.5 fallback smoke-tested;
- no consumed holdout used for post-result tuning;
- plan status updated to complete.

---

# Suggested Phase-5 artifacts

```text
evaluation/
├── writer_relevance_phase5_historical_70.approved.jsonl
├── writer_relevance_phase5_historical_70_manifest.json
├── writer_relevance_phase5_cv_folds.json
├── writer_relevance_phase5_cv_report.json
├── writer_relevance_phase5_production_v1_recipe.json
├── writer_relevance_phase5_holdout_targets.json
├── writer_relevance_phase5_holdout_frozen_queries.json
├── writer_relevance_phase5_holdout_manifest.json
├── writer_relevance_phase5_holdout_annotations.jsonl
├── writer_relevance_phase5_holdout_labels.approved.json
├── writer_relevance_phase5_holdout_annotations.approved.jsonl
└── writer_relevance_phase5_acceptance_report.json

artifacts/production/
└── bge-reranker-v2-m3-production-v1/
    ├── ...
    └── PRODUCTION_MANIFEST.json
```

Large model binaries remain outside normal Git history.

---

# Wave order

```text
A  historical 70-query pool
        ↓
B  grouped-CV harness
        ↓
C  recipe selection on historical data only
        ↓
D  freeze Production-v1 recipe
        ↓
E  train + hash + persist Production-v1
        ↓
F  fresh 30-query / 900-pair holdout
        ↓
G  one-shot acceptance evaluation
        ↓
H  registry promotion + production packaging
```

Do not reorder D/F/G in a way that exposes fresh-holdout model results before the recipe is frozen.

---

# Chat handoff checkpoint — start here next

Phase 4 is complete at:

```text
3dbe3a0bc8b58a7361c75f99546d71d527767271
```

Phase-5 planning branch:

```text
plan/phase5-production-v1-2026-09-16
```

Plan:

```text
plans/phase5-production-v1-2026-09-16.md
```

The next chat should **not** restart Phase-4 evaluation or recreate its checkpoint.

The user already created a private persistent Kaggle Dataset containing the Phase-4 persisted artifacts and confirmed the expected files are present. If GPU work is later needed, restore from that private Dataset rather than retraining from scratch.

## First task in the next chat

Start Wave A.

1. create an implementation branch from the Phase-5 plan branch after the user approves implementation;
2. recover/materialize the exact approved 50-query JSONL;
3. materialize the exact approved Phase-4 20-query JSONL from its frozen source + label overlay;
4. verify source SHA-256 values:
   - 50-query: `6e767583302a6df75c9b76d86fc73cbda150c98fbbe7c95b912a650a04a1a515`;
   - Phase-4 20-query: `7d719f22bf7834ab24bfacd91b3535871f05aa5db1e9273579e0e76a8f7c204c`;
5. inspect query IDs and prove zero duplicate query IDs across the two sources;
6. build the combined 70-query historical artifact + manifest;
7. validate schema v3 and severe-error constraints;
8. freeze combined SHA;
9. then proceed to Wave B grouped CV.

## Files to read first

```text
plans/phase5-production-v1-2026-09-16.md
evaluation/writer_relevance_artifact_registry.json
evaluation/writer_relevance_phase4_holdout_evaluation_summary.json
evaluation/writer_relevance_phase4_holdout_manifest.json
evaluation/writer_relevance_phase4_holdout_labels.approved.json
scripts/writer_relevance_phase4_labels_materialize.py
thai_writer_runtime.py
thai_writer_learned.py
thai_writer_neural.py
thai_writer_search.py
```

The 50-query approved JSONL is not guaranteed to exist in Git and may need to be restored from the user's existing persistent artifact source/File Library before Wave A can materialize the 70-query pool.

## Important process rules

- do not open any PR without explicit user approval;
- do not use the consumed Phase-4 holdout as a new unbiased benchmark;
- do not select Production-v1 from the fresh Phase-5 holdout;
- do not overwrite the historical Phase-3 locked identity;
- do not overwrite the Phase-4 reproduction identity;
- do not call any new model `production-v1` until Wave D has frozen the recipe;
- large model binaries stay out of Git;
- persist/hash the final Production-v1 artifact before opening the Phase-5 acceptance holdout.

No PR is opened as part of this planning branch.


---

# Wave A implementation checkpoint — 2026-09-16

Implementation branch:

```text
feat/phase5-wave-a-historical-70-2026-09-16
```

Completed in the implementation branch:

- added `scripts/writer_relevance_phase5_historical_pool.py`;
- source SHA-256 is verified before data is accepted;
- Writer Relevance schema v3 and severe-error/utility constraints are validated;
- 50-query and Phase-4 query IDs are required to be disjoint;
- source pair/query counts are frozen at 1500/50 and 600/20;
- combined target is frozen at 2100 pairs / 70 unique queries;
- original query/candidate/annotation payloads are preserved and Phase-5 source provenance is added per row;
- combined JSONL and manifest are written only after validation passes;
- added focused unit tests for provenance preservation, duplicate-query rejection, hash mismatch, severe-error validation, and historical-policy metadata.

Source recovery still required before Wave A can be accepted:

1. `evaluation/writer_relevance_50_annotations.approved.jsonl`
   - expected SHA-256: `6e767583302a6df75c9b76d86fc73cbda150c98fbbe7c95b912a650a04a1a515`
   - expected: 50 unique query IDs / 1500 pairs.
2. Exact Phase-4 source material:
   - preferred approved file: `evaluation/writer_relevance_phase4_holdout_annotations.approved.jsonl`
   - expected SHA-256: `7d719f22bf7834ab24bfacd91b3535871f05aa5db1e9273579e0e76a8f7c204c`
   - or the frozen unlabeled candidate source
     `evaluation/writer_relevance_phase4_holdout_annotations.jsonl`
     with SHA-256
     `93592bfaa38ade132f0699df856cd82e4c4f7e8c4dcf5f5f754073ed68d84328`;
     the approved label overlay is already committed and the existing materializer must reproduce the approved SHA exactly.

Do not proceed to Wave B until the real sources have been restored, the 70-query artifact has been materialized, and its combined SHA has been frozen.

No Phase-5 fresh acceptance holdout has been opened, and no Production-v1 recipe/model selection has been performed.
