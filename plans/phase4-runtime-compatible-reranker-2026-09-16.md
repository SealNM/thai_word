# Thai Words — Phase 4 Runtime-Compatible Writer Reranker Plan

Date: 2026-09-16  
Status: **Phase 4 complete — Waves A-I closed; no post-holdout model promotion**  
Base commit: `959c502254529e7260fdbf98a615b0e4e7858145`  
Working branch: `feat/phase4-runtime-compatible-reranker-2026-09-16`

## Phase 3 decision carried forward

Phase 3 is complete and must not be reopened for model selection.

Selected architecture:

```text
V2.5 candidate retrieval
        |
        v
top-30 candidate senses
        |
        +--> learned writer-utility / severe-error signal
        |
        +--> fine-tuned BAAI/bge-reranker-v2-m3
        |
        v
within-query rank normalization
        |
        v
locked blend: 0.5 learned + 0.5 neural
        |
        v
final ranking
```

Frozen benchmark result for the locked hybrid:

- Useful@10: **0.99**
- HighUtility@10: **0.94**
- Noise@10: **0.01**
- SevereError@10: **0.01**
- relation diversity: **2.8**
- NDCG@10: **0.857411**
- MRR high utility: **1.0**

The Phase 3 benchmark is now consumed.

**Do not** use its 10 queries to:
- select a new alpha;
- choose a different model;
- tune learning rate / epochs;
- change feature weights;
- accept or reject a new runtime representation.

Any future model-selection claim requires a new holdout.

---

## Why Phase 4 cannot simply plug the Phase 3 checkpoint into `search_v2.py`

Phase 3 learned and neural inputs contain a field that the normal runtime does not currently own:

```text
query.category
```

The Phase 3 learned baseline uses:

```text
query_category
```

and the CrossEncoder text pair contains:

```text
หมวด: <query.category>
```

But the production search contract today is:

```text
HybridSearcher.search(query, sense=...)
```

It has no authoritative category/POS input.

Therefore loading the Phase 3 model and silently replacing category with `<none>` would create an unbenchmarked input contract.

Phase 4 must first make the model contract match real runtime data.

---

# Phase 4 objectives

1. preserve V2.5 retrieval behavior;
2. introduce a reusable writer-reranker runtime layer;
3. make every model input available from normal search runtime;
4. persist all non-neural learned artifacts instead of fitting at request time;
5. load the 2GB neural checkpoint lazily and explicitly;
6. fail safely back to V2.5 if reranker artifacts are unavailable;
7. measure latency / RAM / VRAM before enabling by default;
8. create a new holdout before making any post-Phase-3 quality claim;
9. never tune against the consumed Phase 3 benchmark.

---

# Wave A — Runtime contract audit

## A1. Freeze the Phase 3 input schema

Document every field consumed by each signal.

### V2.5 retrieval evidence

Available directly from `HybridSearcher` results:

- `v25_rank` / original rank;
- V2.5 fusion score;
- relation tier;
- lexical score;
- lexical rank;
- dense similarity;
- dense rank;
- relation hint;
- lexical form;
- sense resolution;
- candidate definition;
- matched candidate sense.

### Query fields

Available:
- query word;
- selected query sense;
- selected query definition.

Not reliably available:
- `query.category`.

## A2. Add a runtime adapter

Create one canonical adapter, tentatively:

```text
thai_writer_reranker.py
```

Responsibilities:

- convert a selected query sense + V2.5 result into the exact runtime feature row;
- never read annotation fields;
- expose one shared representation to learned and neural scorers;
- assign deterministic `v25_rank` before reranking;
- reject malformed/missing retrieval evidence in strict mode.

No scoring logic should duplicate the Phase 3 row construction in multiple files.

## A3. Category-gap diagnostic

Run a **development diagnostic only** on the already-used train/validation data:

Compare:
1. Phase 3 inputs with category;
2. category replaced by `<none>`;
3. category field removed from learned features + neural text.

This diagnostic may guide implementation engineering, but it is **not a new unbiased benchmark**.

Preferred production direction if quality remains reasonable:

> remove query category entirely from the runtime model contract.

Reason:
- users should not have to specify POS/category;
- automatic category inference would add another model and another failure mode;
- dictionary data does not currently provide a guaranteed normalized category;
- self-contained query+sense+retrieval evidence is easier to reproduce.

If category removal materially changes ranking on development data, stop and create the new holdout before choosing a replacement design.

---

# Wave B — Persist the learned scorer

The Phase 3 learned baseline is currently refit inside evaluation scripts.

That is not a production runtime.

Create a persisted learned-ranker artifact containing:

- `DictVectorizer`;
- ordinal cumulative models for utility thresholds 1 / 2 / 3;
- severe-error probability model;
- seed;
- severe penalty;
- feature schema version;
- training dataset hash;
- model metadata.

Suggested artifact layout:

```text
artifacts/writer-reranker/
├── learned_ranker.joblib
└── learned_ranker_metadata.json
```

Requirements:

- no fitting during a search request;
- deterministic artifact build command;
- metadata must state whether category is part of the feature contract;
- loader must reject incompatible schema versions;
- scoring output should expose expected utility, severe probability, and safe score for diagnostics.

---

# Wave C — Neural runtime wrapper

Create a small neural scorer around the saved CrossEncoder checkpoint.

External checkpoint:

```text
artifacts/phase3/bge-reranker-v2-m3-locked/
```

The checkpoint should **not** be committed to normal Git history.

Runtime configuration should support:

```text
THAI_WORD_WRITER_RERANKER_MODEL_PATH
```

or an explicit CLI argument.

Requirements:

- lazy model load;
- CPU and CUDA device selection;
- configurable neural batch size;
- no model download during a normal request unless explicitly allowed;
- clear error when checkpoint is absent;
- optional warmup method;
- report model path, device, and load time;
- use the same text-pair builder as training/runtime contract.

---

# Wave D — Production ranking wrapper

Introduce a writer-aware wrapper around V2.5 instead of modifying V2.5 retrieval internals.

Tentative API:

```python
searcher = WriterSearch.from_paths(
    lexical_index="artifacts/v1",
    dense_index="artifacts/v2/embeddinggemma-300m-256",
    learned_ranker="artifacts/writer-reranker",
    neural_model="artifacts/phase3/bge-reranker-v2-m3-locked",
)

results = searcher.search(
    "ฝน",
    sense=1,
    top_k=10,
    rerank_pool=30,
)
```

Flow:

```text
V2.5 top-30
 -> runtime adapter
 -> learned score
 -> neural score
 -> within-query average-rank normalization
 -> fixed alpha 0.5
 -> stable tie-break using original V2.5 rank
 -> return top-k
```

Important:
- V2.5 remains candidate retrieval;
- reranker must not introduce candidates outside the V2.5 pool;
- default `rerank_pool=30` because Phase 3 was trained/evaluated on 30-candidate pools;
- alpha remains **0.5** until a future model-selection cycle with a new holdout;
- include original V2.5 rank and Phase 4 score in diagnostic output.

---

# Wave E — Failure behavior and feature flag

Do not make the heavy reranker an unconditional dependency immediately.

Support modes:

```text
off
optional
required
```

### off
Return V2.5 exactly.

### optional
Attempt writer reranking.
If learned/neural artifact loading fails, return V2.5 and expose:

```json
{
  "reranker_status": "fallback_v25",
  "reranker_error": "..."
}
```

### required
Artifact/model failure raises an error.

Initial default:

> **optional or off**, not required.

Only change the product default after runtime latency/resource checks pass.

---

# Wave F — CLI integration

Extend `scripts/search_v2.py` or add a separate `scripts/search_writer.py`.

Prefer a new command initially so V2.5 behavior remains unchanged.

Suggested usage:

```bash
python scripts/search_writer.py "ฝน" \
  --index artifacts/v1 \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --learned-ranker artifacts/writer-reranker \
  --neural-model artifacts/phase3/bge-reranker-v2-m3-locked \
  --sense 1 \
  --top-k 10 \
  --rerank-pool 30 \
  --device cuda
```

CLI output should include:

- final rank;
- original V2.5 rank;
- hybrid score;
- learned normalized score;
- neural normalized score;
- raw learned safe score;
- raw neural score;
- candidate definition / matched sense;
- relation evidence;
- reranker status.

Diagnostic fields can later be hidden in a user-facing web API.

---

# Wave G — Runtime performance gate

The 2GB CrossEncoder can dominate latency and memory.

Measure separately:

1. V2.5 search latency;
2. first neural model load;
3. warm neural query latency;
4. 30-pair CrossEncoder scoring time;
5. total reranked search time;
6. peak RAM;
7. peak VRAM on CUDA;
8. CPU fallback latency.

Record at least:
- CPU;
- one Kaggle/Colab-class GPU;
- intended deployment hardware when known.

Do not optimize by changing model/alpha against the consumed Phase 3 benchmark.

Permitted engineering optimizations:
- batching;
- lazy load;
- model process reuse;
- caching identical query+sense requests;
- safe dtype/device settings;
- request queueing.

Architecture/model compression or distillation is a **new model iteration** and needs a new holdout.

---

# Wave H — New Phase 4 holdout

Before any post-Phase-3 model change is promoted, create a new untouched holdout.

Recommended target:

- **20 new target senses**
- **30 V2.5 candidates each**
- **600 annotated pairs**
- zero query-sense overlap with the original 50 targets

Suggested coverage:
- 5 nouns / concrete-scene concepts;
- 5 verbs / actions;
- 5 adjectives / states;
- 5 emotions / abstract concepts.

Selection rules:
- freeze query list before evaluating the new ranker;
- pin senses explicitly;
- avoid choosing only known hard/easy cases;
- do not include the old Phase 3 benchmark queries;
- annotation schema remains Writer Relevance schema v3.

The new holdout is for the next model-selection claim.

The old 50-query dataset may be treated as development/training data after Phase 3 is closed, but the old benchmark result must remain archived and never be presented as an unbiased evaluation of a model retrained on those labels.

---

# Wave I — Production training policy

Two distinct artifacts must be named clearly.

## I1. Phase 3 locked benchmark artifact

The exact checkpoint that produced the frozen benchmark result.

Do not overwrite it.

```text
bge-reranker-v2-m3-locked
```

## I2. Future production-trained artifact

If we later retrain using more of the 50 already-labeled queries, store it under a different path/version.

Example:

```text
bge-reranker-v2-m3-production-v1
```

Never claim the old frozen benchmark score for a newly retrained production artifact.

A fresh holdout is required for that model.

---

# Test coverage

Add focused tests for:

1. runtime adapter never exposes human annotation labels;
2. adapter preserves V2.5 rank/evidence;
3. learned artifact save/load round trip;
4. feature-schema mismatch rejection;
5. neural text pair matches the declared runtime contract;
6. fixed alpha = 0.5;
7. rank normalization is per query;
8. tie-break falls back to original V2.5 rank;
9. reranker cannot add candidates outside top-N V2.5 pool;
10. optional mode falls back exactly to V2.5;
11. required mode fails on missing artifact;
12. `--list-senses` never loads the neural model;
13. deterministic learned artifact build;
14. no Phase 3 benchmark file is referenced by runtime selection logic.

---

# Acceptance gate for first implementation

Phase 4 runtime integration is considered structurally complete when:

- V2.5 search remains unchanged when reranking is off;
- learned artifact can be built and loaded without fitting at request time;
- saved BGE checkpoint loads from an explicit external path;
- top-30 V2.5 results can be reranked with fixed alpha 0.5;
- missing neural model cleanly falls back in optional mode;
- no human annotation field enters inference;
- category dependency is resolved explicitly, not silently;
- runtime performance report exists;
- tests pass.

Quality promotion beyond the exact Phase 3 contract requires the new Phase 4 holdout.

---

# Recommended implementation order

1. **A — runtime contract audit + category diagnostic**
2. **B — learned scorer artifact**
3. **C — neural runtime wrapper**
4. **D — WriterSearch wrapper**
5. **E/F — fallback modes + CLI**
6. **G — latency/resource profiling**
7. **H — new 20-query holdout**
8. only then consider category-free retraining, distillation, smaller reranker, or a production-retrained model

This order avoids making a 2GB experimental artifact a hard runtime dependency before we know the actual interface and deployment cost.


## Wave A implementation checkpoint — 2026-09-16

Implemented on `feat/phase4-runtime-compatible-reranker-2026-09-16`:

- `thai_writer_runtime.py`
  - canonical annotation-free adapter from one V2.5 result into the writer-reranker inference row;
  - preserves the retrieval evidence used by Phase 3;
  - exposes category modes `include`, `none`, and `omit`;
  - provides shared learned-feature and neural text-pair builders;
  - validates V2.5 rank and candidate shape.

- `scripts/writer_relevance_phase4_category_diagnostic.py`
  - train/validation-only development diagnostic;
  - compares learned scorer behavior for category included / forced to `<none>` / omitted;
  - optionally evaluates the saved Phase 3 CrossEncoder checkpoint under the same three runtime input modes;
  - explicitly reports `benchmark_split_evaluated: false`;
  - never uses benchmark rows for model selection.

- `tests/test_writer_runtime_contract.py`
  - no human annotation leakage;
  - V2.5 retrieval evidence preservation;
  - category-mode behavior for learned and neural representations;
  - evidence remains unchanged when category is removed;
  - invalid mode/rank guards.

Next Wave A action:

Run the diagnostic with the externally archived Phase 3 checkpoint. This is development analysis only; do not treat its validation result as a new unbiased benchmark.

```bash
python -m scripts.writer_relevance_phase4_category_diagnostic \
  --input evaluation/writer_relevance_50_annotations.approved.jsonl \
  --model-path artifacts/phase3/bge-reranker-v2-m3-locked \
  --device cuda \
  --eval-batch-size 4 \
  --output evaluation/writer_relevance_phase4_category_diagnostic.json
```

Decision gate:

- if `omit` or `none` is close to the original category-aware contract, proceed toward a category-free persisted production scorer;
- if category removal materially harms validation behavior, do not invent or tune an automatic category classifier against the consumed Phase 3 benchmark; freeze a fresh Phase 4 holdout first.


## Wave A diagnostic result — category-free contract accepted

The train/validation-only category diagnostic completed successfully using the saved Phase 3 checkpoint.

No benchmark rows were evaluated or used for selection.

### Learned scorer

| Category mode | Feature count | Useful | HighUtility | Noise | Severe | Diversity | NDCG | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| include | 43 | 0.966667 | 0.933333 | 0.033333 | 0.033333 | 3.111111 | 0.889148 | 1.0 |
| none | 34 | 0.966667 | 0.944444 | 0.033333 | 0.033333 | 2.888889 | 0.891203 | 1.0 |
| omit | 33 | 0.966667 | 0.944444 | 0.033333 | 0.033333 | 2.888889 | 0.891203 | 1.0 |

### Saved neural checkpoint

| Category mode | Useful | HighUtility | Noise | Severe | Diversity | NDCG | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| include | 0.988889 | 0.988889 | 0.011111 | 0.011111 | 2.444444 | 0.902097 | 1.0 |
| none | 0.988889 | 0.988889 | 0.011111 | 0.011111 | 2.666667 | **0.908070** | 1.0 |
| omit | 0.988889 | 0.988889 | 0.011111 | 0.011111 | 2.666667 | 0.906976 | 1.0 |

V2.5 validation NDCG remains **0.862307**.

Decision:

> Use **category_mode=omit** as the Phase 4 production contract.

Reasons:

- normal runtime has no authoritative category field;
- removing category does not degrade Useful/Noise/Severe/MRR;
- learned NDCG slightly improves from **0.889148 -> 0.891203**;
- neural NDCG remains above the category-aware checkpoint (**0.906976 vs 0.902097**);
- `none` is numerically slightly higher for neural scoring, but it preserves a synthetic `หมวด: <none>` token that has no semantic meaning at runtime;
- `omit` is therefore the cleaner and more truthful production contract.

This is an engineering contract decision on already-used development data, not a new unbiased quality benchmark.

A compact result is archived in:

- `evaluation/writer_relevance_phase4_category_diagnostic_summary.json`

---

## Wave B implementation checkpoint — persisted learned scorer

Implemented:

- `thai_writer_learned.py`
  - category-free `LearnedWriterRanker`;
  - three cumulative writer-utility probability models;
  - severe-error probability model;
  - expected utility / severe probability / safe score output;
  - deterministic seed handling;
  - runtime schema and artifact version guards;
  - persisted `DictVectorizer` and fitted models with `joblib`;
  - metadata validation on load;
  - no fitting during inference.

- `scripts/build_writer_learned_ranker.py`
  - train-split-only artifact builder;
  - records source dataset SHA-256;
  - writes the production category contract as `omit`;
  - records feature names/count, seed, severe penalty, train pair/query counts.

- `tests/test_writer_learned_ranker.py`
  - fit/save/load round trip;
  - inference without annotation fields;
  - category excluded from production feature schema;
  - runtime-schema mismatch rejection;
  - same-seed deterministic score check;
  - rejection of category-aware production fit.

Artifact layout:

```text
artifacts/writer-reranker/
├── learned_ranker.joblib
└── learned_ranker_metadata.json
```

Build command:

```bash
python -m scripts.build_writer_learned_ranker \
  --input evaluation/writer_relevance_50_annotations.approved.jsonl \
  --output artifacts/writer-reranker \
  --seed 42 \
  --severe-penalty 1.0
```

Next gate:

1. run focused Wave A/B tests;
2. build the learned artifact;
3. load it back and score validation rows without fitting;
4. once the persisted artifact is verified, continue to Wave C neural runtime wrapper.


## Wave C implementation checkpoint — neural runtime wrapper

Implemented:

- `thai_writer_neural.py`
  - `NeuralWriterRanker` with lazy CrossEncoder loading;
  - explicit local checkpoint path or `THAI_WORD_WRITER_RERANKER_MODEL_PATH`;
  - runtime download disabled by default;
  - optional `allow_download=True` only for deliberate development use;
  - CPU/CUDA device pass-through;
  - configurable max length and batch size;
  - category-free neural input via the shared Wave A runtime contract;
  - finite-score/count validation;
  - `warmup()` method;
  - runtime metadata including resolved source, actual/requested device, and model load seconds.

- `tests/test_writer_neural_ranker.py`
  - verifies true lazy loading;
  - environment variable model-path fallback;
  - missing checkpoint does not silently download;
  - explicit download opt-in;
  - category is omitted from runtime neural text;
  - warmup path;
  - empty input never loads the model.

Production behavior:

```text
explicit model_path
    ↓
THAI_WORD_WRITER_RERANKER_MODEL_PATH
    ↓
error if no local checkpoint
```

Normal production inference must not contact Hugging Face automatically.

Recommended verification using the archived Phase 3 checkpoint:

```bash
python -m unittest discover \
  -s tests \
  -p 'test_writer_neural_ranker.py' \
  -v
```

Then:

```python
from thai_writer_neural import NeuralWriterRanker

ranker = NeuralWriterRanker(
    "artifacts/phase3/bge-reranker-v2-m3-locked",
    device="cuda",
    batch_size=4,
)

print(ranker.runtime_info())  # loaded=false
print(ranker.warmup())        # loaded=true + load/device metadata
```

Next implementation wave:

> Wave D — combine V2.5, persisted learned scorer, and lazy neural scorer into `WriterSearch` with fixed alpha 0.5 and a strict top-30 candidate boundary.


## Wave D implementation checkpoint — WriterSearch

Implemented:

- `thai_writer_search.py`
  - `WriterSearch` wraps V2.5 without modifying `HybridSearcher`;
  - loads persisted learned ranker via `LearnedWriterRanker.load()`;
  - uses lazy `NeuralWriterRanker`;
  - default and locked `rerank_pool=30`;
  - fixed `alpha=0.5` enforced in code;
  - uses the same average-rank normalization as Phase 3 via `scipy.stats.rankdata(method="average")`;
  - learned and neural scores are normalized within the current query pool;
  - hybrid score is `0.5 * learned_rank + 0.5 * neural_rank`;
  - exact ties fall back to original V2.5 rank;
  - cannot introduce candidates outside the V2.5 top-N pool;
  - carries original V2.5 result metadata through to final results;
  - adds writer-reranker diagnostics without requiring annotation fields.

Diagnostic output fields include:

- `final_rank`
- `original_v25_rank`
- `writer_hybrid_score`
- `writer_learned_rank_score`
- `writer_neural_rank_score`
- `writer_learned_safe_score`
- `writer_expected_utility`
- `writer_severe_probability`
- `writer_neural_score`
- `writer_alpha`
- `reranker_status=writer_reranked`

The wrapper deliberately refuses any alpha other than **0.5**. A new alpha is a model-selection decision and requires a new holdout.

- `tests/test_writer_search.py`
  - fixed alpha guard;
  - exact Phase 3 rank-normalization direction;
  - average-rank ties;
  - strict V2.5 pool boundary;
  - equal locked blend;
  - V2.5 rank tie-break;
  - annotation-free runtime rows;
  - diagnostic output;
  - rerank-pool size guard.

Expected runtime flow is now:

```text
query
  -> HybridSearcher V2.5 top-30
  -> runtime_row_from_v25_result
  -> persisted LearnedWriterRanker
  -> lazy NeuralWriterRanker
  -> per-query average-rank normalization
  -> alpha 0.5 blend
  -> V2.5 tie-break
  -> top-k
```

Recommended focused test:

```bash
python -m unittest discover \
  -s tests \
  -p 'test_writer_search.py' \
  -v
```

After this passes, Wave E/F can add explicit `off / optional / required` fallback behavior and the separate `scripts/search_writer.py` CLI without altering V2.5's existing CLI.


## Waves E/F implementation checkpoint — fallback modes + writer CLI

Wave D focused tests passed **9/9** on Kaggle.

Observed real checkpoint warmup before Wave E/F:

- checkpoint: `artifacts/phase3/bge-reranker-v2-m3-locked`
- requested device: `cuda`
- actual device: `cuda:0`
- category mode: `omit`
- runtime download: disabled
- first model load component: **13.910393 s**

This timing is carried forward as a Wave G runtime baseline, not a quality metric.

### Wave E — fallback modes

`WriterSearch` now supports:

- `off` — raw V2.5 output; learned/neural rerankers are not required;
- `optional` — attempt reranking, but return the unchanged V2.5 result ordering on artifact/scoring failure while exposing status/error on the searcher/CLI envelope;
- `required` — reranker failures propagate as errors.

Default for the new writer CLI is `optional`.

The `off` path is evaluated before rerank-pool validation, so even `top_k > rerank_pool` behaves exactly like V2.5 because rerank_pool is irrelevant when reranking is disabled.

### Wave F — writer CLI

Added `scripts/search_writer.py` without modifying `scripts/search_v2.py`.

Example:

```bash
python scripts/search_writer.py "ฝน" \
  --index artifacts/v1 \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --learned-ranker artifacts/writer-reranker \
  --neural-model artifacts/phase3/bge-reranker-v2-m3-locked \
  --sense 1 \
  --top-k 10 \
  --rerank-pool 30 \
  --reranker-mode optional \
  --neural-device cuda \
  --include-runtime
```

CLI envelope reports:

- query/sense;
- requested reranker mode;
- `reranker_status`;
- `reranker_error`;
- results;
- optional runtime metadata.

`--list-senses` exits before `WriterSearch.from_paths`, so it cannot construct/load the writer reranker.

Focused tests cover:

- raw V2.5 equality in off mode;
- off mode ignoring rerank-pool constraints;
- optional runtime fallback;
- optional initialization fallback;
- required error propagation;
- required component guard;
- successful writer status;
- invalid mode rejection;
- list-senses bypassing WriterSearch construction;
- CLI default mode = optional.

Next: Wave G runtime profiling with real artifacts. No ranking parameter/model changes are permitted as part of that profiling.


## Wave G implementation checkpoint — runtime profiler

Added:

- `scripts/profile_writer_runtime.py`
- `tests/test_writer_runtime_profile.py`

The profiler does **not** evaluate quality and does not reference the consumed Phase 3 benchmark for selection.

It measures:

1. startup time excluding lazy neural weights;
2. V2.5 top-N search latency;
3. persisted learned-score latency;
4. first full writer search latency (cold neural path);
5. neural model load seconds reported by `NeuralWriterRanker`;
6. warm 30-pair neural scoring latency across repeated runs;
7. warm full writer-search latency across repeated runs;
8. process peak RSS;
9. CUDA allocated/reserved and peak allocated/reserved memory;
10. cold result words for sanity only.

The profiler records:

```json
{
  "quality_selection_performed": false,
  "phase3_benchmark_used": false
}
```

GPU example:

```bash
python -m scripts.profile_writer_runtime "ฝน" \
  --index artifacts/v1 \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --learned-ranker artifacts/writer-reranker \
  --neural-model artifacts/phase3/bge-reranker-v2-m3-locked \
  --sense 1 \
  --top-k 10 \
  --rerank-pool 30 \
  --repeats 5 \
  --dense-device cuda \
  --neural-device cuda \
  --output evaluation/writer_relevance_phase4_runtime_profile_gpu.json
```

CPU comparison:

```bash
python -m scripts.profile_writer_runtime "ฝน" \
  --index artifacts/v1 \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --learned-ranker artifacts/writer-reranker \
  --neural-model artifacts/phase3/bge-reranker-v2-m3-locked \
  --sense 1 \
  --top-k 10 \
  --rerank-pool 30 \
  --repeats 3 \
  --dense-device cpu \
  --neural-device cpu \
  --output evaluation/writer_relevance_phase4_runtime_profile_cpu.json
```

The previously observed neural checkpoint load component of **13.910393 s** remains a useful comparison point, but the profiler should be treated as the canonical Wave G measurement.

Wave G decision must be based on runtime/resource practicality only. Do not change alpha/model architecture from these measurements. If the 2GB reranker is operationally too expensive, compression/distillation becomes a future model iteration requiring the fresh Phase 4 holdout.


### Kaggle base-artifact preflight incident

The first end-to-end `search_writer.py` run on a fresh Kaggle session failed before V2.5 retrieval:

```text
FileNotFoundError: artifacts/v1/entries.json
```

Cause:

- writer reranker artifacts and the V1/V2.5 retrieval artifacts are separate;
- the current Kaggle session had the learned/neural writer artifacts, but not `artifacts/v1`;
- `optional` mode can only fall back to V2.5 when V2.5 itself is available.

This is an environment/artifact setup issue, not a writer-ranker regression.

CLI hardening:

- `scripts/search_writer.py` now preflights the base retrieval files before constructing `WriterSearch`;
- required lexical files:
  - `entries.json`
  - `senses.json`
  - `metadata.json`
- required dense files:
  - `dense_metadata.json`
  - `dense_embeddings.npy`
- a missing base artifact now returns concise deterministic rebuild commands rather than a low-level traceback;
- the CLI intentionally does not auto-build the dense index because building embeddings may be expensive and should remain explicit;
- `--list-senses` checks only the lexical artifact and still bypasses dense/neural loading.

Canonical rebuild:

```bash
python scripts/build_index.py --output artifacts/v1

python scripts/build_dense_index.py \
  --index artifacts/v1 \
  --model embeddinggemma-300m-256 \
  --output artifacts/v2/embeddinggemma-300m-256 \
  --device cuda
```

The writer learned artifact and Phase 3 neural checkpoint do not need to be rebuilt.


### EmbeddingGemma gated-access incident on Kaggle

The next Kaggle rebuild step failed while loading the unchanged V2.5 dense model:

```text
GatedRepoError: 401 Unauthorized
google/embeddinggemma-300m
```

Verified cause:

- `google/embeddinggemma-300m` is a gated Hugging Face model;
- its repository metadata is public, but file access requires accepting Google's usage license;
- the runtime must authenticate with a Hugging Face account that has accepted the license;
- Hugging Face supports authentication through the `HF_TOKEN` environment variable or `hf auth login`.

This is an access/authentication incident, not a model or code-quality regression.

Policy:

> Do **not** swap to another embedding model to bypass the gate. V2.5 was selected with `embeddinggemma-300m-256`; changing the embedding model would change the retrieval baseline.

Build-script hardening:

- `scripts/build_dense_index.py` now recognizes gated/401 model-access failures;
- it emits a concise license/authentication instruction instead of a large HTTP traceback;
- it reports whether `HF_TOKEN` is set without ever printing the token value;
- it explicitly reminds operators that the V2.5 rebuild must remain on `embeddinggemma-300m-256`.

Kaggle recovery flow:

1. Open `https://huggingface.co/google/embeddinggemma-300m` while signed in.
2. Accept Google's usage license for EmbeddingGemma.
3. Create/use a Hugging Face read token for that same account.
4. Put it in a private Kaggle secret, preferably named `HF_TOKEN`.
5. Export the secret into the process environment before invoking the build command.
6. Optionally verify authentication with `hf auth whoami`.
7. Re-run the exact V2.5 dense-index build; no retraining or model-selection step is involved.


### EmbeddingGemma dense-index rebuild completed

After accepting the EmbeddingGemma license and authenticating the Kaggle runtime, the exact V2.5 dense index rebuild completed successfully.

Observed build result:

- model key: `embeddinggemma-300m-256`
- model id: `google/embeddinggemma-300m`
- query method: `encode_query`
- document method: `encode_document`
- truncate dim: **256**
- normalized embeddings: **true**
- rows / indexed senses: **52,004**
- dimensions: **256**
- dtype: `float32`
- dense embedding artifact bytes: **53,252,096**
- dense embedding artifact size: **50.785 MiB**
- requested device: `cuda`
- actual device: `cuda:0`
- model load: **22.886 s**
- document encoding: **267.416 s**
- text template: `{headword}: {definition}`

Interpretation:

- V1/V2.5 base retrieval artifacts are now available in the Kaggle runtime;
- the rebuild preserved the locked V2.5 model/profile and did not change ranking architecture;
- the gated-access incident is resolved;
- no retraining or benchmark selection occurred.

Next gate:

1. run the full `search_writer.py` path with `reranker_mode=optional`;
2. confirm `reranker_status=writer_reranked` rather than `fallback_v25`;
3. if successful, run the existing Wave G profiler on the same GPU runtime and archive the resulting runtime report.


## Wave G result — GPU runtime gate completed

The full writer-search path completed successfully on Kaggle with:

```text
reranker_status = writer_reranked
reranker_error = null
```

The end-to-end search used:

- V1 lexical artifacts;
- V2.5 `embeddinggemma-300m-256` dense retrieval;
- persisted category-free learned writer ranker;
- saved Phase 3 `BAAI/bge-reranker-v2-m3` checkpoint;
- fixed alpha **0.5**;
- top-30 rerank pool;
- CUDA device `cuda:0`.

The observed query `ฝน#1` returned a writer-reranked top 10 and demonstrated that candidates can move substantially inside the fixed V2.5 pool. This is a runtime integration sanity check only, not a quality-selection benchmark.

### GPU profiler result

Profile:

- query: `ฝน`
- sense: `1`
- top-k: **10**
- rerank pool: **30**
- repeats: **5**
- candidate count: **30**
- startup excluding lazy neural weights: **18.082087 s**
- V2.5 search: **0.386395 s**
- learned scorer: **0.002860 s**
- cold full writer search: **5.377671 s**
- neural model load component: **3.774303 s**
- warm neural 30-pair scoring:
  - mean **1.384402 s**
  - median **1.383754 s**
  - min **1.350166 s**
  - max **1.418807 s**
- warm full writer search:
  - mean **1.620064 s**
  - median **1.628537 s**
  - min **1.584697 s**
  - max **1.644806 s**
- peak process RSS: **4,619.375 MiB**
- CUDA devices available: **2**
- cold CUDA allocated: **3,352.776 MiB**
- cold CUDA reserved: **3,452.0 MiB**
- cold CUDA max allocated: **3,388.608 MiB**
- warm CUDA allocated: **3,352.776 MiB**
- warm CUDA reserved: **3,452.0 MiB**
- warm CUDA max allocated: **3,389.389 MiB**
- warm CUDA max reserved: **3,452.0 MiB**

Profiler safeguards confirmed:

```json
{
  "quality_selection_performed": false,
  "phase3_benchmark_used": false
}
```

### Runtime interpretation

The learned scorer is effectively negligible relative to neural inference.

The main ongoing cost is the CrossEncoder:

- V2.5 alone: about **0.39 s**
- warm full writer path: about **1.62 s**
- incremental warm cost over V2.5: about **1.23 s**
- neural scoring itself: about **1.38 s**

The warm path is practical for an explicit writer-oriented search request on a persistent GPU worker, but it is too heavy to run automatically on every keystroke/autocomplete event.

Operational decision for the first product integration:

- keep V2.5 for instant/live suggestions;
- expose writer reranking for explicit submitted searches or behind a feature flag;
- do not make the heavy reranker a hard availability dependency;
- keep the `optional` fallback path;
- keep a persistent/warm worker when neural reranking is enabled;
- do not change alpha/model/rerank-pool from these runtime measurements.

The CLI may remain `optional` by default for integration testing. A web/live-search endpoint should initially use V2.5 for live typing and invoke writer reranking only after explicit search submission.

A compact machine-readable summary is archived in:

- `evaluation/writer_relevance_phase4_runtime_profile_gpu_summary.json`

Wave G is complete.

Next:

> Wave H — create and freeze a new 20-query / 600-pair holdout before any model, compression, distillation, alpha, feature, or architecture iteration.


## Wave H checkpoint — fresh holdout headwords frozen

A fresh 20-target holdout headword list has been selected **before** Phase-4 candidate/model output is inspected.

The new set is stronger than the minimum no-query-sense-overlap rule: it has **zero headword overlap** with the original 50-target writer-relevance set.

### Frozen headword groups

#### noun / scene — 5

- `ทะเล`
- `ภูเขา`
- `แม่น้ำ`
- `ดอกไม้`
- `เงา`

#### verb / action — 5

- `กอด`
- `จูบ`
- `ก้ม`
- `หัน`
- `หลบ`

#### adjective / state — 5

- `เงียบ`
- `แห้ง`
- `หนัก`
- `เบา`
- `หวาน`

#### emotion / abstract — 5

- `คิดถึง`
- `หวัง`
- `หึง`
- `สงสัย`
- `กังวล`

Source:

- `evaluation/writer_relevance_phase4_holdout_targets.json`

Status of that file:

```text
headwords_frozen_senses_pending
```

The headword list must not be replaced because a candidate list looks weak/hard/easy.

### Holdout preparation workflow

Added:

- `scripts/writer_relevance_phase4_holdout.py`
- `tests/test_writer_relevance_phase4_holdout.py`

The workflow has three explicit stages:

```text
inspect
  -> human review of dictionary senses only
freeze
  -> immutable 20 target senses + hashes
export
  -> exactly 30 V2.5 candidates each = 600 unlabeled pairs
```

Safeguards:

- exactly **20** targets;
- exactly **5** targets in each of four groups;
- no duplicate new headwords;
- no headword overlap with `writer_relevance_50_targets.json`;
- sense IDs must exist in the V1 lexical artifact;
- target config hash is pinned during inspection/freeze;
- frozen target file hash is checked again before candidate export;
- export is exactly **30 V2.5 candidates × 20 = 600 pairs**;
- candidate rows use `split=phase4_holdout`;
- writer learned/neural/hybrid rerankers are not imported or used for target selection;
- candidate export uses V2.5 only;
- no quality metrics are calculated during preparation;
- export manifest explicitly records `writer_reranker_used=false` and `holdout_opened_for_model_evaluation=false`.

### Next gate — resolve dictionary senses

Run on the existing Kaggle runtime where `artifacts/v1` is already available:

```bash
python -m scripts.writer_relevance_phase4_holdout inspect \
  --config evaluation/writer_relevance_phase4_holdout_targets.json \
  --old-targets evaluation/writer_relevance_50_targets.json \
  --index artifacts/v1 \
  --output evaluation/writer_relevance_phase4_holdout_sense_report.json
```

For targets with exactly one dictionary sense, the script fills `recommended_sense` automatically.

For `needs_review` targets, review only the dictionary definitions and set `recommended_sense` in the sense report. Do **not** run candidate export or writer reranking before all 20 senses are pinned.

Only after that review should `freeze` be run.


## Wave H sense review — decisions locked

The 20-target sense inspection completed with:

- **20/20** headwords found;
- **8** unique-sense targets resolved automatically;
- **12** multi-sense targets reviewed manually;
- **0** missing headwords;
- no V2.5 candidate list, learned score, neural score, hybrid score, or quality metric was consulted during sense selection.

Locked decisions are stored in:

- `evaluation/writer_relevance_phase4_holdout_sense_decisions.json`

Pinned target senses:

```text
ทะเล#1
ภูเขา#1
แม่น้ำ#1
ดอกไม้#2
เงา#1

กอด#1
จูบ#1
ก้ม#1
หัน#1
หลบ#1

เงียบ#1
แห้ง#1
หนัก#1
เบา#1
หวาน#1

คิดถึง#1
หวัง#1
หึง#1
สงสัย#1
กังวล#1
```

Notable reviewed choices:

- `ทะเล#1` — the large salt-water body; sense 2 is adjectival/compound usage;
- `ดอกไม้#2` — the dictionary cross-reference to `ดอก ๑`; the other senses are not the plant-flower concept;
- `เงา#1` — the dark shape caused by blocking light;
- `หัน#1` — change from one direction to another;
- `หลบ#1` — avoid/evade; sense 2 is specifically hide;
- `เงียบ#1` — no sound / quiet;
- `แห้ง#1` — no water / not wet;
- `หนัก#1` — heavy by weight;
- `เบา#1` — light by weight;
- `หวาน#1` — sugar-like taste;
- `หึง#1` — romantic jealousy;
- `สงสัย#1` — uncertainty/doubt matching the intended concept.

The freeze command now accepts the locked decisions file directly and validates every chosen sense against the original sense-inspection report. This avoids manual edits to the generated report while preserving the original inspection evidence.

Next command:

```bash
python -m scripts.writer_relevance_phase4_holdout freeze \
  --config evaluation/writer_relevance_phase4_holdout_targets.json \
  --report evaluation/writer_relevance_phase4_holdout_sense_report.json \
  --decisions evaluation/writer_relevance_phase4_holdout_sense_decisions.json \
  --old-targets evaluation/writer_relevance_50_targets.json \
  --output evaluation/writer_relevance_phase4_holdout_frozen_queries.json \
  --manifest evaluation/writer_relevance_phase4_holdout_manifest.json
```

The freeze step records the decisions-file SHA-256 in the holdout manifest.

Only after freeze succeeds should V2.5 candidate export run.


## Wave H freeze complete — 20 target senses immutable

The reviewed sense decisions were applied to the inspected dictionary senses and the fresh holdout target set is now frozen **before candidate export**.

Canonical files:

- `evaluation/writer_relevance_phase4_holdout_targets.json` — headwords frozen before sense inspection;
- `evaluation/writer_relevance_phase4_holdout_sense_decisions.json` — reviewed query→sense decisions based only on dictionary definitions;
- `evaluation/writer_relevance_phase4_holdout_frozen_queries.json` — immutable 20 target senses;
- `evaluation/writer_relevance_phase4_holdout_manifest.json` — freeze hashes and leakage guards.

Frozen target-file SHA-256:

```text
f6c9fef35a556ecc306d67e42eea605ee85f804f21371595d22d1cca654fcd34
```

Freeze manifest records:

- target count: **20**
- candidates per query: **30**
- expected pair count: **600**
- original-50 headword overlap: **0**
- candidate exported: **false**
- writer reranker used for target selection: **false**
- writer reranker used for candidate export: **false**
- quality selection performed: **false**

The raw Kaggle sense-inspection output does not need to be edited manually; the committed sense-decision file and frozen target definitions are now the canonical reviewed decision artifacts. The inspection itself remains reproducible from the frozen headword list + V1 lexical artifact.

### Next gate — export exactly 600 V2.5 pairs

Run:

```bash
python -m scripts.writer_relevance_phase4_holdout export \
  --config evaluation/writer_relevance_phase4_holdout_frozen_queries.json \
  --manifest evaluation/writer_relevance_phase4_holdout_manifest.json \
  --index artifacts/v1 \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --device cuda \
  --output evaluation/writer_relevance_phase4_holdout_annotations.jsonl \
  --export-manifest evaluation/writer_relevance_phase4_holdout_export_manifest.json
```

Expected output:

- **20** query senses;
- **30** V2.5 candidates per sense;
- **600** rows total;
- labels all null;
- `split=phase4_holdout`;
- `writer_reranker_used=false`;
- `holdout_opened_for_model_evaluation=false`.

Do not run learned/neural/hybrid evaluation on the 600 rows before annotation is complete and the labeled holdout is frozen.


### Wave H freeze artifact integrity fix

The first committed `writer_relevance_phase4_holdout_frozen_queries.json` was malformed due to a file-generation error in the commit step, not due to the sense-review workflow.

Observed failures:

- JSON parse error near the `เงา` record;
- legacy unit-test namespace without `decisions` caused `AttributeError`.

Fixes:

- rebuilt the frozen target JSON from structured data using the reviewed sense decisions and inspected dictionary definitions;
- restored the intended frozen-target SHA-256:
  `f6c9fef35a556ecc306d67e42eea605ee85f804f21371595d22d1cca654fcd34`;
- changed `freeze_targets()` to read the optional decisions argument with `getattr(args, "decisions", None)`, preserving compatibility with older/internal callers and tests;
- manifest hash remains valid because it already recorded the intended structured frozen file hash;
- no target, sense decision, model choice, candidate list, or quality-selection policy changed.

This is an artifact-integrity/code-compatibility fix only.


## Wave H candidate export complete — 600 unlabeled pairs

The fresh holdout candidate export completed successfully on the frozen V2.5 artifacts.

Recorded export:

- target senses: **20**
- candidates per target: **30**
- pair count: **600**
- candidate system: **V2.5**
- candidate file: `evaluation/writer_relevance_phase4_holdout_annotations.jsonl`
- candidate file SHA-256:
  `93592bfaa38ade132f0699df856cd82e4c4f7e8c4dcf5f5f754073ed68d84328`
- writer reranker used: **false**
- quality selection performed: **false**
- labels present: **false**
- holdout opened for model evaluation: **false**

The canonical export record is:

- `evaluation/writer_relevance_phase4_holdout_export_manifest.json`

The main holdout manifest is now marked `candidate_exported=true`; the preparation script will refuse to silently overwrite/re-export the same frozen holdout.

### Annotation gate

No learned/neural/hybrid evaluation may run on these 600 pairs before annotation is complete and the labeled file is frozen.

Use the existing schema-v3 annotation tool, which autosaves and resumes:

```bash
python -m scripts.substitutability_benchmark annotate \
  evaluation/writer_relevance_phase4_holdout_annotations.jsonl
```

For a safer one-query-at-a-time workflow:

```bash
python -m scripts.substitutability_benchmark annotate \
  evaluation/writer_relevance_phase4_holdout_annotations.jsonl \
  --query ทะเล
```

or label at most one query-sized batch at a time:

```bash
python -m scripts.substitutability_benchmark annotate \
  evaluation/writer_relevance_phase4_holdout_annotations.jsonl \
  --limit 30
```

During annotation:

- judge writer usefulness independently of V2.5 rank;
- do not run writer learned/neural/hybrid scores for assistance;
- retain schema v3 utility / semantic relation / style tags;
- severe relations `opposite_misleading`, `sense_mismatch`, `unrelated` must remain utility 0;
- do not calculate model-comparison metrics yet.

After all 600 rows are labeled:

```bash
python -m scripts.substitutability_benchmark validate \
  evaluation/writer_relevance_phase4_holdout_annotations.jsonl
```

Only after full validation passes should the labeled holdout be frozen under a new SHA-256 and opened for a single controlled model-selection cycle.


## Chat handoff checkpoint — Phase 4 holdout annotation next

Date: 2026-09-16

Current branch:

```text
feat/phase4-runtime-compatible-reranker-2026-09-16
```

Current completed state:

- Waves **A-G complete**;
- Wave H target headwords frozen;
- all 20 target senses reviewed and frozen;
- zero headword overlap with the original 50-query dataset;
- V2.5 dense artifact rebuilt with `embeddinggemma-300m-256`;
- exactly **600 unlabeled holdout pairs** exported from V2.5 only;
- candidate export locked in manifest;
- writer learned/neural/hybrid rerankers have **not** been used to inspect or score the new holdout;
- no quality selection has been performed on the new holdout.

Canonical frozen target SHA-256:

```text
f6c9fef35a556ecc306d67e42eea605ee85f804f21371595d22d1cca654fcd34
```

Canonical 600-pair candidate-file SHA-256:

```text
93592bfaa38ade132f0699df856cd82e4c4f7e8c4dcf5f5f754073ed68d84328
```

Expected candidate file from Kaggle:

```text
evaluation/writer_relevance_phase4_holdout_annotations.jsonl
```

Important handoff note:

> Kaggle's interactive annotation CLI is not convenient for the user because it requires typing into the notebook terminal. The next chat should therefore continue by having the user upload the generated `writer_relevance_phase4_holdout_annotations.jsonl` file directly to ChatGPT.

Once that file is uploaded, the next implementation task is:

1. verify its SHA-256 equals
   `93592bfaa38ade132f0699df856cd82e4c4f7e8c4dcf5f5f754073ed68d84328`;
2. verify it contains exactly **600 rows**, **20 query IDs**, and **30 V2.5 candidates per query**;
3. annotate all rows using Writer Relevance schema v3:
   - `utility`: 0..3;
   - `semantic_relation`;
   - `style_tags`;
   - optional concise notes only where useful;
4. annotation must be based on query meaning + candidate meaning + writer usefulness only;
5. do **not** use learned/neural/hybrid scores or rankings to assist annotation;
6. enforce severe relations
   `opposite_misleading`, `sense_mismatch`, `unrelated`
   => utility **0**;
7. validate all 600 labeled rows;
8. create a separate approved/frozen labeled holdout artifact with a new SHA-256;
9. update the holdout manifest:
   - `labels_present=true`;
   - record labeled-file SHA-256;
   - keep `holdout_opened_for_model_evaluation=false` until validation/freeze completes;
10. only after the labeled holdout is frozen should the model-comparison/evaluation step be opened.

Do not ask the user to repeat the Phase 4 setup in the next chat; this checkpoint is the authoritative continuation point.

Files to read first in the next chat:

- `plans/phase4-runtime-compatible-reranker-2026-09-16.md`
- `evaluation/writer_relevance_phase4_holdout_manifest.json`
- `evaluation/writer_relevance_phase4_holdout_export_manifest.json`
- `evaluation/writer_relevance_phase4_holdout_frozen_queries.json`
- uploaded `writer_relevance_phase4_holdout_annotations.jsonl`

No PR has been opened. Opening any PR still requires explicit user approval.


## Wave H label freeze complete — 600 / 600 pairs immutable

Date: 2026-09-16

The uploaded V2.5 candidate export was verified before annotation:

- source SHA-256: `93592bfaa38ade132f0699df856cd82e4c4f7e8c4dcf5f5f754073ed68d84328`;
- exactly **600** rows;
- exactly **20** frozen query IDs;
- exactly **30** V2.5 candidates per query;
- all source labels were null before annotation.

All 600 rows are now annotated under Writer Relevance schema v3.

Annotation safeguards:

- decisions used query meaning + candidate meaning + writer usefulness only;
- candidate order was randomized during annotation review to reduce V2.5-rank bias;
- learned/neural/hybrid writer-reranker output was not opened or used;
- severe relations `opposite_misleading`, `sense_mismatch`, and `unrelated` all have utility **0**;
- source row content is unchanged except for the annotation object;
- validation completed with **0 errors**.

### Frozen label representation

To avoid committing a duplicate ~648 KB JSONL while keeping the result exactly reproducible, the canonical labels are stored as a source-hash-locked overlay:

- `evaluation/writer_relevance_phase4_holdout_labels.approved.json`
- overlay SHA-256: `1029ba5808a0fa0eba3cb82d9faf2f292f365db4b1791c20a8172676c236f561`

Materializer:

- `scripts/writer_relevance_phase4_labels_materialize.py`

The materializer refuses a source file whose SHA differs from the frozen V2.5 export, applies all 600 labels, runs schema-v3 validation, writes the approved JSONL, and verifies the exact final hash.

Expected materialized artifact:

- `evaluation/writer_relevance_phase4_holdout_annotations.approved.jsonl`
- SHA-256: `7d719f22bf7834ab24bfacd91b3535871f05aa5db1e9273579e0e76a8f7c204c`

Run:

```bash
python scripts/writer_relevance_phase4_labels_materialize.py \
  --source evaluation/writer_relevance_phase4_holdout_annotations.jsonl \
  --labels evaluation/writer_relevance_phase4_holdout_labels.approved.json \
  --output evaluation/writer_relevance_phase4_holdout_annotations.approved.jsonl
```

### Frozen label distribution

Utility:

- 3: **202**
- 2: **186**
- 1: **111**
- 0: **101**

Semantic relation:

- direct: **163**
- subtype: **101**
- broader_concept: **22**
- manner_action: **28**
- scene_context: **76**
- effect_state: **49**
- weak_related: **60**
- opposite_misleading: **6**
- sense_mismatch: **43**
- unrelated: **48**
- unclear: **4**

### Holdout gate remains closed

The manifest intentionally remains:

```text
holdout_opened_for_model_evaluation = false
```

This commit is the immutable label boundary. Do not train, tune alpha, alter features, select candidates, or choose a model against this holdout before the boundary is fixed.

Next controlled step:

1. materialize and hash-check the approved JSONL;
2. explicitly open the post-freeze evaluation gate in a separate step;
3. evaluate frozen V2.5 and the already-frozen writer rerankers;
4. record metrics without retraining, threshold tuning, alpha search, candidate reselection, or architecture changes on this holdout.


## Wave H post-freeze evaluation gate opened — 2026-09-16

Label boundary commit:

```text
2b5f398f0c5a7ac263e0bfd19ca4f0616870bedb
```

The 600-pair Phase-4 holdout remains immutable at:

- approved JSONL SHA-256: `7d719f22bf7834ab24bfacd91b3535871f05aa5db1e9273579e0e76a8f7c204c`;
- label overlay SHA-256: `1029ba5808a0fa0eba3cb82d9faf2f292f365db4b1791c20a8172676c236f561`.

The evaluation gate is now explicitly open in:

- `evaluation/writer_relevance_phase4_evaluation_gate.json`.

The gate locks:

- category mode = `omit`;
- alpha = **0.5**;
- rerank pool = **30**;
- normalization = within-query average rank;
- tie-break = original V2.5 rank;
- learned artifact training source / seed / severe penalty;
- the exact Phase-3 locked-candidate manifest Git blob;
- the saved `bge-reranker-v2-m3-locked` checkpoint identity.

Policy remains:

- no retraining;
- no threshold tuning;
- no alpha search;
- no candidate reselection;
- no architecture changes;
- one controlled evaluation cycle only.

### V2.5 baseline recorded before writer-reranker output was opened

On the frozen 20-query / 600-pair holdout at K=10:

- Useful@10: **0.915**
- HighUtility@10: **0.795**
- Noise@10: **0.085**
- SevereError@10: **0.085**
- relation diversity: **2.8**
- NDCG@10: **0.7801111077355579**
- MRR high utility: **0.9625**

Archived in:

- `evaluation/writer_relevance_phase4_v25_baseline_summary.json`

with SHA-256:

```text
bf6fd359c859d414f92d13d81ed18b719563f091cd9eb3740acb4db37bc856b9
```

No learned, neural, or hybrid Phase-4 result has been inspected yet.

### One-shot evaluator

Added:

- `scripts/writer_relevance_phase4_holdout_eval.py`
- `tests/test_writer_relevance_phase4_holdout_eval.py`

The evaluator scores the **exact frozen 600-candidate pool** instead of rerunning retrieval. This prevents candidate drift from being mixed into the reranker comparison.

Before reading/scoring the holdout it requires:

- explicit `--confirm-phase4-holdout`;
- exact approved-dataset SHA;
- exact Phase-3 manifest Git blob;
- learned-artifact metadata match;
- fixed alpha / pool / category contract;
- all no-tuning policy flags.

Focused gate/runtime tests passed before this checkpoint.

### Single controlled Kaggle run

Materialize the approved JSONL if needed:

```bash
python scripts/writer_relevance_phase4_labels_materialize.py \
  --source evaluation/writer_relevance_phase4_holdout_annotations.jsonl \
  --labels evaluation/writer_relevance_phase4_holdout_labels.approved.json \
  --output evaluation/writer_relevance_phase4_holdout_annotations.approved.jsonl
```

Then run exactly one controlled evaluation:

```bash
python scripts/writer_relevance_phase4_holdout_eval.py \
  --input evaluation/writer_relevance_phase4_holdout_annotations.approved.jsonl \
  --gate evaluation/writer_relevance_phase4_evaluation_gate.json \
  --phase3-manifest evaluation/writer_relevance_phase3_locked_candidate.json \
  --learned-ranker artifacts/writer-reranker \
  --neural-model artifacts/phase3/bge-reranker-v2-m3-locked \
  --device cuda \
  --eval-batch-size 4 \
  --k 10 \
  --include-per-query \
  --output evaluation/writer_relevance_phase4_holdout_evaluation_report.json \
  --confirm-phase4-holdout
```

After that run, archive the resulting JSON without tuning from its result, update the manifest as evaluation-complete, and close this holdout cycle before considering any retraining/distillation/architecture iteration.

## Wave H reproduction evaluation complete — holdout cycle closed

Date: 2026-09-16

The single controlled Phase-4 holdout evaluation completed on the exact frozen 20-query / 600-pair labeled holdout:

- approved holdout SHA-256: `7d719f22bf7834ab24bfacd91b3535871f05aa5db1e9273579e0e76a8f7c204c`;
- label boundary commit: `2b5f398f0c5a7ac263e0bfd19ca4f0616870bedb`;
- category mode: `omit`;
- alpha: **0.5**;
- rerank pool: **30**;
- evaluation count: **1**;
- no threshold tuning, alpha search, candidate reselection, or architecture changes were performed from the holdout result.

### Aggregate result at K=10

| System | Useful | HighUtility | Noise | Severe | Diversity | NDCG | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2.5 | 0.915 | 0.795 | 0.085 | 0.085 | 2.80 | 0.780111 | 0.9625 |
| Learned | 0.915 | 0.810 | 0.085 | 0.085 | 2.65 | 0.784920 | 0.9250 |
| Neural | 0.965 | 0.900 | 0.035 | 0.035 | 3.00 | 0.878771 | 1.0000 |
| Locked hybrid 0.5/0.5 | 0.945 | 0.870 | 0.055 | 0.055 | 2.65 | 0.864456 | 1.0000 |

Locked hybrid vs V2.5:

- Useful@10: **+0.030**;
- HighUtility@10: **+0.075**;
- Noise@10: **-0.030**;
- SevereError@10: **-0.030**;
- NDCG@10: **+0.084345**;
- MRR high utility: **+0.0375**;
- relation diversity: **-0.15**.

Per-query NDCG for the locked hybrid vs V2.5:

- wins: **12 / 20**;
- ties: **3 / 20**;
- losses: **5 / 20**;
- largest gains: `หนัก#1`, `หวาน#1`, `หึง#1`, `เงา#1`, `เบา#1`;
- largest regression: `ทะเล#1` (-0.138009 NDCG).

### Neural-only observation is not a selection decision

Neural-only is stronger than the locked hybrid in the aggregate Phase-4 metrics:

- NDCG: **0.878771 vs 0.864456**;
- Useful: **0.965 vs 0.945**;
- HighUtility: **0.900 vs 0.870**;
- Noise/Severe: **0.035 vs 0.055**;
- diversity: **3.00 vs 2.65**.

This observation **must not** be used to switch the product to neural-only from this holdout. Doing so would be post-holdout model selection. The locked alpha 0.5 contract remains unchanged for this completed evaluation cycle. Any future neural-only / alpha / model / distillation decision requires another fresh holdout.

### Artifact-recovery caveat

The Kaggle session containing the original external Phase-3 checkpoint was lost before this evaluation. The exact approved 50-query training data was restored by SHA, and the learned scorer was rebuilt deterministically. The neural checkpoint was then **reproduced from the already-frozen Phase-3 configuration**:

- base model: `BAAI/bge-reranker-v2-m3`;
- epochs: 1;
- batch size: 2;
- learning rate: 1e-5;
- max length: 384;
- seed: 42;
- train split only.

No Phase-4 holdout labels or results were used to train or change this configuration. However, the reproduced neural checkpoint is **not verified as byte-identical** to the original Phase-3 checkpoint because the original external artifact was unavailable and no binary checkpoint hash had been frozen.

Therefore this result is recorded as:

> **Phase-4 frozen-configuration reproduction evaluation**

and not as proof of the exact original checkpoint binary.

Compact immutable result:

- `evaluation/writer_relevance_phase4_holdout_evaluation_summary.json`
- SHA-256: `b54e7568bc1591e1abc256951cd2a95967f71bbc1461309ea43b38a9579d0c3e`

The raw full evaluation JSON was generated in Kaggle as `evaluation/writer_relevance_phase4_holdout_evaluation_report.json`; the compact repo summary preserves the aggregate metrics, policy controls, artifact-recovery note, and per-query comparison counts.

### Wave H closure

Wave H is closed with no post-holdout tuning.

Wave I has **not** started. If the next iteration changes the model or promotes neural-only, create a separately named production artifact and a new fresh holdout before making a new unbiased quality claim.

## Wave I complete — production artifact policy frozen

Date: 2026-09-16

Wave I closes Phase 4 without training or selecting another model from the consumed Phase-4 holdout.

Implemented:

- `evaluation/writer_relevance_artifact_registry.json`
  - records the historical Phase-3 locked benchmark identity;
  - records the Phase-4 checkpoint as a separate **reproduction** identity after session loss;
  - records that the original Phase-3 binary is no longer available and no full checkpoint binary SHA-256 had been frozen;
  - prevents the reproduction from inheriting the Phase-3 frozen benchmark metrics;
  - leaves production status as `not_created`.

- `thai_writer_artifact_policy.py`
  - validates immutable benchmark naming;
  - validates distinct reproduction naming;
  - rejects claims that the reproduced checkpoint is the exact Phase-3 binary;
  - reserves production names as `bge-reranker-v2-m3-production-v{N}`;
  - requires every future production quality claim to use a fresh holdout;
  - rejects reuse of the consumed Phase-4 holdout for future model selection.

- `scripts/validate_writer_artifact_policy.py`
  - validates the committed registry without loading model weights.

- `tests/test_writer_artifact_policy.py`
  - covers benchmark/reproduction identity separation;
  - prevents inherited Phase-3 benchmark claims;
  - enforces versioned production names;
  - enforces a fresh holdout for future production quality claims.

Important identity clarification:

- historical benchmark artifact: `bge-reranker-v2-m3-locked`;
- Phase-4 recovered/retrained binary: logical identity
  `bge-reranker-v2-m3-reproduction-2026-09-16`;
- the Kaggle evaluation temporarily used the old `bge-reranker-v2-m3-locked` directory name for compatibility with the pre-registered evaluator, but that path does **not** transfer the historical benchmark identity;
- future production artifacts must use a new path such as
  `bge-reranker-v2-m3-production-v1`.

No `production-v1` model is created in this wave. Creating or promoting one now from the observed Phase-4 results would be post-holdout model selection.

### Phase 4 final state

- Waves **A-I complete**;
- V2.5 retrieval remains unchanged;
- category-free writer runtime exists;
- learned/neural runtime artifacts are supported;
- fixed-alpha writer reranking is implemented;
- fallback/CLI/performance gates are implemented;
- the fresh Phase-4 holdout is consumed and closed;
- no post-holdout tuning or neural-only promotion was performed;
- future model changes require a new fresh holdout.

Phase 4 can now be closed. The next model-development cycle should start with a new plan/phase and a newly frozen holdout before comparing neural-only, another alpha, compression, distillation, or retraining choices.

