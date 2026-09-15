# Thai Words — Phase 4 Runtime-Compatible Writer Reranker Plan

Date: 2026-09-16  
Status: **In progress — Waves A-G complete; Wave H targets frozen, V2.5 candidate export pending**  
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
