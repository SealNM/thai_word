# Thai Words — Phase 4 Runtime-Compatible Writer Reranker Plan

Date: 2026-09-16  
Status: **In progress — Waves A-F runtime integration implemented; Wave G performance gate pending**  
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
