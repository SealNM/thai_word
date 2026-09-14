# V2.6 — Local Multilingual Embedding Benchmark

Status: **implementation ready; Colab build + benchmark pending**

Branch: `feat/dictionary-semantic-v26-local-embedding-benchmark`

Baseline:
- V2.5 lexical + dense hybrid
- `google/embeddinggemma-300m`
- 256-dimensional Matryoshka embeddings
- preferred V2.5 checkpoint: `3884e3a81ef9ed50fbd13ce62e0eaa7e2858c83c`

## Goal

Test newer free/local multilingual embedding models before changing the downstream reranking architecture.

The comparison must keep the V2.5 lexical search, weighted RRF, query sense selection, and dictionary data unchanged. Only the dense embedding model changes.

Primary question:

> Can a newer local embedding model improve the candidate pool enough that common, valid Thai lexical substitutes appear earlier, while preserving V2.5 semantic precision?

## Models

### 1. Qwen3-Embedding-0.6B / 256d

Model:
- `Qwen/Qwen3-Embedding-0.6B`
- Apache-2.0
- 0.6B parameters
- 100+ languages
- up to 1024 dimensions with Matryoshka truncation
- instruction-aware retrieval

For Thai Words, use a custom English retrieval instruction because Qwen recommends task-specific instructions and notes that its training instructions are primarily English:

```text
Instruct: Given a Thai dictionary query with its selected sense, retrieve Thai dictionary entries that can substitute for the query while preserving meaning and grammatical role
Query:{query}
```

Documents remain unprefixed:

```text
{headword}: {definition}
```

The first benchmark uses 256d for a direct footprint comparison with V2.5.

### 2. Snowflake Arctic Embed m v2.0 / 256d

Model:
- `Snowflake/snowflake-arctic-embed-m-v2.0`
- Apache-2.0
- multilingual model with Thai explicitly covered in its language set
- native 768d
- official Matryoshka 256d representation

Official retrieval format:
- query prefix: `query: `
- documents: no prefix

The first benchmark uses the official 256d Matryoshka representation.

## Implementation

- [x] Add `qwen3-embedding-0.6b-256` profile to `thai_dense_v2.py`.
- [x] Add Thai Words-specific Qwen3 retrieval instruction.
- [x] Keep Qwen instructions on query embeddings only.
- [x] Add `arctic-embed-m-v2-256` profile.
- [x] Use Arctic's official `query: ` prefix.
- [x] Keep documents unprefixed for both challengers.
- [x] Use 256d Matryoshka truncation for a fair V2.5 comparison.
- [x] Reuse the existing V2 dense-index builder.
- [x] Reuse the existing V2 weighted-RRF evaluator.
- [x] Add no-model unit tests in `tests/test_dense_v26_local.py`.
- [ ] Run local profile tests.
- [ ] Build Qwen3 256d index.
- [ ] Build Arctic 256d index.
- [ ] Run shared 10-query evaluation against V2.5.
- [ ] Inspect V3 holdout only after the 10-query pilot.
- [ ] Record build time, model load time, and query latency.
- [ ] If Arctic 256d wins materially, optionally compare its native 768d representation.
- [ ] If neither challenger beats V2.5, keep EmbeddingGemma and return focus to Gemma 4 reranking.

## Artifact paths

```text
artifacts/v2/embeddinggemma-300m-256
artifacts/v26/qwen3-embedding-0.6b-256
artifacts/v26/arctic-embed-m-v2-256
```

## Colab test

```bash
python -m unittest discover -s tests -p "test_dense_v26_local.py" -v
```

## Build Qwen3

```bash
python -u scripts/build_dense_index.py \
  --model qwen3-embedding-0.6b-256 \
  --output artifacts/v26/qwen3-embedding-0.6b-256 \
  --batch-size 32 \
  --device cuda
```

No Gemini API key is required. The Hugging Face model is public; an HF token is optional for download rate limits.

## Build Arctic

```bash
python -u scripts/build_dense_index.py \
  --model arctic-embed-m-v2-256 \
  --output artifacts/v26/arctic-embed-m-v2-256 \
  --batch-size 64 \
  --device cuda
```

Arctic uses `trust_remote_code=True` because the official model repository supplies its GTE model implementation.

## Shared evaluation

```bash
python -u scripts/evaluate_v2.py \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --dense-index artifacts/v26/qwen3-embedding-0.6b-256 \
  --dense-index artifacts/v26/arctic-embed-m-v2-256 \
  --top-k 10 \
  --device cuda \
  --output artifacts/v26/local-embedding-benchmark.json
```

## Decision criteria

Inspect the same ten benchmark queries first.

High-value signals:
- `บ้าน`: `เรือน / บ้านเรือน / บ้านช่อง` appear earlier.
- `สวย`: `งาม / งดงาม` improve without broad unrelated adjectives.
- `เร็ว`: `ไว / รวดเร็ว / ด่วน` improve without rare forms dominating.
- `เดิน`: walking substitutes remain ahead of `วิ่ง`.
- `ฝน`: rain substitutes remain ahead of generic weather associations.
- `รัก`: verb/emotion substitutes remain ahead of nouns and associated concepts.
- `มืด`: darkness/light adjectives remain ahead of conditions such as blindness.

Do not choose a winner from generic MTEB scores alone. Thai Words' dictionary benchmark and holdout are the decision source.
