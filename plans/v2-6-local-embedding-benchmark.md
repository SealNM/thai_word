# V2.6 — Local Multilingual Embedding Benchmark

Status: **Qwen3 256d index built successfully; Arctic build + shared benchmark pending**

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
- [x] Run local profile tests.
- [x] Build Qwen3 256d index.
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


## Qwen3-Embedding-0.6B 256d build result

Real Colab T4 build completed successfully after restarting the runtime, capping sequence length to 512, and using batch size 8.

Observed metadata:
- rows: 52,004 dictionary senses
- dimensions: 256
- dtype: float32
- embedding file size: ~50.785 MiB
- model load: ~23.47s
- encoding: ~1,208.94s (~20m 09s)
- total batches: 6,501
- throughput: ~5.38 batches/s at batch size 8
- device: cuda:0
- normalized embeddings: yes
- query instruction: Thai Words-specific lexical substitution / grammatical-role preservation instruction
- document template: `{headword}: {definition}`

Important operational note:
- The earlier CUDA OOM was caused by a dirty Colab runtime that still had ~14.3 GiB of T4 VRAM in use from previous model experiments.
- After restarting the runtime, VRAM returned to 0 MiB / 15,360 MiB and the Qwen3 build completed cleanly.
- Disk space (~37 GiB free at the time) was not the cause of the failure.


## Local profile test result

Colab test suite `tests/test_dense_v26_local.py` passed 4/4:
- Arctic official query prefix
- Qwen instruction-aware 256d profile
- challenger documents remain unprefixed
- Qwen instruction applied to query only


## V2.5 baseline rebuild result

Fresh Colab T4 rebuild completed successfully and is stored in the persistent artifact path.

Observed metadata:
- model: `google/embeddinggemma-300m`
- profile: `embeddinggemma-300m-256`
- rows: 52,004 dictionary senses
- dimensions: 256
- dtype: float32
- embedding file size: ~50.785 MiB
- model load: ~41.323s
- encoding: ~259.965s (~4m 20s)
- batches: 813 at batch size 64
- throughput: ~3.13 batches/s
- device: cuda:0
- normalized embeddings: yes
- native query/document methods: `encode_query` / `encode_document`


## Qwen3 256d persistent rebuild result

Fresh Colab T4 rebuild completed successfully and is stored in the persistent artifact path.

Observed metadata:
- model: `Qwen/Qwen3-Embedding-0.6B`
- profile: `qwen3-embedding-0.6b-256`
- rows: 52,004 dictionary senses
- dimensions: 256
- dtype: float32
- embedding file size: ~50.785 MiB
- model load: ~34.638s
- encoding: ~1,326.543s (~22m 07s)
- batches: 6,501 at batch size 8
- throughput: ~4.90 batches/s
- device: cuda:0
- normalized embeddings: yes
- max sequence length: 512
- query instruction: Thai Words lexical-substitution / grammatical-role instruction
- documents remain unprefixed

Relative to the fresh V2.5 EmbeddingGemma baseline, Qwen3 encoding is about 5.1x slower on the same T4 (1,326.543s vs 259.965s). Quality must therefore improve materially to justify replacing V2.5.


## Arctic Colab compatibility note

Initial Arctic load failed on Colab because the model config enables `use_memory_efficient_attention=true`, and its remote GTE implementation asserts that `xformers` must be installed when that path is enabled.

For the benchmark we do not install `xformers`. The Arctic profile now passes:

```python
config_kwargs={"use_memory_efficient_attention": False}
```

to Sentence Transformers. This uses the model's standard attention path and avoids an unnecessary optional dependency on Colab. The semantic model weights, 256d Matryoshka truncation, query prefix, and document formatting are unchanged.
