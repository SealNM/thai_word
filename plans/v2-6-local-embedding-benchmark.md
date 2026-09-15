# V2.6 — Local Multilingual Embedding Benchmark

Status: **10-query pilot complete; keep V2.5 EmbeddingGemma as retrieval baseline**

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

### 2. Snowflake Arctic Embed l v2.0 / 256d

Model:
- `Snowflake/snowflake-arctic-embed-l-v2.0`
- Apache-2.0
- multilingual model with Thai explicitly covered in its language set
- native 1024d
- official Matryoshka 256d representation

Official retrieval format:
- query prefix: `query: `
- documents: no prefix

The first benchmark uses the official 256d Matryoshka representation.

## Implementation

- [x] Add `qwen3-embedding-0.6b-256` profile to `thai_dense_v2.py`.
- [x] Add Thai Words-specific Qwen3 retrieval instruction.
- [x] Keep Qwen instructions on query embeddings only.
- [x] Add `arctic-embed-l-v2-256` profile.
- [x] Use Arctic-L's official `query: ` prefix.
- [x] Keep documents unprefixed for both challengers.
- [x] Use 256d Matryoshka truncation for a fair V2.5 comparison.
- [x] Reuse the existing V2 dense-index builder.
- [x] Reuse the existing V2 weighted-RRF evaluator.
- [x] Add no-model unit tests in `tests/test_dense_v26_local.py`.
- [x] Run local profile tests.
- [x] Build Qwen3 256d index.
- [x] Build Arctic-L 256d index.
- [x] Run shared 10-query evaluation against V2.5.
- [ ] V3 holdout deferred: no challenger won the 10-query pilot materially.
- [x] Record build time; query-ranking pilot completed. CPU query latency may be recorded separately if needed.
- [ ] If Arctic-L 256d wins materially, optionally compare its native 768d representation.
- [x] Neither challenger beats V2.5 consistently; keep EmbeddingGemma and return focus to ranking/reranking.

## Artifact paths

```text
artifacts/v2/embeddinggemma-300m-256
artifacts/v26/qwen3-embedding-0.6b-256
artifacts/v26/arctic-embed-l-v2-256
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
  --batch-size 8 \
  --device cuda
```

No Gemini API key is required. The Hugging Face model is public; an HF token is optional for download rate limits.

## Build Arctic

```bash
python -u scripts/build_dense_index.py \
  --model arctic-embed-l-v2-256 \
  --output artifacts/v26/arctic-embed-l-v2-256 \
  --batch-size 32 \
  --device cuda
```

Arctic-L uses the native XLM-RoBERTa implementation and does not require `trust_remote_code=True`.

## Shared evaluation

```bash
python -u scripts/evaluate_v2.py \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --dense-index artifacts/v26/qwen3-embedding-0.6b-256 \
  --dense-index artifacts/v26/arctic-embed-l-v2-256 \
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


### Arctic second Colab failure: unpadding path

After disabling optional xFormers memory-efficient attention, the first encode batch still failed with a CUDA device-side gather assertion. The published Arctic config keeps `unpad_inputs=true`; its remote GTE implementation then enters a pad/unpad path even when memory-efficient attention is disabled.

For the Colab/local benchmark profile, disable both optimized flags together:

```python
config_kwargs={
    "use_memory_efficient_attention": False,
    "unpad_inputs": False,
}
```

This keeps standard padded attention end-to-end and avoids the incompatible gather/pad path. Model weights, query/document formatting, and 256d Matryoshka truncation remain unchanged.

Operational note: a CUDA device-side assertion poisons the active CUDA context. Restart the Colab session/runtime before retrying after this error.


## Arctic-M V2 compatibility decision

The medium V2 checkpoint (`Snowflake/snowflake-arctic-embed-m-v2.0`) is removed from the primary benchmark path after two reproducible Colab CUDA failures on the first batch:
1. its custom remote GTE code initially required optional xFormers because the published config enables memory-efficient attention;
2. after disabling that path, the model still triggered a CUDA index/gather device-side assertion at batch 0, including with batch size 32.

This is not an OOM signal and not a corpus-row-specific failure; it happens immediately on the first batch. The checkpoint's own Sentence Transformers metadata was produced with an older stack (Sentence Transformers 2.7.0.dev0, Transformers 4.39.3, PyTorch 2.1.0+cu121), while the current Colab benchmark environment uses a much newer stack. Rather than pinning the whole notebook to an old runtime or continuing to patch custom remote code, the benchmark substitutes `Snowflake/snowflake-arctic-embed-l-v2.0`.

Why Arctic-L:
- same Arctic Embed 2.0 multilingual family;
- Thai is explicitly included in its 74-language model card;
- Apache-2.0;
- official 256d Matryoshka results;
- native XLM-RoBERTa implementation, so no `trust_remote_code=True`;
- similar non-embedding compute class (~303M non-embedding parameters), while being operationally much more portable.

The medium profile remains in code only as an experimental/reference profile and is no longer required for the V2.6 decision.


## Arctic-L 256d persistent build result

Fresh Colab T4 build completed successfully and is stored in the persistent artifact path.

Observed metadata:
- model: `Snowflake/snowflake-arctic-embed-l-v2.0`
- profile: `arctic-embed-l-v2-256`
- rows: 52,004 dictionary senses
- dimensions: 256
- dtype: float32
- embedding file size: ~50.785 MiB
- model load: ~40.086s
- encoding: ~482.202s (~8m 02s)
- batches: 1,626 at batch size 32
- throughput: ~3.37 batches/s
- device: cuda:0
- normalized embeddings: yes
- max sequence length: 512
- native Transformers implementation; no remote-code config overrides

Relative build speed on the same Colab T4:
- V2.5 EmbeddingGemma: 259.965s
- Arctic-L: 482.202s (~1.85x slower than V2.5)
- Qwen3: 1,326.543s (~5.10x slower than V2.5)

Arctic-L is therefore ~2.75x faster to encode than Qwen3 in this setup while producing the same 256d artifact size. Retrieval quality remains the deciding factor.


## 10-query pilot result and decision

The shared hybrid benchmark compared:
- V2.5 `embeddinggemma-300m-256`
- V2.6 `qwen3-embedding-0.6b-256`
- V2.6 `arctic-embed-l-v2-256`

All runs used the same V1 lexical index, selected query senses, 300-entry lexical/dense candidate pools, weighted RRF, and protected lexical-tier sort. Only the dense embedding model changed.

### Qualitative result by query

| Query | Best / notable result | Notes |
| --- | --- | --- |
| ฝน | EmbeddingGemma ≈ Qwen3 | Both keep core rain terms high; Arctic-L drifts toward rarer `วัสนะ / พรรษ / เผลียง`. |
| โกรธ | EmbeddingGemma | Puts common `โมโห` first; challengers do not add a clear quality gain. |
| เดิน | Arctic-L | Stronger walking-specific tail (`เดินเหิน / คลาไคล`), although V2.5 remains competitive. |
| สวย | Qwen3 / Arctic-L | `งดงาม` rises to #4 vs #8 in V2.5, but none surfaces the desired common `งาม` in top 10. |
| มืด | Arctic-L slight | `มืดมน` enters top 10; otherwise all three remain dominated by the same lexical tier. |
| รัก | Arctic-L slight | `รักใคร่ / จงรัก / ผูกพัน` are useful, but the tail still contains weak items. Qwen3 introduces noun/associated forms such as `ที่รัก / ดวงใจ`. |
| พูด | EmbeddingGemma | Most natural/common substitutes overall (`จา / เว้า / เจรจา / กล่าว / ออกปาก`). |
| เร็ว | Arctic-L | Clear local win: `รวดเร็ว / ฉับไว` rank well. Qwen3 regresses badly with `เข้า / ช้า / หน่อย`. |
| กลัว | EmbeddingGemma ≈ Qwen3 | No material improvement from switching models. |
| บ้าน | No model solves the target | All three are still dominated by `ภูม / เวศม์ / นิเวศ / วาสะ...`; none promotes `เรือน / บ้านเรือน / บ้านช่อง` into top 10. |

### Decision

Do **not** replace V2.5 EmbeddingGemma with Qwen3 or Arctic-L.

Reasons:
1. No challenger wins consistently across the ten Thai dictionary queries.
2. Qwen3 has several useful local improvements but also explicit semantic regressions (for example `เร็ว → ช้า`) and is ~5.1x slower to build than V2.5 on T4.
3. Arctic-L is operationally better than Qwen3 and wins some queries (`เดิน`, `เร็ว`), but regresses on others (`ฝน`) and is still ~1.85x slower to build than V2.5.
4. The most important failure, `บ้าน`, is unchanged by switching embeddings.

### Bottleneck diagnosis

The pilot confirms that the remaining problem is primarily **ranking after candidate retrieval**, not lack of semantic candidates.

`HybridSearcher.search()` builds the candidate set from the union of the top 300 lexical and top 300 dense entries, but final ordering sorts first by `protected_relation_tier`, then weighted-RRF score, lexical score, and dense similarity. A standalone lexical relation with tier >= 4 therefore outranks lower-tier candidates before dense quality is considered.

This explains why replacing only the dense model changes some middle/tail ordering but does not fix high-priority cases such as `บ้าน`. The next iteration should therefore focus on:
- lexical-validity / grammatical-role gating;
- commonness after semantic validity;
- reranking a small candidate pool (Gemma 4 direct remains the strongest tested direction);
- revisiting protected-tier policy only if it prevents common valid substitutes from surfacing.

V3 holdout is not required for choosing between these three embeddings because neither challenger produced a material pilot win. Keep the artifacts for future ensemble/recall experiments, but preserve V2.5 EmbeddingGemma as the retrieval baseline.
