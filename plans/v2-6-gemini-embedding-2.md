# V2.6 — Gemini Embedding 2 Retrieval Experiment

Status: **archived / not promoted; project baseline remains V2.5 EmbeddingGemma**

Branch: `feat/dictionary-semantic-v26-gemini-embedding-2`

Base checkpoint: V2.5 preferred baseline
`3884e3a81ef9ed50fbd13ce62e0eaa7e2858c83c`

## Why V2.6

V2.5 with `google/embeddinggemma-300m` remains the strongest local retrieval baseline.

Before adding more downstream reranking heuristics, test whether Google's newer retrieval model can improve the candidate pool itself. Do not use Gemma 4 hidden states as embeddings; use Google's current embedding model that is trained explicitly for retrieval:

- `gemini-embedding-2`
- stable Gemini API model
- supports 128–3072 output dimensions
- recommended dimensions include 768 / 1536 / 3072
- text-only retrieval guidance uses asymmetric formatting:
  - query: `task: search result | query: {content}`
  - document: `title: {title} | text: {content}`

## Experiment

Keep the V2.5 lexical side and weighted RRF unchanged.

Compare:

1. V2.5 baseline — EmbeddingGemma 300M / 256d
2. V2.6 parity — Gemini Embedding 2 / 256d
3. V2.6 recommended-size control — Gemini Embedding 2 / 768d

This separates:
- model-generation improvement at the same 256d footprint;
- additional quality from a larger recommended embedding size.

## Implementation

- [x] Add `thai_dense_v26.py`.
- [x] Add Gemini API key resolution from `GEMINI_API_KEY` or `GOOGLE_API_KEY`, including Colab Secrets.
- [x] Use official Gemini Embedding 2 retrieval text formatting.
- [x] Keep one vector per dictionary sense.
- [x] Store vectors in the existing dense artifact shape so V2 hybrid fusion can be reused.
- [x] L2-normalize returned vectors defensively before storing/searching.
- [x] Add resumable partial index builds to avoid losing API work after interruption.
- [x] Retry transient 429 / 5xx API errors with exponential backoff.
- [x] Add `scripts/build_dense_v26.py`.
- [x] Add `scripts/evaluate_v26.py`.
- [x] Add `scripts/search_v26.py`.
- [x] Add no-API unit tests in `tests/test_dense_v26.py`.
- [ ] Run V2.6 unit tests.
- [ ] Build 256d index.
- [ ] Build 768d index.
- [ ] Compare 10-query benchmark against V2.5.
- [ ] Inspect V3 holdout after the 10-query pilot.
- [ ] Record query API latency and candidate quality.
- [ ] Check whether improved retrieval allows later Gemma 4 judging to use a smaller candidate pool.

## Artifact paths

```text
artifacts/v2/embeddinggemma-300m-256
artifacts/v26/gemini-embedding-2-256
artifacts/v26/gemini-embedding-2-768
```

## Colab build

Use the same Gemini API key for both builds.

```bash
python -u scripts/build_dense_v26.py \
  --dimension 256 \
  --batch-size 50 \
  --output artifacts/v26/gemini-embedding-2-256
```

Then:

```bash
python -u scripts/build_dense_v26.py \
  --dimension 768 \
  --batch-size 50 \
  --output artifacts/v26/gemini-embedding-2-768
```

Both commands resume compatible partial builds automatically.

## Evaluation

```bash
python -u scripts/evaluate_v26.py \
  --baseline-index artifacts/v2/embeddinggemma-300m-256 \
  --gemini-index artifacts/v26/gemini-embedding-2-256 \
  --gemini-index artifacts/v26/gemini-embedding-2-768 \
  --top-k 10 \
  --device cuda \
  --output artifacts/v26/v25-vs-v26.json
```

## Decision

Promote V2.6 only if it materially improves candidate quality, especially:
- `บ้าน`: surfaces `เรือน / บ้านเรือน / บ้านช่อง` earlier;
- `สวย`: surfaces `งาม / งดงาม` earlier;
- `เร็ว`: surfaces `ไว / รวดเร็ว / ด่วน` reliably;
- does not regress semantic validity for `เดิน / ฝน / รัก / มืด`.

If 256d is already clearly better than V2.5, prefer it for storage/compute unless 768d provides a meaningful additional gain.


## Project-level disposition

This experiment was implemented as an API-based embedding challenger, but it was never promoted to the baseline. Subsequent embedding/ranking research continued to select V2.5 EmbeddingGemma as the preferred local retrieval foundation.

Keep this branch as a reproducible API challenger only. Do not add the Gemini API dependency to the normal search path without a future frozen benchmark showing a material, repeatable gain over V2.5.

Canonical current decision:
`plans/semantic-search-research-summary-2026-09-15.md` on `feat/dictionary-semantic-v2-5-embeddinggemma`.
