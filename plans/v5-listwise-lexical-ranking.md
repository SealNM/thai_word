# V5 — Listwise Lexical Ranking

Status: **core implementation ready; first real-model pilot pending**

Branch: `feat/dictionary-semantic-v5-listwise-ranker`

Base: `feat/dictionary-semantic-v4-instruction-reranker`

## Why V5

V4 established that V2.5 already retrieves useful candidates, but pointwise/cross-encoder reranking still confuses lexical substitutes with semantic neighbors:

- `เดิน -> วิ่ง`
- `รัก -> คู่รัก`
- `ฝน -> น้ำตก / เมฆ`
- `มืด -> หน้ามืด / เสียสายตา`
- `บ้าน -> วาสะ / อธิวาส` before common alternatives

Increasing model size from Qwen3-Reranker-0.6B to 4B improved several queries but was slow to download/load and still did not resolve the core ranking behavior consistently.

V5 changes the ranking algorithm rather than adding another heuristic.

## Product objective

For a Thai fiction writer, rank words by:

1. semantic / sense preservation,
2. matching grammatical role,
3. natural lexical substitutability,
4. common contemporary wording before equally accurate rare/literary wording,
5. preserve useful literary/archaic vocabulary lower in the list.

The listwise ranker sees all candidate entries together so candidates compete directly.

## Architecture

```text
query + selected sense + category
              |
              v
        V2.5 retrieval
              |
         top 40 candidates
              |
              v
     multilingual LISTWISE
          reranker
   (all candidates jointly)
              |
       +------+------+
       |             |
   pure listwise   light fusion
                   V2.5 0.2
                   listwise 1.0
       |             |
       +------+------+
              |
            top 10
```

## Initial local model

Primary experimental model:

- `jinaai/jina-reranker-v3.5`
- 0.6B parameters
- multilingual
- listwise / last-but-not-late ranking
- local `transformers` inference via `model.rerank(query, documents)`
- model file is approximately 1.2 GB rather than the ~8 GB 4B Qwen experiment

Important licensing constraint:

- CC BY-NC 4.0
- suitable for this research/pilot
- **not accepted as a final commercial production dependency**
- if V5 validates listwise ranking, production must either:
  - obtain compatible commercial rights,
  - replace it with a commercially compatible listwise model,
  - use a hosted provider with acceptable terms, or
  - distill/train a compatible ranker.

## V5 task text

The query explicitly tells the ranker to compare candidates against one another and prioritize:

- replacement in a natural Thai sentence,
- intended sense,
- grammatical role,
- common contemporary Thai when semantic quality is equal,
- literary/archaic alternatives after equally accurate common forms,
- rejection of associated-only words and semantic-neighbor errors.

Candidate documents contain only grounded dictionary word + matched definition.

V2.5 relation labels are not sent to the listwise model.

## Implementation

- [x] Create `thai_listwise_v5.py`.
- [x] Add `JinaListwiseRanker` using the model's native listwise `rerank` method.
- [x] Add strict result validation: every input candidate must appear exactly once.
- [x] Add category + selected sense to the listwise query.
- [x] Add pure `listwise` ranking mode.
- [x] Add optional light `fusion` control (V2.5 rank weight 0.2, listwise weight 1.0).
- [x] Do **not** add TNC/commonness heuristics to the first V5 pilot.
- [x] Default candidate pool to 40 to reduce latency while retaining broader V2.5 recall.
- [x] Add `scripts/search_v5.py`.
- [x] Add `scripts/evaluate_v5.py`.
- [x] Add no-model unit tests in `tests/test_listwise_v5.py`.
- [x] Add live progress output with `flush=True`.
- [x] Evaluator reports model loading, each query start/completion, elapsed time, progress, and ETA.
- [x] Recommended Colab invocation uses `python -u` to disable stdout buffering.
- [ ] Run unit tests after pulling the branch.
- [ ] Run Jina v3.5 on the 10-query benchmark.
- [ ] Inspect lexical-neighbor failures first: `เดิน / รัก / มืด / บ้าน / ฝน`.
- [ ] Compare pure listwise against light V2.5 fusion.
- [ ] Record model load time and seconds/query.

## Evaluation decision

The first V5 pilot should answer one question:

> Does a small listwise model produce a more writer-useful lexical ordering than the pointwise Qwen V4 pipeline?

Success examples:

- `บ้าน`: common valid forms such as `เรือน / บ้านเรือน / บ้านช่อง` should outrank `วาสะ / อธิวาส` when meanings are comparably valid.
- `เดิน`: `วิ่ง` should not beat genuine walking substitutes merely because it is semantically close.
- `รัก`: noun forms such as `คู่รัก` should not beat verb/emotion substitutes.
- `มืด`: compounds/conditions such as `หน้ามืด` should not beat light/darkness substitutes.
- `พูด`: `พูดจา / กล่าว / เอ่ย / เจรจา` should remain high.

Do not tune commonness weights before this listwise comparison is complete.

## Later controls, not first run

If local listwise ranking wins:

1. compare a hosted frontier listwise reranker such as Mixedbread's current listwise offering;
2. compare Cohere Rerank as a multilingual external control;
3. investigate commercially compatible local listwise models;
4. if necessary, distill the listwise ordering into a smaller production-safe model.

## Recommended first V5 pilot

Pull the branch, run tests, then:

```bash
python -u scripts/evaluate_v5.py \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --candidate-pool 40 \
  --top-k 10 \
  --device cuda \
  --mode listwise \
  --mode fusion \
  --output artifacts/v5/jina-v3.5-pilot.json
```

The evaluator prints live status before model loading and before/after every query, including ETA.
