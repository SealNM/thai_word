# V5 — Listwise Lexical Ranking

Status: **V5.1 Qwen experiment complete; V5.2 instruction-following multilingual reranker implemented, real-model pilot pending**

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
- [x] Run Jina v3.5 on the 10-query benchmark.
- [x] Inspect lexical-neighbor failures first: `เดิน / รัก / มืด / บ้าน / ฝน`.
- [x] Compare pure listwise against light V2.5 fusion.
- [x] Record model load time and seconds/query.

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


## First real Jina v3.5 pilot — findings

Runtime:
- V2.5 load: ~33s.
- Jina v3.5 load: ~18s.
- 10-query evaluation: ~32s total.
- Average: ~3.22s/query.
- Live progress / ETA worked as intended and solved the previous "silent run" problem.

Quality improvements:
- `ฝน`: `พิรุณ / พลาหก / โปรย` moved to the top; `น้ำตก` fell below stronger lexical candidates.
- `เดิน`: `ย่างเท้า / ย่างตีน / ก้าว / ย่าง` beat `วิ่ง`; this is a clear improvement over V4 pointwise reranking.
- `พูด`: `พูดจา / เอ่ย / จา / เว้า` remained high.
- `กลัว`: `เกรงกลัว / หวาดกลัว / หวาดเกรง` appeared high.

Remaining problems:
- `บ้าน`: rare/dictionary forms `ภูม / วาสะ / เวศม์ / วสนะ / คาม / อธิวาส` still outrank `บ้านเรือน`; commonness preference is not being followed strongly.
- `สวย`: `สะ / ย้อง / สุทัศน์` still outrank more natural common forms such as `งดงาม`.
- `เร็ว`: rare forms `รยะ / สีฆ- / เชาว์ / ชัพ` remain too high; `ไว / รวดเร็ว / ด่วน` should be stronger.
- `มืด`: `เสียสายตา` remains too high.
- `รัก`: `สายใจ / จอด / ชู้สาว` still leak into the top set.

Interpretation:
- The listwise architecture is useful: relative semantic ordering improved, especially `เดิน`.
- Jina v3.5 is still fundamentally trained as a relevance reranker. Its native prompt ranks passages by relevance to a query; the Thai Words custom priority text is embedded inside the query rather than handled by a true instruction-following ranking interface.
- Therefore the remaining bottleneck is now instruction obedience / lexical-ranking objective, not raw model size.

## V5.1 direction — newer instruction-following ranker

Do not add another commonness heuristic before testing a ranker that can explicitly follow ranking criteria.

Primary local experiment:
- `Qwen/Qwen3.5-0.8B`
- Apache-2.0
- modern 0.8B post-trained model
- use as a **generative listwise ranker**, not as a pairwise scorer:
  - give target word + selected sense + grammatical role;
  - give all candidate IDs + word + definition together;
  - ask for a strict JSON ordering of candidate IDs;
  - explicitly require semantic preservation first and common contemporary Thai before equally valid literary/archaic forms.
- This directly tests the user's hypothesis that a newer small foundation model can outperform older reranker foundations at the same scale.

Secondary local control:
- `ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b`
- 1B, 100+ languages, explicitly instruction-following reranker.
- Research-only license (CC BY-NC-SA 4.0), so treat it as a quality control rather than final production dependency.

Hosted frontier control, only if needed:
- `mixedbread-ai/mxbai-rerank-v3.1-listwise`
- listwise + natural-language instruction following.
- Use only after local V5.1 results, so we do not add API cost/dependency prematurely.

### V5.1 experiment order

1. Keep V2.5 retrieval unchanged.
2. Restore candidate pool to 50 for the model comparison, because `เรือน` appeared with the earlier top-50 pipeline and may be outside the current top-40 pool.
3. Test Qwen3.5-0.8B generative listwise first.
4. Compare against Jina v3.5 using exactly the same 50 candidates.
5. Only if Qwen3.5 remains weak, test the ContextualAI 1B instruction-following reranker.
6. Do not reintroduce TNC/commonness until the instruction-following comparison is complete.
7. Preserve live progress + ETA for every evaluator.

Decision gate:
- Prefer the smallest model that improves `บ้าน / สวย / เร็ว` common-vs-rare ordering **without regressing** `เดิน / ฝน / รัก / มืด` semantic validity.


## V5.1 implementation checkpoint

Implemented on the same V5 branch:

- [x] Add `thai_generative_v51.py`.
- [x] Add `Qwen35GenerativeListwiseRanker` using `Qwen/Qwen3.5-0.8B`.
- [x] Use Qwen3.5 as a true generative listwise ranker: all candidate IDs, words, and definitions are presented together and the model returns a full ordering.
- [x] Make ranking policy explicit: intended sense -> grammatical role -> natural lexical substitute -> common contemporary Thai before equally valid rare/literary forms.
- [x] Keep Qwen3.5 in its default non-thinking mode and use deterministic generation for this ranking task.
- [x] Add strict JSON-array output contract.
- [x] Add robust parser that accepts a complete JSON ordering, recovers wrapped output, removes duplicates/out-of-range IDs, and appends any missing IDs in original V2.5 order instead of crashing.
- [x] Record whether each Qwen generation parsed completely or required recovery.
- [x] Add `scripts/evaluate_v51.py`.
- [x] Precompute the V2.5 top-50 pools once so Jina and Qwen3.5 see **exactly the same candidates**.
- [x] Load Jina and Qwen3.5 sequentially and release GPU cache between rankers.
- [x] Preserve live progress, per-query timing, ETA, and immediate top-10 output.
- [x] Add `scripts/search_v51.py`.
- [x] Add parser/prompt unit tests in `tests/test_generative_v51.py`.
- [ ] Run V5 and V5.1 no-model tests in Colab after pulling.
- [ ] Run the real Jina-vs-Qwen3.5 shared-top-50 comparison.
- [ ] Inspect parse completeness; a frequent recovered/partial Qwen output is itself a failure signal.
- [ ] Compare `บ้าน / สวย / เร็ว` common-vs-rare ordering.
- [ ] Confirm `เดิน / ฝน / รัก / มืด` do not regress on semantic validity.
- [ ] Compare seconds/query and model load time.

### Qwen3.5 runtime note

The official Qwen3.5 model card states that a recent/latest Hugging Face Transformers build is required. The repository dependency range already permits newer Transformers versions, but an older existing Colab environment may need an explicit upgrade before the first V5.1 run.

### Recommended V5.1 comparison

```bash
python -u scripts/evaluate_v51.py \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --candidate-pool 50 \
  --top-k 10 \
  --device cuda \
  --ranker jina-v3.5 \
  --ranker qwen3.5-0.8b \
  --mode listwise \
  --mode fusion \
  --output artifacts/v5/qwen35-vs-jina-shared50.json
```

If Qwen3.5 fails to load because the installed Transformers build is too old, upgrade Transformers first and restart the Colab runtime before rerunning the comparison.


## V5.1 first shared-top-50 run — diagnosis

Observed runtime:
- Jina v3.5: ~3.13s/query.
- Qwen3.5-0.8B: ~13.80s/query.
- Qwen model load: ~36s.

Critical finding:
- Every Qwen query reported `generation parse: recovered (json)`.
- Every final Qwen top-10 exactly matched the V2.5 baseline.
- Therefore this run does **not** measure Qwen3.5 ranking quality. The generative ranker failed to provide enough explicit candidate IDs and the fallback path filled the missing IDs in original V2.5 order.

The previous output contract asked the 0.8B model to emit a complete permutation of all 50 candidates. That is unnecessary for a top-10 product result and adds output latency / format pressure.

### V5.1 short-form ranking fix

Implemented:
- [x] Ask Qwen for only the best 15 candidate IDs from the shared top-50 pool.
- [x] Keep all 50 candidates in the prompt so global comparison is unchanged.
- [x] Append unranked candidates in original V2.5 order only after the explicit Qwen top-15, preserving compatibility with the V5 ranking interface.
- [x] Reduce default generation budget from 384 to 192 tokens.
- [x] Stop greedy decoding; use conservative non-thinking sampling for the Qwen3.5 instruction model.
- [x] Make the parser choose the largest valid JSON array rather than the first bracket pair.
- [x] Recover ordered singleton labels such as `[7] [3] [12]` when small-model formatting drifts.
- [x] Keep numeric recovery as a final diagnostic fallback.
- [x] Track explicit parsed ID count separately from the fallback-completed permutation.
- [x] Evaluator prints `parsed/requested IDs`.
- [x] Evaluator prints a raw generation preview whenever recovery is still required.
- [x] Add tests for largest-array selection and singleton-label recovery.
- [ ] Rerun Qwen3.5-0.8B only first; do not rerun Jina unnecessarily.
- [ ] Require at least 10/15 explicit IDs on most queries before judging Qwen ranking quality.
- [ ] If explicit output is still poor, stop generative 0.8B and move to an instruction-trained reranker rather than increasing output length again.

Recommended focused rerun:

```bash
python -u scripts/evaluate_v51.py \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --candidate-pool 50 \
  --top-k 10 \
  --device cuda \
  --ranker qwen3.5-0.8b \
  --qwen-output-count 15 \
  --qwen-max-new-tokens 192 \
  --mode listwise \
  --mode fusion \
  --output artifacts/v5/qwen35-short15-rerun.json
```


## V5.1 short-form Qwen rerun — final finding

Observed:
- Model load: ~12s.
- 10-query run: ~29s total.
- Average: ~2.94s/query.
- This is fast enough for experimentation, but output reliability and ranking quality were not good enough.

Parse behavior:
- Every query still required recovery.
- Explicit IDs per requested top-15 ranged from 5/15 to 14/15.
- Several generations simply echoed the leading V2.5 IDs.
- Some generations produced semantically weak or clearly odd candidates high in the ranking.

Examples:
- `ฝน`: `ตก / หยด / เละ` leaked into the top set.
- `โกรธ`: `ออกฤทธิ์ / ดูหรู / แผลงฤทธิ์แผลงเดช` leaked upward.
- `พูด`: `เคาะ / เพ้อ / เป็นปากเสียง` ranked too high.
- `บ้าน`: `บ้านเรือน` improved strongly, but `เวศม์` still ranked first and `บริเวณ / โรงเรือน` leaked upward.
- `สวย / มืด / กลัว` remained unstable.
- `รัก / เร็ว` often echoed the V2.5 ordering rather than demonstrating a reliable new ranking policy.

Decision:
- [x] Stop the Qwen3.5-0.8B generative-listwise path.
- [x] Do not increase generation length or model size to rescue the same approach.
- [x] Keep the code as an experiment/reference only.
- [x] Move to a model trained specifically for instruction-following reranking.

## V5.2 — Instruction-following multilingual reranker

Primary local control:
- `ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b`
- 1B parameters.
- 100+ languages.
- 32K context.
- Designed specifically for custom reranking instructions.
- Supports Sentence Transformers `CrossEncoder.rank/predict(..., prompt=instruction)`.
- Research-only license: CC BY-NC-SA 4.0; do not treat as final production dependency.

Why this test:
- V5 showed that listwise comparison can improve semantic-neighbor errors.
- V5.1 showed that a small general instruction model is not reliably structured enough to act as the ranking engine.
- V5.2 isolates the remaining hypothesis: whether a **reranker trained to obey custom ranking instructions** can enforce Thai Words' priority order more reliably than relevance-only Jina or generative Qwen.

Implemented:
- [x] Add `thai_instruct_v52.py`.
- [x] Use the official Sentence Transformers CrossEncoder path.
- [x] Pass Thai Words' ranking policy through the model's native `prompt` instruction interface.
- [x] Ranking policy prioritizes intended sense and grammatical role before commonness.
- [x] Common contemporary Thai is preferred only among equally valid lexical substitutes.
- [x] Keep the shared V2.5 top-50 retrieval pool.
- [x] Add pure `instruct` mode.
- [x] Add light V2.5 fusion control with V2.5 weight 0.2 and instruction-reranker weight 1.0.
- [x] Auto-select BF16 on supported CUDA hardware, otherwise FP16 on CUDA and FP32 on CPU.
- [x] Add live progress, per-query latency, and ETA.
- [x] Add `scripts/evaluate_v52.py`.
- [x] Add no-model unit coverage in `tests/test_instruct_v52.py`.
- [ ] Run V5.2 tests in Colab.
- [ ] Run the 10-query V5.2 real-model pilot.
- [ ] Inspect `บ้าน / สวย / เร็ว` for common-vs-rare ordering.
- [ ] Confirm `เดิน / ฝน / รัก / มืด` do not regress on semantic validity.
- [ ] Compare latency against Jina (~3.13s/query) and Qwen3.5 (~2.94s/query).

Recommended pilot:

```bash
python -u scripts/evaluate_v52.py \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --candidate-pool 50 \
  --top-k 10 \
  --device cuda \
  --mode instruct \
  --mode fusion \
  --output artifacts/v5/ctxl-v2-instruct-1b-pilot.json
```

If V5.2 still cannot obey the common-vs-rare priority reliably, the next meaningful control is a true listwise + instruction-following service such as Mixedbread's `mxbai-rerank-v3.1-listwise`, rather than another local general-purpose LLM experiment.
