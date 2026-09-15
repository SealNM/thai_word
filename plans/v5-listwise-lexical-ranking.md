# V5 — Listwise Lexical Ranking

Status: **research archived; later distillation/reranking experiments did not beat V2.5 overall**

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
- [x] Confirm `เดิน / ฝน / รัก / มืด` do not regress on semantic validity.
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
- [x] Add light V2.5 fusion control with V2.5 weight 0.1 and instruction-reranker weight 1.0. The original 0.2 control could overturn the instruction rank in a simple adversarial case, so V5.2 keeps V2.5 strictly secondary.
- [x] Auto-select BF16 on supported CUDA hardware, otherwise FP16 on CUDA and FP32 on CPU.
- [x] Add live progress, per-query latency, and ETA.
- [x] Add `scripts/evaluate_v52.py`.
- [x] Add no-model unit coverage in `tests/test_instruct_v52.py`.
- [x] Run V5.2 tests in Colab.
- [x] Run the 10-query V5.2 real-model pilot.
- [x] Inspect `บ้าน / สวย / เร็ว` for common-vs-rare ordering.
- [ ] Confirm `เดิน / ฝน / รัก / มืด` do not regress on semantic validity.
- [x] Compare latency against Jina (~3.13s/query) and Qwen3.5 (~2.94s/query).

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


### V5.2 fusion-test correction

The first no-model test exposed that `v25_weight=0.2` was not actually light enough: a V2.5 rank-1 candidate at instruction rank 3 could beat an instruction rank-1 candidate that V2.5 had at rank 20.

Decision:
- reduce the V5.2 default V2.5 fusion weight to `0.1`;
- keep instruction weight at `1.0`;
- update the unit test to encode this intended behavior.

This changes only the V5.2 fusion control; pure `instruct` mode is unaffected.


## V5.2 real-model finding — instruction-following reranker is not the answer

Runtime:
- Model load: ~72s.
- 10-query run: ~89s total.
- Average: ~8.86s/query.
- This is materially slower than Jina v3.5 (~3.13s/query) and Qwen3.5-0.8B (~2.94s/query).

Quality:
- `สวย` improved in one important way: `งาม / งดงาม` moved near the top.
- `พูด` remained usable around `พูดจา / จา / กล่าว / เจรจา`.

However the model regressed or remained weak on the critical failure cases:
- `เดิน`: `วิ่ง` ranked #2, worse than Jina listwise.
- `มืด`: `มุมมืด / หน้ามืด / เดือนมืด` outranked direct darkness substitutes.
- `รัก`: `ที่รัก / สารภาพ / คู่รัก` leaked into the top results.
- `บ้าน`: `ภูม / นิเวศ- / เวศม์ / อธิวาส / วาสะ` still dominated; `บ้านเรือน / เรือน / บ้านช่อง` did not become the desired top group.
- `เร็ว`: rare forms such as `เชาว์ / รยะ / ระเร็ว / สีฆ-` remained too high.
- `กลัว`: `กระดก` leaked into the top set.

Decision:
- [x] Stop the ContextualAI V5.2 path.
- [x] Pause further local model-hopping for this ranking problem.
- [x] Keep Jina v3.5 as the strongest semantic-ranking research baseline so far.
- [x] Treat the remaining issue as a product-specific ordering problem: semantically valid candidates are often present, but rare dictionary forms are over-promoted.

## V5.3 proposed direction — semantic rank + rarity penalty, not commonness boost

Previous commonness fusion failed because high-frequency unrelated words received a positive boost. V5.3 should invert that design:

> Commonness must never create semantic relevance. It may only demote candidates that are demonstrably rare.

Proposed architecture:

```text
V2.5 top 50
    ↓
Jina v3.5 listwise semantic ranking
    ↓
top semantic band / score-aware window
    ↓
rarity penalty only
    - no positive frequency boost
    - token-aware phrase familiarity
    - capped demotion
    ↓
top 10
```

Design rules:
1. Jina remains the semantic ordering signal.
2. TNC/commonness is used only as a **negative rarity prior**.
3. Very common words never get bonus points merely for being frequent.
4. Rare/archaic candidates can move down only a bounded number of positions.
5. Multiword forms such as `บ้านเรือน / บ้านช่องห้องหอ` use token-aware familiarity so they are not penalized solely because the exact phrase is sparse.
6. Apply the penalty only inside a semantic window (for example Jina top 15–20 or score-near-top band), so unrelated candidates cannot enter from the tail.
7. Sweep only small demotion caps (for example 2 / 4 / 6) rather than broad frequency weights.
8. Inspect raw Jina listwise scores and TNC counts in the evaluator before choosing a production threshold.

Primary expected effect:
- `บ้าน`: demote `วาสะ / เวศม์ / วสนะ / อธิวาส` enough for `เรือน / บ้านเรือน / บ้านช่อง` to surface, without boosting generic `ที่ / อยู่ / ปลูก`.
- `สวย`: allow `งดงาม / งาม` to pass rarer dictionary forms.
- `เร็ว`: allow `ไว / รวดเร็ว / ด่วน` to pass `รยะ / สีฆ- / เชาว์`.
- Preserve Jina's semantic win on `เดิน`, where `วิ่ง` already ranks below direct walking alternatives.

This is now preferred over testing another local reranker model.


## V5.3 priority change — test Google's current Gemma 4 before rarity penalties

User preference: because V2.5 already works well with Google's EmbeddingGemma, test the current Gemma family before committing to heuristic rarity penalties.

Current Google release research (September 2026):
- Gemma 4 is newer than the Gemma 3 / Gemma 3n / T5Gemma 2 generation.
- Google released Gemma 4 in March 2026 and a 12B Unified variant in June 2026.
- The smallest instruction model is `google/gemma-4-E2B-it`.
- E2B has ~2.3B effective parameters (~5.1B total including per-layer embeddings), 128K context, multilingual pretraining over 140+ languages, native system-role support, and optional thinking mode.
- Gemma 4 is Apache-2.0, which is materially better for a future production path than the research-only Jina / ContextualAI licenses.
- Full BF16 E2B checkpoint is ~10.2 GB.
- Google also publishes an official 8-bit mobile-Transformers checkpoint:
  `google/gemma-4-E2B-it-qat-mobile-transformers`
  at ~2.46 GB, making it the preferred first Colab experiment.
- Google also publishes an official Q4 GGUF (~3.35 GB text weights), but introducing llama.cpp is unnecessary for the first experiment.

### Why Gemma 4 E2B is worth a clean test

This is not another reranker-model swap. Gemma 4 is a newer instruction model with:
- native system instructions;
- substantially stronger general language/instruction capability than the tiny Qwen3.5-0.8B experiment;
- a direct lineage with the Gemma family already successful in V2.5 retrieval;
- an official compact quantized checkpoint suitable for local experimentation.

### V5.3 experiment design

Do **not** ask Gemma 4 to emit a full permutation of 50 candidates.

Run two clean modes on the exact same V2.5 top-50 pools:

1. **Gemma-direct**
   - Gemma sees all top-50 candidates.
   - Ask only for the best 10 candidate IDs.
   - Thinking disabled for latency.
   - Native system prompt carries the Thai Words ranking policy.
   - Strict parser diagnostics; no silent fallback when fewer than 10 explicit IDs are returned.

2. **Jina -> Gemma judge**
   - Jina v3.5 first narrows the V2.5 top-50 to a semantic top-20.
   - Gemma 4 sees those 20 and returns the best 10.
   - Purpose: preserve Jina's semantic wins such as `เดิน` while letting Gemma decide common-vs-rare ordering inside the semantically credible band.

Primary model:
- `google/gemma-4-E2B-it-qat-mobile-transformers` (official 8-bit, ~2.46 GB)

Control only if mobile checkpoint compatibility becomes a problem:
- `google/gemma-4-E2B-it` BF16 (~10.2 GB)

Generation:
- thinking OFF;
- follow Gemma 4 recommended sampling defaults unless deterministic ranking proves stable;
- request only top 10 IDs;
- keep raw output + explicit-ID count in evaluation logs.

Decision questions:
- Does `บ้าน` move toward `เรือน / บ้านเรือน / บ้านช่อง` before `วาสะ / เวศม์ / อธิวาส`?
- Does `สวย` prefer `งาม / งดงาม` over rare dictionary forms?
- Does `เร็ว` prefer `ไว / รวดเร็ว / ด่วน` over `รยะ / สีฆ- / เชาว์`?
- Can Gemma preserve Jina's improvements on `เดิน / ฝน` without introducing `วิ่ง / น้ำตก / เมฆ` too high?
- Is output formatting reliable enough for top-10 structured ranking?
- Is latency acceptable relative to Jina (~3.13s/query)?

Rarity-penalty V5.3 work is postponed and becomes V5.4 only if Gemma 4 does not solve the product-specific common-vs-rare ordering sufficiently.


## V5.3 implementation checkpoint — Gemma 4

Implemented:
- [x] Add `thai_gemma_v53.py`.
- [x] Default to Google's official compact checkpoint:
  `google/gemma-4-E2B-it-qat-mobile-transformers`.
- [x] Use `AutoProcessor` + `AutoModelForMultimodalLM` following Google's Gemma 4 Transformers interface.
- [x] Use native `system` role for the Thai Words ranking policy.
- [x] Disable Gemma thinking with `enable_thinking=False`.
- [x] Use Google's published sampling defaults: temperature 1.0, top-p 0.95, top-k 64.
- [x] Request only top-10 candidate IDs; never ask for a full top-50 permutation.
- [x] No silent ranking fallback: incomplete Gemma output remains visibly incomplete.
- [x] Keep raw generation output and parse-completeness metadata.
- [x] Add `Gemma-direct`: judge the same V2.5 top-50 pool directly.
- [x] Add `Jina -> Gemma`: Jina semantically narrows top-50 to top-20, then Gemma chooses top-10.
- [x] Cache all Jina top-20 pools first, unload Jina, clear GPU cache, and only then load Gemma. This avoids keeping both rankers resident during Gemma inference.
- [x] Add live setup/model/query progress and ETA.
- [x] Add `scripts/evaluate_v53.py`.
- [x] Add no-model parser/prompt tests in `tests/test_gemma_v53.py`.
- [ ] Run V5.3 no-model tests in Colab.
- [ ] Run the real Gemma 4 direct + Jina→Gemma pilot.
- [ ] Require 10/10 explicit IDs on most queries before trusting the quality comparison.
- [ ] Compare `บ้าน / สวย / เร็ว` common-vs-rare ordering.
- [ ] Confirm `เดิน / ฝน / รัก / มืด` semantic validity.
- [ ] Record direct vs cascade latency.

### Recommended V5.3 pilot

```bash
python -u scripts/evaluate_v53.py \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --candidate-pool 50 \
  --jina-pool 20 \
  --top-k 10 \
  --device cuda \
  --mode direct \
  --mode jina-gemma \
  --output artifacts/v5/gemma4-e2b-direct-vs-jina20.json
```

Notes:
- The first Gemma run may download roughly 2.5 GB of model weights.
- Gemma 4 requires a recent Transformers build. If the runtime cannot resolve the Gemma 4 architecture or `AutoModelForMultimodalLM`, upgrade Transformers and restart Colab.
- The official Gemma 4 E2B model card states 128K context for the smaller E2B/E4B models, native system-role support, configurable thinking, and multilingual pretraining over 140+ languages.


## V5.3 real Gemma 4 pilot — findings

Reliability:
- Gemma returned a complete explicit 10/10-ID ranking for every query in both direct and Jina→Gemma modes.
- This is a major improvement over Qwen3.5-0.8B, which repeatedly required parser recovery.

Runtime:
- Gemma model load: ~60s.
- Gemma-direct on top-50: ~19–22s/query.
- Jina→Gemma on top-20: ~17–19s/query, plus the one-time Jina narrowing pass.
- Both modes together: ~38.05s/query.
- Quality is promising, but direct top-50 latency is too high for a final interactive search path.

### Quality result

Gemma-direct is the strongest overall direction tested so far for the product-specific common-vs-rare requirement.

Strong wins:
- `เดิน`: `ก้าว / ย่างเท้า / ย่าง / ผเดิน ...`; importantly `วิ่ง` disappeared from the top-10.
- `สวย`: `งดงาม` ranked #1 and common/natural appearance vocabulary improved substantially.
- `พูด`: `พูดจา / จา / ... / เอ่ย / เอื้อนเอ่ย` remained useful.
- `เร็ว`: `รีบ / ไว / เร็ว ๆ / รีบรุด / เร่ง / เร่งรีบ / ฉับไว / ว่องไว`; this is the clearest common-first result so far.
- `กลัว`: common/direct fear forms such as `หวาดกลัว / หวั่นหวาด / หวาดเกรง / เกรงกลัว` surfaced well.
- `บ้าน`: `บ้านเรือน` reached #2 in direct mode and #1 after Jina narrowing, a large improvement over previous rankers.

Remaining semantic leakage in direct mode:
- `ฝน`: verbs/associations such as `ตก / ปรอย` and weather neighbors such as `เมฆ / พยับเมฆ` still rank too high.
- `รัก`: `สารภาพ / ที่รัก / น้ำใจ / ชู้สาว` are related but not clean replacements.
- `มืด`: mostly improved, but `บอด` remains a role/meaning mismatch.
- `สวย`: `ดี` is overly general despite the otherwise strong ordering.
- `บ้าน`: `คาม` still outranks `บ้านเรือน` in direct mode.

### Direct vs Jina→Gemma

Jina→Gemma is not a universal improvement:
- It helps `บ้าน` by moving `บ้านเรือน` to #1 and reduces some broad tail noise.
- It helps semantic safety on parts of `ฝน`.
- But it often reintroduces Jina's rare-word bias and can remove strong common candidates before Gemma sees them.
- It is clearly worse than Gemma-direct for `เดิน / สวย / มืด / เร็ว / กลัว` common-first behavior.

Decision:
- [x] Keep Gemma-direct as the primary V5.3 direction.
- [x] Keep Jina→Gemma as a diagnostic/control, not the default architecture.
- [x] Do not proceed to rarity-penalty V5.4 yet.
- [x] First test whether a stronger lexical-validity prompt and a smaller direct candidate pool can preserve Gemma's quality while reducing latency.

## V5.3.1 — hard lexical gate + direct top-30

Implemented prompt revision:
- prompt version `v5.3.1-hard-lexical-gate`;
- make same-sense and same-grammatical-role requirements explicit hard gates;
- tell Gemma to infer candidate role from the headword/definition when no POS metadata is available;
- explicitly state that associated words must rank below true substitutes even if frequent or semantically close;
- explicitly demote different parts of speech, cause/effect, objects/agents, compounds with another role, and manner/subtype changes;
- commonness is considered only after lexical validity;
- if fewer than 10 strong substitutes exist, related alternatives may fill only the lower positions.

Rationale:
- the model is already strong at common-vs-rare ordering;
- the remaining errors are primarily lexical-validity leaks (`ฝน→ตก/เมฆ`, `รัก→สารภาพ/ที่รัก`, `มืด→บอด`);
- therefore the next test should tighten semantic/grammatical gating rather than add frequency heuristics.

Focused next run:
- Gemma-direct only;
- reduce V2.5 candidate pool from 50 to 30;
- still request top-10;
- compare against the existing top-50 Gemma-direct pilot;
- success requires keeping `บ้านเรือน / งดงาม / ไว / รวดเร็ว` style gains while reducing semantic leaks and bringing latency down materially.

Recommended command:

```bash
python -u scripts/evaluate_v53.py \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --candidate-pool 30 \
  --top-k 10 \
  --device cuda \
  --mode direct \
  --output artifacts/v5/gemma4-e2b-direct30-hardgate.json
```


## Final project-level checkpoint after V5

Gemma 4 direct remained the strongest *quality-oriented* judge tested in this line of research, especially for common-first ordering, but it was too slow for normal interactive search and still leaked lexical-role/association errors.

The project subsequently tested:
- V2.6 bounded TNC rarity reranking;
- V2.7 Gemma-distilled linear pairwise ranking;
- PyThaiNLP static word vectors.

None produced a consistent holdout win over V2.5. V2.7 in particular introduced large regressions such as ฝน→เมฆ, รัก→จอด, พูด→พจน์/ปาก, and กลัว→กระดก.

Therefore:
- do not continue V5.3.1 tuning on the existing 10-query benchmark;
- keep Gemma 4 as a useful offline teacher/research reference only;
- keep V2.5 as the validated runtime baseline;
- require a larger frozen human-reviewed benchmark before another ranking-model round.

Canonical handoff:
`plans/semantic-search-research-summary-2026-09-15.md` on `feat/dictionary-semantic-v2-5-embeddinggemma`.
