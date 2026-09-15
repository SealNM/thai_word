# Thai Words — Writer Lexical Relevance & Utility Ranking Plan

Date: 2026-09-15  
Status: **In progress — Phase 3 candidate locked; checkpoint verified; frozen benchmark ready**  
Baseline: `feat/dictionary-semantic-v2-5-embeddinggemma`  
Working branch: `feat/dictionary-substitutability-benchmark`

> Note: the branch/script names still use `substitutability` for continuity, but the product objective is now broader: **Writer Lexical Relevance / Writer Utility Ranking**.

## Product goal

Thai Words is not a pure synonym or lexical-substitution search engine.

The real goal is:

> When a writer searches for a concept, return words that are useful for writing about that concept, ordered by how directly and naturally useful they are.

A result does **not** need to be a one-to-one synonym of the query to be valuable.

For example, for `ฝน`, all of these can be useful:

- direct / near synonyms: `พิรุณ`, `พรรษ`;
- kinds of rain: `ฝนซู่`, `ฝนไล่ช้าง`, `ฝนหลวง`;
- manner/action words: `โปรย`, `ปรอย`, `ตก`;
- scene/context words: `พยับเมฆ`, `เมฆ`, `ละออง`;
- effect/state words: `เปียก`, `เละ`, `น้ำป่า`;
- literary or imagery-rich vocabulary: `โบกขรพรรษ`.

A writer may use several related words together rather than replacing the query directly, for example:

```text
พยับเมฆลอยครึ่มต่ำลงมา ส่งน้ำฝนโปรยเป็นสาย
```

Therefore grammatical-role mismatch is **not automatically a negative signal**. A verb, modifier, scene word, effect word, or associated image can be a good result if it is genuinely useful for describing the searched concept.

## Core ranking principle

The ranking objective is hierarchical writer usefulness, not generic semantic similarity and not synonym purity.

A rough priority is:

1. direct / highly usable lexical alternatives;
2. near-synonyms, register shifts, literary alternatives;
3. subtypes or more specific forms;
4. manner/action words useful for describing the concept;
5. scene/context/imagery words;
6. effects/states/consequences that can inspire description;
7. weakly related or misleading material;
8. unrelated/noise.

This is a **ranking preference**, not a hard partition. A particularly useful descriptive word may rank above a rare direct synonym depending on writer utility.

Commonness/frequency is secondary. It may help order equally useful words, but must not create relevance on its own.

Rare, literary, archaic, or stylistically marked vocabulary is valuable for this product and should remain discoverable.

## Why V2.5 remains the retrieval baseline

The experiments summarized in `plans/semantic-search-research-summary-2026-09-15.md` show that changing embeddings, RRF weights, frequency priors, generic rerankers, static word vectors, and teacher-distilled rankers does not reliably solve the ordering problem.

However, the Phase 1 pilot also shows that V2.5 often retrieves a **useful pool of writing vocabulary** even when the internal ordering is imperfect.

For `ฝน`, top-30 included useful material such as:

- `พิรุณ`
- `พลาหก`
- `ฝนซู่`
- `โปรย`
- `ปรอย`
- `พยับเมฆ`
- `ละออง`
- `ฝนสั่งฟ้า`
- `โบกขรพรรษ`

alongside weaker/noisy items.

This supports the current architecture decision:

> **Keep V2.5 for candidate retrieval. Build a new layer that learns writer relevance and writer utility.**

Do not restart embedding/model search merely to improve the original ten-query ordering.

## Target architecture

```text
query + selected sense
        |
        v
V2.5 lexical + EmbeddingGemma retrieval
        |
        v
top-N candidate senses
        |
        v
Writer Relevance / Relation Classifier
        |
        +--> direct
        +--> subtype
        +--> broader_concept
        +--> manner_action
        +--> scene_context
        +--> effect_state
        +--> weak_related
        +--> opposite / misleading
        +--> sense mismatch
        +--> unrelated / unclear
        |
        +--> independent style/register tags
        |
        v
Writer Utility score/class
        |
        v
Hierarchical Writer Utility Ranking
        |
        v
top results / optional relation grouping
```

The classifier/ranker should answer two separate questions:

1. **How is this candidate related to the searched concept?**
2. **How useful is that relationship to a writer?**

It should not reduce the problem to:

> Can this word replace the query one-to-one?

## Human annotation contract — revised objective

Each target-sense / candidate-sense pair should receive three independent annotation axes:

1. writer utility;
2. semantic relation;
3. style/register tags.

### Writer utility

- `3` — highly useful; strong candidate to show near the top
- `2` — clearly useful for writing, but less direct / more contextual / more stylistic
- `1` — weak but potentially inspiring or useful in a narrower context
- `0` — not useful, misleading, contradictory, or noise

Utility is the main product label.

### Relation type

The relation taxonomy should support useful non-synonym vocabulary.

Schema v3 semantic classes:

- `direct` — direct lexical alternative / synonym
- `subtype` — narrower type or specific form of the concept
- `broader_concept` — a broader concept that contains the query concept
- `manner_action` — action/manner commonly used to describe the concept
- `scene_context` — surrounding scene or context useful for description
- `effect_state` — state/effect/consequence useful for description
- `weak_related` — genuinely related but low writer utility
- `opposite_misleading` — contradiction/antonym or likely to mislead
- `sense_mismatch` — wrong dictionary sense / homonym leakage
- `unrelated` — unrelated/noise
- `unclear` — cannot judge confidently

Style/register is a separate multi-label axis:

- `literary`
- `archaic`
- `formal`
- `colloquial`
- `technical`
- `dialect`
- `figurative`
- `other`
- `unknown`

An empty style-tag list means unmarked/general language.

This separation is deliberate: words such as `พรรษ` can be semantically `direct` while stylistically `literary`, and `โบกขรพรรษ` can be a `subtype` while also carrying literary/archaic style tags.

### Important annotation rules

- Different grammatical role is **not automatically wrong**.
- `ฝน -> โปรย` can be useful.
- `ฝน -> ปรอย` can be useful.
- `ฝน -> พยับเมฆ` can be useful.
- `ฝน -> เมฆ` may still have writer utility even though it is not a synonym.
- A subtype such as `ฝนซู่` should generally rank below a very strong direct alternative, but it is not a negative.
- A rare/literary word is not negative merely because it is uncommon.
- Annotators judge usefulness for writing around the selected query sense, not synonym equivalence alone.
- Utility, semantic relation, and style/register are independent axes.
- Semantic relation describes **how concepts connect**, not whether the result is good or bad.
- Utility describes **how useful** the candidate is to the writer.
- Style/register describes linguistic flavor and must not be encoded inside semantic relation.

## Phase 1 — Benchmark plumbing

Deliverables:
- [x] branch from the validated V2.5 baseline;
- [x] pilot query config using the shared regression queries;
- [x] deterministic V2.5 candidate exporter;
- [x] JSONL annotation schema plumbing;
- [x] annotation validator plumbing;
- [x] baseline metric calculator plumbing;
- [x] tests for schema/metrics/export;
- [x] document the annotation workflow;
- [x] end-to-end export of 300 pairs from 10 pilot queries on Colab.

Observed pilot result:
- EmbeddingGemma loaded successfully on CPU;
- 300 annotation pairs were exported;
- manual inspection of `ฝน#1` confirmed that V2.5 retrieves many useful writer-oriented related terms, not just synonyms;
- this inspection caused the product objective to be broadened from strict substitutability to writer lexical relevance.

Important follow-up:
- [x] revise the annotation relation enum/validator to the broader writer-relevance taxonomy above before human labeling begins.

No production ranking code changes should be made until the revised benchmark labels are in place.

### Phase 1 workflow

Export 30 V2.5 candidates per pilot target sense:

```bash
python scripts/substitutability_benchmark.py export \
  --index artifacts/v1 \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --candidates 30 \
  --output evaluation/substitutability_annotations.jsonl
```

Validate the fresh unlabeled template:

```bash
python scripts/substitutability_benchmark.py validate \
  evaluation/substitutability_annotations.jsonl \
  --allow-unlabeled
```

After validation, annotate one query at a time in Colab:

```bash
python scripts/substitutability_benchmark.py annotate \
  evaluation/substitutability_annotations.jsonl \
  --query ฝน
```

The interactive annotator:
- shows query/candidate definitions and original V2.5 rank;
- asks for Writer Utility `0..3`;
- asks for one Writer Relevance relation;
- autosaves after every completed pair;
- resumes by skipping rows that already have valid labels;
- supports `s` to skip and `q` to leave the session safely;
- supports `--review` to revisit existing labels;
- supports `--limit N` for short annotation sessions.

Use per-query annotation first so the label policy can be reviewed after each 30-candidate block before scaling to the full benchmark.

Historical note: the schema was first revised to Writer Relevance schema v2. Re-export the unlabeled pilot after pulling the latest branch before human labeling, because the first 300-row export used schema v1.

### Historical schema v2 status — superseded by v3

Implemented in `thai_substitutability.py`:
- `SCHEMA_VERSION = 2`;
- cross-role descriptive relations are valid and may receive high writer utility;
- `manner_action`, `scene_context`, `effect_state`, `subtype`, and literary relations are no longer treated as negatives;
- only intrinsically severe relations (`opposite_misleading`, `sense_mismatch`, `unrelated`) are forced to utility 0;
- writer utility and relation type remain separate labels.

Revised baseline metrics:
- `NDCG@K`;
- `Useful@K` / useful rate for utility >= 1;
- `HighUtility@K` / high-utility rate for utility >= 2;
- `Noise@K` / noise rate for utility 0;
- severe-error count/rate;
- useful relation diversity;
- reciprocal rank of the first high-utility result.

The initial Colab file `evaluation/substitutability_annotations.jsonl` was exported before schema v2. It is unlabeled, so do not migrate it manually; pull the branch and re-run the export command to regenerate the same 300 V2.5 pairs with schema v2.

Historical schema v2 test coverage contained 9 cases covering:
- deterministic sense-pair IDs;
- useful cross-role `manner_action`;
- useful `scene_context`;
- severe-error relations requiring utility 0;
- low-utility weak relations;
- exported schema-v2 rows;
- ambiguous target-sense pinning;
- writer-utility ranking metrics including NDCG and severe-error counting;
- interactive annotation autosave/resume.

The full suite should be re-run in the Colab/Kaggle environment after pulling this branch. The schema/metric logic was sanity-checked while implementing v2; the repository environment test run remains the final verification step.

Recommended verification sequence:

```bash
git pull origin feat/dictionary-substitutability-benchmark

python -m unittest discover -s tests -p 'test_substitutability_benchmark.py'

python scripts/substitutability_benchmark.py export \
  --index artifacts/v1 \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --candidates 30 \
  --output evaluation/substitutability_annotations.jsonl

python scripts/substitutability_benchmark.py validate \
  evaluation/substitutability_annotations.jsonl \
  --allow-unlabeled
```


### First human-label checkpoint — ฝน#1 (30 candidates)

The first complete human annotation block produced:

- utility 0: 11
- utility 1: 8
- utility 2: 5
- utility 3: 6

The utility scale behaved usefully, but the v2 relation taxonomy mixed semantic relation with style/register. Concrete examples:

- `ฝนซู่`, `ฝนไล่ช้าง`, `ฝนห่าแก้ว`, `ฝนสั่งฟ้า`, `ฝนหลวง` are fundamentally `subtype`, even when their writer utility differs;
- `พยับเมฆ` and `เมฆ` are better modeled as `scene_context`, not register/literary semantic relations;
- `เละ`, `น้ำป่า`, `เปียก` can be `effect_state` even when utility is 0;
- `ตก` and `ลง` can be `manner_action` even when the annotator does not consider them useful results;
- `หยาดน้ำฟ้า` motivates `broader_concept`;
- `พรรษ` shows why direct semantic relation and literary register must be separate.

Decision: **schema v2 is superseded by schema v3 before annotating the remaining pilot rows.**

### Writer Relevance schema v3

Implementation status:
- [x] schema v3 constants and validator;
- [x] independent semantic relation axis;
- [x] multi-label style/register tags;
- [x] v2 -> v3 migration preserving writer utility;
- [x] resumable annotation CLI that skips re-entering preserved utility;
- [x] per-query metrics for partially annotated pilot files;
- [x] 12 focused unit tests written for schema/migration/annotation/metrics;
- [ ] final unit-test execution on Colab/Kaggle after pulling the latest branch.

Schema v3 fields:

```json
{
  "utility": 0,
  "semantic_relation": "subtype",
  "style_tags": ["literary", "archaic"],
  "legacy_relation": "manner_action",
  "notes": ""
}
```

Rules:

- `utility` remains ordinal `0..3`;
- `semantic_relation` uses the semantic-only taxonomy above;
- `style_tags` is multi-label; `[]` means unmarked/general language;
- `legacy_relation` exists only to preserve v2 annotation history and is not training truth;
- severe semantic errors (`opposite_misleading`, `sense_mismatch`, `unrelated`) require utility 0;
- other semantic relations do **not** imply any fixed utility.

Migration policy:

- preserve all existing writer-utility labels;
- never auto-convert v2 relation labels into v3 semantic truth;
- store the old relation under `legacy_relation`;
- re-review only semantic relation + style for already utility-labeled rows.

Migration command:

```bash
python scripts/substitutability_benchmark.py migrate-v3 \
  evaluation/substitutability_annotations.jsonl \
  --output evaluation/substitutability_annotations.v3.jsonl
```

Then validate partial v3 data:

```bash
python scripts/substitutability_benchmark.py validate \
  evaluation/substitutability_annotations.v3.jsonl \
  --allow-unlabeled
```

Re-review the 30 `ฝน` rows. Existing utility values are preserved automatically:

```bash
python scripts/substitutability_benchmark.py annotate \
  evaluation/substitutability_annotations.v3.jsonl \
  --query ฝน
```

After all 30 rows are complete under v3, measure only that query even though the rest of the pilot is still unlabeled:

```bash
python scripts/substitutability_benchmark.py metrics \
  evaluation/substitutability_annotations.v3.jsonl \
  --query ฝน \
  --k 10
```


### Frozen checkpoint: ฝน#1 — schema v3 baseline

After re-reviewing the first 30 candidates under schema v3, the frozen V2.5 top-10 baseline is:

- Useful@10: **8/10 (0.80)**
- HighUtility@10: **6/10 (0.60)**
- Noise@10: **2/10 (0.20)**
- SevereError@10: **2/10 (0.20)**
- relation diversity among useful top-10: **3**
- NDCG@10: **0.5893666256**
- MRR(first utility >= 2): **1.0**

Interpretation:
- V2.5 candidate retrieval is already strong enough to surface useful writing vocabulary: 80% of visible top-10 received utility >= 1;
- ordering remains substantially improvable: NDCG@10 ~= 0.589 despite a high first result;
- 20% visible noise/severe-error rate is too high for the final product;
- this supports keeping V2.5 as retrieval while learning a writer-utility reranking layer rather than replacing retrieval.

This query is now a checkpoint, not a tuning target. Do not tune thresholds or model architecture only against ฝน#1.


### Frozen checkpoint: โกรธ#1 — schema v3 baseline

V2.5 top-10 after human annotation:

- Useful@10: **10/10 (1.00)**
- HighUtility@10: **10/10 (1.00)**
- Noise@10: **0/10 (0.00)**
- SevereError@10: **0/10 (0.00)**
- relation diversity: **2**
- NDCG@10: **0.9603250160**
- MRR(first utility >= 2): **1.0**

Compared with `ฝน#1`, this query is almost ideal. This supports the hypothesis that ranking difficulty varies by concept and lexical-neighborhood structure rather than V2.5 retrieval failing uniformly. Continue the remaining pilot queries before selecting reranker thresholds or objectives.


### Frozen checkpoint: เดิน#1 — schema v3 baseline

V2.5 top-10 after human annotation:

- Useful@10: **9/10 (0.90)**
- HighUtility@10: **8/10 (0.80)**
- Noise@10: **1/10 (0.10)**
- SevereError@10: **1/10 (0.10)**
- relation diversity: **4**
- NDCG@10: **0.8374247076**
- MRR(first utility >= 2): **1.0**

Interpretation:
- V2.5 retrieves a strong motion/action candidate pool;
- one visible severe/noise result remains;
- ranking quality is good but materially below the near-ideal `โกรธ#1`;
- the higher relation diversity (4) is consistent with motion queries mixing direct alternatives, manners/actions, nearby motion concepts, and contextual vocabulary.

### Three-query pilot snapshot

Across `ฝน#1`, `โกรธ#1`, and `เดิน#1`:

- mean Useful@10: **0.90**
- mean HighUtility@10: **0.80**
- mean Noise@10 rate: **0.10**
- mean SevereError@10 rate: **0.10**
- mean NDCG@10: **~0.7957**
- MRR(first utility >= 2): **1.0 for all three**

Early pattern: V2.5 consistently puts at least one high-utility result first and retrieves mostly useful vocabulary, while the main remaining weakness is how the rest of the top results are ordered and filtered. This continues to support V2.5 retrieval + learned writer-utility reranking.


### Frozen checkpoint: สวย#1 — schema v3 baseline

V2.5 top-10 after human annotation:

- Useful@10: **10/10 (1.00)**
- HighUtility@10: **10/10 (1.00)**
- Noise@10: **0/10 (0.00)**
- SevereError@10: **0/10 (0.00)**
- relation diversity: **2**
- NDCG@10: **1.0000000000**
- MRR(first utility >= 2): **1.0**

This is the first pilot query whose V2.5 top-10 ordering exactly matches the ideal utility ordering under the human labels.

### Four-query pilot snapshot

Across `ฝน#1`, `โกรธ#1`, `เดิน#1`, and `สวย#1`:

- mean Useful@10: **0.925**
- mean HighUtility@10: **0.850**
- mean Noise@10 rate: **0.075**
- mean SevereError@10 rate: **0.075**
- mean NDCG@10: **0.8467790873**
- MRR(first utility >= 2): **1.0 for all four**

The widening spread from NDCG 0.589 to 1.000 reinforces that query difficulty is heterogeneous. Some lexical neighborhoods are already ordered nearly perfectly by V2.5, while broader scene/context-heavy concepts remain much harder.


### Approved 10-query pilot — schema v3

The full 10-query / 300-pair pilot has now been reviewed and approved as the workflow checkpoint.

Frozen file identity:
- file: `substitutability_annotations.v3.assistant_completed.corrected.jsonl`
- SHA-256: `158d1511f6deb7e688dbe768bf095e2b4c14517f27525721ba1bd03673dc5ff9`
- rows: **300**
- target senses: **10**
- candidates per target: **30**

Final top-10 mean metrics:
- Useful@10 rate: **0.95**
- HighUtility@10 rate: **0.90**
- Noise@10 rate: **0.05**
- SevereError@10 rate: **0.05**
- relation diversity: **2.6**
- NDCG@10: **0.8509870262**
- MRR(first utility >= 2): **1.0**

Full-label distribution:
- utility 3: **185**
- utility 2: **67**
- utility 1: **27**
- utility 0: **21**

Final correction before approval:
- `บ้าน -> ที่` was corrected to `utility=0`, `semantic_relation=unrelated` because the selected sense of `ที่` means a location marker / `ณ`, not a house/place synonym.

Interpretation remains unchanged: V2.5 retrieval is strong enough to keep, while ranking/filtering quality varies substantially by query. The pilot is now frozen as a workflow/baseline checkpoint and must not be used alone for model selection.

## Phase 2 — Human-rated writer-relevance dataset

Status: **active**

Current implementation:
- [x] freeze the approved 10-query / 300-pair pilot manifest;
- [x] add a curated pool of 40 additional writer-oriented headwords, bringing the intended total to 50 target senses;
- [x] spread the pool across nature/scene, emotion, motion/posture, speech, appearance/sensory, physical state, place/environment, expression/perception/mental concepts;
- [x] add `inspect-targets` so every proposed headword is checked against the frozen V2.5 lexical artifact before its sense is frozen;
- [x] add `freeze-targets` so a reviewed sense report becomes an explicit frozen benchmark config without hand-editing query JSON;
- [x] run `inspect-targets` on Colab/Kaggle and resolve every ambiguous/missing headword;
- [x] freeze the 40 new target senses and the combined 50-target benchmark definition;
- [ ] export 30 V2.5 candidates per newly frozen target sense;
- [ ] annotate/review the expanded benchmark;
- [ ] split by target/headword family into train/validation/frozen benchmark only after labels are complete.

Sense inspection command:

```bash
python scripts/substitutability_benchmark.py inspect-targets \
  --config evaluation/writer_relevance_phase2_targets.json \
  --index artifacts/v1 \
  --output evaluation/writer_relevance_phase2_sense_report.json
```

A target is marked:
- `unique` when the headword has exactly one stored sense and can be frozen automatically;
- `explicit` when a configured sense is valid;
- `needs_review` when multiple dictionary senses exist;
- `missing_headword` when the proposed form is not an exact dictionary headword.

Do not export Phase 2 candidates until all 40 new targets have a verified intended sense.


After reviewing / filling every `recommended_sense` in the sense report, freeze the Phase-2 config:

```bash
python scripts/substitutability_benchmark.py freeze-targets \
  --report evaluation/writer_relevance_phase2_sense_report.json \
  --output evaluation/writer_relevance_phase2_frozen_queries.json
```

The freeze step refuses to continue if any target is missing a verified sense or if the selected sense is not present in the inspected dictionary senses.

Approved Phase-2 sense resolution:
- all **40/40** expansion headwords resolved;
- no missing exact headwords;
- `คลาน#1` selected for the general hand-and-knee crawling sense;
- `ฝัน#2` selected for the verbal “see/experience a story while asleep” sense;
- frozen expansion config: `evaluation/writer_relevance_phase2_frozen_queries.json`;
- combined 50-target definition: `evaluation/writer_relevance_50_targets.json`.

Export **only the 40 new targets** so the already-approved 300 pilot pairs are not regenerated:

```bash
python scripts/substitutability_benchmark.py export \
  --config evaluation/writer_relevance_phase2_frozen_queries.json \
  --index artifacts/v1 \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --candidates 30 \
  --output evaluation/writer_relevance_phase2_annotations.jsonl
```

Expected output size: **1,200 unlabeled pairs** (40 targets x 30 candidates). Validate immediately after export:

```bash
python scripts/substitutability_benchmark.py validate \
  evaluation/writer_relevance_phase2_annotations.jsonl \
  --allow-unlabeled
```


After the revised annotation schema is ready:

1. use the 10-query pilot to verify annotation workflow only;
2. expand to at least 50 frozen target senses before model selection;
3. include noun, verb, adjective/state, emotion, motion, speech, abstract concepts, scene-oriented concepts, common vocabulary, literary vocabulary, archaic forms, colloquial forms, and ambiguous headwords;
4. annotate approximately 30 V2.5 candidates per target sense;
5. deliberately preserve a mixture of:
   - direct alternatives;
   - useful descriptive/contextual vocabulary;
   - subtypes;
   - literary words;
   - hard misleading neighbors;
   - true noise;
6. split by target/headword family rather than random pair split;
7. keep a frozen benchmark separate from train/validation.

LLM judgments may be used later as annotation assistance, but human judgment of **writer usefulness** is the source of truth.

## Phase 3 — Writer Relevance / Utility model

Only after enough human labels exist:

- compare small multilingual cross-encoder/classifier candidates;
- input target word + target definition + candidate word + candidate definition + bounded lexical evidence;
- predict relation class and writer utility;
- optimize for useful ordering and rejection of misleading/noise results;
- do not treat cross-POS candidates as negative by default;
- do not promote a model solely because it improves the original 10 queries.

Potential model outputs:

```text
relation probabilities
writer_utility probability / ordinal score
confidence
```

The model should learn distinctions such as:

```text
ฝน -> พิรุณ       direct          utility 3
ฝน -> โปรย        manner_action   utility 2
ฝน -> พยับเมฆ     scene_context   utility 2
ฝน -> เปียก        effect_state    utility 1
ฝน -> กลึ้ง        unrelated       utility 0
```

## Phase 4 — Hierarchical writer-utility ranking

Do not collapse all evidence into one unconstrained semantic score.

Suggested ranking logic:

1. reject only truly misleading/unrelated/noise candidates;
2. preserve relation type;
3. prioritize writer utility;
4. use relation prior as one ranking feature;
5. use V2.5 retrieval evidence as another feature;
6. use commonness/register only as a tiebreaker among already useful candidates.

A direct synonym should usually receive a favorable prior, but the system must still allow useful descriptive words to appear prominently.

The product may later expose relation groups/categories in the UI instead of forcing every useful word into one undifferentiated list.

## Phase 5 — Offline writer-relevance graph

Once the model is stable:

```text
~52k senses
   |
V2.5 top-N candidates
   |
writer relevance / utility model
   |
precomputed directed sense edges
```

Store:
- source sense;
- candidate sense;
- relation type;
- writer utility;
- confidence;
- V2.5 retrieval evidence;
- optional register/commonness metadata.

Runtime search should primarily resolve the target sense and read precomputed ranked edges.

## Metrics — revised

The old strict `Unsafe@10 = utility-0 cross-role candidate` interpretation is no longer sufficient.

Primary product metrics should become:

- **NDCG@10 / NDCG@20** using human writer-utility labels `0..3`;
- **Useful@10** — proportion of top-10 results with utility `>=1`;
- **HighUtility@10** — count/proportion with utility `>=2`;
- **Noise@10** — utility-0 results in visible top results;
- **MRR@3** or first-high-utility reciprocal rank;
- relation diversity/coverage, monitored so the system does not collapse into synonyms only.

A separate severe-error metric should track:
- obvious antonyms/contradictions;
- unrelated noise;
- misleading homonym/sense leakage.

Do not count a useful cross-role word such as `โปรย` as unsafe merely because its grammatical role differs from `ฝน`.

## Pilot policy

The original ten queries remain a regression/pipeline pilot only. They are not large enough for model selection or threshold tuning.

The `ฝน#1` pilot specifically establishes the product requirement that **useful related vocabulary is desirable even when it is not a synonym**.

Do not tune the architecture against these ten queries. The first serious ranking/model experiment begins only after the expanded frozen human writer-relevance benchmark exists.

## Final product rule

> Thai Words should help a writer find the next useful word, not merely the nearest synonym.

Preserve meaning and relevance, favor direct useful alternatives early, but also surface actions, imagery, subtypes, context, effects, and literary vocabulary that can help the writer construct richer prose.


### Branch history note

After the first Colab checkout, the research branch was squashed/force-updated while refining schema v2. This can make an already-checked-out local branch diverge from origin. During active notebook testing, do not rewrite this branch history again. If a notebook is still on the pre-squash branch, fetch and reset the local branch to origin after preserving any local annotation file.


### Phase-2 pre-annotation checkpoint

Phase-2 annotation review is complete and **approved**.

Pre-annotation checkpoint:
- 1,200 / 1,200 Phase-2 pairs labeled under schema v3;
- 194 pairs flagged for high-priority human review;
- full CSV and review-only CSV generated separately;
- severe relations were checked to require utility 0;
- human review is complete; these labels are approved for Phase 3 development.

Approved Phase-2 top-10 means:
- Useful@10 rate: **0.9375**
- HighUtility@10 rate: **0.9000**
- Noise@10 rate: **0.0625**
- SevereError@10 rate: **0.0625**
- NDCG@10: **0.8627717380**
- MRR(first utility >= 2): **0.9875**

NDCG uses the repository metric definition: ideal DCG is built from all 30 candidates, then evaluated at K=10. The earlier provisional top-10-only ideal calculation is superseded.

The reviewed labels are frozen; proceed to Phase 3 using train/validation only until a candidate is selected.


### Frozen 50-target checkpoint

Phase 2 labels were reviewed and approved. The combined benchmark now has 50 target senses / 1,500 pairs.

- train: 31 targets / 930 pairs
- validation: 9 targets / 270 pairs
- benchmark: 10 targets / 300 pairs

See:
- `evaluation/writer_relevance_50_split_manifest.json`
- `evaluation/writer_relevance_50_metrics_approved.json`
- `evaluation/writer_relevance_50_frozen_checkpoint.md`

The split is frozen before Phase 3 experiments.


### Phase 3 baseline checkpoint

A leakage-safe learned baseline is implemented in `scripts/writer_relevance_phase3_baseline.py`.

Rules:
- fit on train only;
- evaluate on validation only;
- benchmark split is not read for model evaluation;
- features use V2.5 retrieval evidence and query category, never human labels as inputs;
- ordinal utility is modeled with cumulative binary classifiers;
- a separate severe-error probability can penalize unsafe/noisy candidates.

Validation-only result:
- V2.5: Useful@10 **0.9444**, Noise/SevereError **0.0556**, NDCG@10 **0.862307**, MRR **0.9444**
- ordinal expected utility: Useful@10 **0.9667**, Noise/SevereError **0.0333**, NDCG@10 **0.889110**, MRR **1.000**
- ordinal minus severe penalty: Useful@10 **0.9667**, Noise/SevereError **0.0333**, NDCG@10 **0.889148**, relation diversity **3.1111**, MRR **1.000**

The frozen benchmark remains untouched. This learned baseline is only a floor for Phase 3 model comparison, not yet a production candidate.


### Phase 3 cross-encoder comparison harness

A validation-only cross-encoder experiment harness is implemented in `scripts/writer_relevance_phase3_crossencoder.py`.

Experiment contract:
- train on the frozen `train` target senses only;
- evaluate on the frozen `validation` target senses only;
- keep `benchmark` rows out of model fitting, prediction, and metric calculation;
- encode target word + target definition + category against candidate word + candidate definition + bounded V2.5 lexical/dense retrieval evidence;
- never expose human utility/relation/style labels in model input;
- train writer utility as a scalar target `0..1` mapped from human utility `0..3`;
- report deltas against both V2.5 validation metrics and the Phase 3 learned-baseline floor;
- optionally emit per-query validation metrics for error analysis;
- do not select or promote a model from the original 10-query pilot.

Default first comparison candidate:
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`;
- this is an experiment default, not a production choice;
- model name, epochs, batch size, learning rate, warmup, max length, and seed are explicit CLI parameters so later candidates can be compared with the same leakage-safe harness.

Run on Colab/Kaggle with the approved 50-target annotation file present:

```bash
python scripts/writer_relevance_phase3_crossencoder.py \
  --input evaluation/writer_relevance_50_annotations.approved.jsonl \
  --include-per-query \
  --output evaluation/writer_relevance_phase3_crossencoder_report.json
```

Focused tests cover:
- model text excludes human annotation fields;
- ordinal utility maps to the expected `0..1` training target;
- metric deltas preserve direction;
- benchmark rows never reach training or prediction.

The frozen benchmark must remain untouched until a candidate is selected from train/validation evidence.

### Phase 3 MMARCO result and hybrid checkpoint

The first trained cross-encoder comparison has been completed on the frozen validation split and archived as:

- `evaluation/writer_relevance_phase3_mmarco_cpu_validation_report.json`
- model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- train: 31 target senses / 930 pairs
- validation: 9 target senses / 270 pairs
- benchmark: not evaluated and not used for training

Validation result:

- V2.5 NDCG@10: **0.862307**
- learned baseline NDCG@10: **0.889148**
- MMARCO cross-encoder NDCG@10: **0.854554**
- MMARCO Useful@10: **0.9667**
- MMARCO HighUtility@10: **0.9444**
- MMARCO Noise/SevereError: **0.0333**
- MMARCO relation diversity: **2.6667**
- MMARCO MRR: **0.9444**

Interpretation:

- the neural reranker matched the learned baseline on Useful@10 and Noise/SevereError;
- it improved HighUtility@10 slightly;
- it regressed materially on NDCG, relation diversity, and MRR;
- therefore it is not promoted as a standalone candidate;
- the result still suggests neural relevance evidence may be useful as one bounded ranking signal rather than replacing the learned baseline.

The frozen benchmark remains untouched.

### Phase 3 hybrid harness

A validation-only hybrid experiment is implemented in `scripts/writer_relevance_phase3_hybrid.py`.

Design:

- keep the leakage-safe learned baseline as the primary signal;
- consume a validation-only neural score artifact keyed by frozen `pair_id`;
- reject score artifacts containing non-validation/unknown pairs or missing validation pairs;
- normalize learned and neural scores independently within each query by average rank;
- blend with `hybrid = (1 - alpha) * learned + alpha * neural`;
- default alpha grid: `0,0.1,0.2,0.3,0.4,0.5`;
- select the validation alpha with the highest NDCG only among candidates that do not worsen learned-baseline Noise@10 or SevereError@10;
- report whether the selected alpha is an actual improvement over alpha=0;
- benchmark rows are never used for hybrid fitting, scoring, alpha selection, or metrics.

The cross-encoder harness now also supports `--score-output`, `--no-finetune`, `--device`, `--trust-remote-code`, and `--use-amp`.

### Next Kaggle experiment

Colab GPU quota was unavailable for this round, so the next GPU experiment should run on Kaggle.

The next research candidate is `BAAI/bge-reranker-v2-m3`:

- multilingual reranker;
- Apache-2.0 model license;
- run pretrained/no-finetune first before spending GPU time on fine-tuning.

Validation-only scoring:

```bash
python -m scripts.writer_relevance_phase3_crossencoder \
  --input evaluation/writer_relevance_50_annotations.approved.jsonl \
  --model BAAI/bge-reranker-v2-m3 \
  --no-finetune \
  --device cuda \
  --eval-batch-size 8 \
  --max-length 384 \
  --score-output evaluation/writer_relevance_phase3_bge_v2_m3_validation_scores.jsonl \
  --include-per-query \
  --output evaluation/writer_relevance_phase3_bge_v2_m3_validation_report.json
```

Hybrid grid:

```bash
python -m scripts.writer_relevance_phase3_hybrid \
  --input evaluation/writer_relevance_50_annotations.approved.jsonl \
  --neural-scores evaluation/writer_relevance_phase3_bge_v2_m3_validation_scores.jsonl \
  --alphas 0,0.1,0.2,0.3,0.4,0.5 \
  --include-per-query \
  --output evaluation/writer_relevance_phase3_bge_v2_m3_hybrid_report.json
```

Decision rule before fine-tuning:

1. compare neural-only BGE against the learned baseline;
2. inspect the full hybrid alpha grid;
3. require no Noise/SevereError regression for the selected alpha;
4. require a real validation NDCG improvement over alpha=0 before treating the neural signal as additive;
5. if no hybrid alpha improves the learned baseline, do not fine-tune this model merely to chase the nine validation queries;
6. keep the frozen benchmark closed until a candidate architecture is selected from train/validation evidence.


### Phase 3 hybrid stability checkpoint

A leave-one-query-out stability analyzer is implemented in `scripts/writer_relevance_phase3_hybrid_stability.py`.

It consumes the already-generated hybrid report, so no model reload, GPU, annotation file, or neural rescoring is required. The analyzer:

- reconstructs mean metrics from the stored per-query metrics for every alpha;
- leaves out one validation query at a time;
- selects alpha from the remaining eight queries using the same NDCG + Noise/SevereError rule as the hybrid harness;
- evaluates that selected alpha only on the held-out query;
- reports per-query NDCG / safety / diversity deltas;
- refuses reports that evaluated benchmark rows or used benchmark rows for alpha selection.

Predeclared stability gate:

1. positive neural weight must be selected in at least two-thirds of leave-one-query-out folds;
2. mean held-out NDCG delta must be positive;
3. held-out NDCG wins must be at least as numerous as losses;
4. there must be zero held-out Noise@10 regressions;
5. there must be zero held-out SevereError@10 regressions;
6. relation diversity is reported diagnostically but does not gate the pass yet.

Observed BGE hybrid stability result from the validation report:

- source model: `BAAI/bge-reranker-v2-m3`;
- validation queries: **9**;
- full-validation selected alpha: **0.4**;
- leave-one-query-out alpha selection: **0.4 in 8/9 folds**, **0.2 in 1/9 fold**;
- positive-alpha folds: **9/9**;
- held-out NDCG: **5 wins / 3 ties / 1 loss**;
- mean held-out NDCG delta: **+0.007288**;
- min held-out NDCG delta: **-0.051552**;
- max held-out NDCG delta: **+0.044799**;
- held-out Noise regressions: **0/9**;
- held-out SevereError regressions: **0/9**;
- diversity regressions: **4/9**;
- mean held-out diversity delta: **-0.444444**;
- stability gate: **PASS**.

Held-out failure / diagnostic cases:

- `ห้อง#1` is the only NDCG loss at the alpha selected from the other eight queries: delta **-0.051552**, with relation diversity delta **-2**;
- `ประตู#1` improves NDCG but loses relation diversity by **2**;
- `หอม#5` improves NDCG but loses relation diversity by **1**;
- `เปียก#1` ties on NDCG but loses relation diversity by **1**.

Interpretation:

- the BGE signal is not dependent on one validation query; the selected weight remains highly stable under leave-one-query-out analysis;
- safety behavior generalizes across all nine held-out folds;
- the architecture is therefore promoted from an exploratory blend to the current Phase 3 candidate architecture;
- relation-diversity collapse remains the primary unresolved trade-off and must be monitored in the next challenger;
- the frozen benchmark remains closed.

Run the stability analyzer on Kaggle/locally from the existing hybrid report:

```bash
python -m scripts.writer_relevance_phase3_hybrid_stability \
  --hybrid-report evaluation/writer_relevance_phase3_bge_v2_m3_hybrid_report.json \
  --output evaluation/writer_relevance_phase3_bge_v2_m3_stability_report.json
```

Next challenger before opening the frozen benchmark:

1. fine-tune `BAAI/bge-reranker-v2-m3` on the 31-query train split only;
2. generate validation-only scores;
3. run the same hybrid alpha grid and stability analyzer;
4. compare pretrained-BGE hybrid vs fine-tuned-BGE hybrid on validation and stability;
5. require no safety regression and explicitly inspect relation diversity, especially `ห้อง#1`;
6. select one architecture/hyperparameter configuration before the single frozen-benchmark evaluation.


### Kaggle multi-GPU fine-tune compatibility

The first fine-tuned BGE challenger attempt on Kaggle failed before the first optimizer step because the notebook exposed multiple GPUs. Hugging Face Trainer therefore entered single-process `torch.nn.DataParallel`, while the Sentence Transformers cross-encoder BinaryCrossEntropy loss accessed `self.model.device`; `DataParallel` does not expose that attribute in this path.

Observed failure:

```text
AttributeError: 'DataParallel' object has no attribute 'device'
```

This is infrastructure/trainer behavior, not a dataset, label, BGE-weight, or benchmark failure. The failed run produced no challenger result and must not be counted as an experiment outcome.

The harness now accepts `--cuda-visible-devices` and applies `CUDA_VISIBLE_DEVICES` before importing Torch/SentenceTransformers. For the current 930-pair train split, use one GPU rather than DP/DDP:

```bash
python -m scripts.writer_relevance_phase3_crossencoder \
  --input evaluation/writer_relevance_50_annotations.approved.jsonl \
  --model BAAI/bge-reranker-v2-m3 \
  --device cuda \
  --cuda-visible-devices 0 \
  --use-amp \
  --epochs 1 \
  --batch-size 2 \
  --eval-batch-size 4 \
  --learning-rate 1e-5 \
  --warmup-ratio 0.1 \
  --max-length 384 \
  --seed 42 \
  --score-output evaluation/writer_relevance_phase3_bge_v2_m3_finetuned_validation_scores.jsonl \
  --include-per-query \
  --output evaluation/writer_relevance_phase3_bge_v2_m3_finetuned_validation_report.json
```

The resulting report records `cuda_visible_devices` so the execution topology is auditable.

For future large-scale training, Sentence Transformers recommends distributed launchers such as `torchrun`/Accelerate (DDP) over single-process DataParallel. That complexity is intentionally deferred here because the Phase 3 train split is only 930 pairs.


### Phase 3 fine-tuned BGE candidate lock

The one-epoch fine-tuned `BAAI/bge-reranker-v2-m3` challenger completed successfully on a single Kaggle GPU after the DataParallel compatibility fix.

Fine-tune configuration:

- train split only: **31 target senses / 930 pairs**
- epochs: **1**
- batch size: **2**
- evaluation batch size: **4**
- learning rate: **1e-5**
- warmup ratio: **0.1**
- max length: **384**
- seed: **42**
- AMP: enabled
- visible CUDA devices: **0**
- benchmark rows used for training: **0**

Fine-tuned neural-only validation result:

- Useful@10: **0.988889**
- HighUtility@10: **0.988889**
- Noise@10: **0.011111**
- SevereError@10: **0.011111**
- relation diversity: **2.555556**
- NDCG@10: **0.900274**
- MRR high utility: **1.000000**

This is the first neural-only candidate to exceed the learned-baseline validation NDCG (**0.889148**) while also reducing Noise/SevereError.

Hybrid validation grid selected **alpha=0.5**:

- learned-baseline weight: **0.5**
- fine-tuned BGE weight: **0.5**
- Useful@10: **0.988889**
- HighUtility@10: **0.966667**
- Noise@10: **0.011111**
- SevereError@10: **0.011111**
- relation diversity: **2.555556**
- NDCG@10: **0.918968**
- MRR high utility: **1.000000**

Compared with the pretrained-BGE hybrid incumbent (alpha=0.4, NDCG **0.907113**), the fine-tuned hybrid improves NDCG by about **+0.011855** while keeping the same aggregate Noise/SevereError rate.

Fine-tuned hybrid leave-one-query-out stability:

- selected alpha: **0.5 in 7/9 folds**, **0.4 in 2/9 folds**
- positive-alpha folds: **9/9**
- held-out NDCG: **5 wins / 3 ties / 1 loss**
- mean held-out NDCG delta: **+0.021372**
- minimum held-out NDCG delta: **-0.088772**
- maximum held-out NDCG delta: **+0.112721**
- held-out Noise regressions: **0/9**
- held-out SevereError regressions: **0/9**
- held-out diversity regressions: **4/9**
- mean held-out diversity delta: **-0.333333**
- stability gate: **PASS**

`ห้อง#1` remains the main failure case: at the selected held-out alpha its NDCG drops by about **-0.088772**. Relation diversity remains the principal trade-off, although the mean held-out diversity loss is slightly smaller than with the pretrained-BGE hybrid.

Decision:

> Lock **fine-tuned BGE + learned baseline, alpha=0.5** as the Phase 3 candidate before the frozen benchmark.

The locked machine-readable configuration is stored in:

- `evaluation/writer_relevance_phase3_locked_candidate.json`

No further model selection, alpha tuning, learning-rate tuning, epoch tuning, or validation-driven architecture changes are allowed before the first frozen-benchmark evaluation.

### Reproducible checkpoint requirement before benchmark

The successful challenger run did not specify `--model-output`, so its fine-tuned weights were not intentionally preserved as the locked artifact. The frozen benchmark must therefore remain closed until the exact locked training configuration is rerun with a saved model checkpoint and its validation report is checked against the locked candidate.

Rerun the same configuration, adding only `--model-output`:

```bash
python -m scripts.writer_relevance_phase3_crossencoder \
  --input evaluation/writer_relevance_50_annotations.approved.jsonl \
  --model BAAI/bge-reranker-v2-m3 \
  --device cuda \
  --cuda-visible-devices 0 \
  --use-amp \
  --epochs 1 \
  --batch-size 2 \
  --eval-batch-size 4 \
  --learning-rate 1e-5 \
  --warmup-ratio 0.1 \
  --max-length 384 \
  --seed 42 \
  --model-output artifacts/phase3/bge-reranker-v2-m3-locked \
  --score-output evaluation/writer_relevance_phase3_bge_v2_m3_locked_validation_scores.jsonl \
  --include-per-query \
  --output evaluation/writer_relevance_phase3_bge_v2_m3_locked_validation_report.json
```

Then run the existing hybrid + stability scripts against the saved-run validation scores. If the locked rerun materially changes candidate selection or fails the predeclared stability gate, do not open the benchmark; investigate reproducibility first rather than tuning against benchmark data.

### Frozen benchmark one-shot harness

`scripts/writer_relevance_phase3_frozen_benchmark.py` is the only intended Phase 3 path for opening the frozen benchmark.

Guards:

- requires the explicit `--confirm-frozen-benchmark` flag;
- verifies the approved 1,500-row dataset SHA-256;
- verifies exact frozen train/validation/benchmark query IDs and pair counts;
- loads the already-saved fine-tuned checkpoint instead of fitting a neural model;
- fits the learned baseline on the train split only;
- reads the locked alpha (**0.5**) from the candidate manifest;
- does not contain an alpha-grid/model-selection step;
- reports V2.5, learned baseline, neural-only, and locked-hybrid benchmark metrics;
- records that no benchmark-based selection or alpha reselection occurred.

Do not run this command until the saved checkpoint reproduction step above is accepted. When ready, the single benchmark command is:

```bash
python -m scripts.writer_relevance_phase3_frozen_benchmark \
  --input evaluation/writer_relevance_50_annotations.approved.jsonl \
  --candidate-manifest evaluation/writer_relevance_phase3_locked_candidate.json \
  --split-manifest evaluation/writer_relevance_50_split_manifest.json \
  --model-path artifacts/phase3/bge-reranker-v2-m3-locked \
  --device cuda \
  --include-per-query \
  --confirm-frozen-benchmark \
  --output evaluation/writer_relevance_phase3_frozen_benchmark_report.json
```

After this command is run once, treat the benchmark as opened. Do not change the locked candidate in response to that result.


### Reproducible locked checkpoint accepted

The locked one-epoch fine-tuned BGE configuration was rerun with `--model-output` so the exact benchmark candidate now exists as a saved checkpoint:

- saved model: `artifacts/phase3/bge-reranker-v2-m3-locked`
- model: `BAAI/bge-reranker-v2-m3`
- training split: 31 queries / 930 pairs
- benchmark rows used for training: 0
- selected hybrid alpha: **0.5**, unchanged from model selection

Saved-checkpoint reproduction validation:

- Useful@10: **0.988889**
- HighUtility@10: **0.977778**
- Noise@10: **0.011111**
- SevereError@10: **0.011111**
- relation diversity: **2.555556**
- NDCG@10: **0.913871**
- MRR high utility: **1.000000**

The earlier candidate-selection run produced NDCG **0.918968**. The saved-checkpoint rerun is lower by about **0.005097**, but it preserves the selected alpha, safety metrics, MRR, and still exceeds the pretrained-BGE hybrid incumbent NDCG **0.907113**.

Saved-checkpoint leave-one-query-out stability:

- alpha selection: **0.5 in 8/9 folds**, **0.3 in 1/9 fold**
- positive-alpha folds: **9/9**
- held-out NDCG: **5 wins / 3 ties / 1 loss**
- mean held-out NDCG delta: **+0.019126**
- minimum held-out NDCG delta: **-0.091648**
- maximum held-out NDCG delta: **+0.078536**
- held-out Noise regressions: **0/9**
- held-out SevereError regressions: **0/9**
- held-out diversity regressions: **4/9**
- mean held-out diversity delta: **-0.444444**
- stability gate: **PASS**

Interpretation:

- the training run is not numerically bit-identical across reruns, but the architecture and selected alpha are stable;
- the reproduced checkpoint remains better than the pretrained hybrid on validation NDCG;
- safety behavior remains unchanged;
- the predeclared stability gate still passes;
- this is sufficient to accept the saved checkpoint without any further tuning.

Decision:

> The Phase 3 candidate is now reproducibly locked. The frozen benchmark may be opened once using the existing one-shot harness and alpha **0.5**. Do not perform any additional validation-driven tuning before or after that benchmark evaluation.

The frozen benchmark command remains:

```bash
python -m scripts.writer_relevance_phase3_frozen_benchmark \
  --input evaluation/writer_relevance_50_annotations.approved.jsonl \
  --candidate-manifest evaluation/writer_relevance_phase3_locked_candidate.json \
  --split-manifest evaluation/writer_relevance_50_split_manifest.json \
  --model-path artifacts/phase3/bge-reranker-v2-m3-locked \
  --device cuda \
  --include-per-query \
  --confirm-frozen-benchmark \
  --output evaluation/writer_relevance_phase3_frozen_benchmark_report.json
```

Once this command has been run, treat the benchmark as opened. Record the result, but do not use it to change alpha, model architecture, learning rate, epoch count, or other Phase 3 selection choices.


### Checkpoint persistence incident before frozen benchmark

The first frozen-benchmark command failed before model prediction because the expected saved model directory did not exist:

```text
FileNotFoundError: Saved fine-tuned model not found:
artifacts/phase3/bge-reranker-v2-m3-locked
```

Root cause:

- the legacy `CrossEncoder.fit(..., output_path=..., save_best_model=False)` path used by the harness did not reliably persist the final model in the installed Sentence Transformers version;
- validation and stability reproduction had succeeded, but the intended checkpoint artifact was therefore absent;
- no frozen-benchmark model prediction or benchmark metric report was produced by the failed command.

Fix:

- training now calls `CrossEncoder.save_pretrained(model_output)` explicitly after `fit()`;
- the save helper verifies that the output directory contains files and raises immediately otherwise;
- validation reports now expose `model_checkpoint_output` and `model_checkpoint_saved`;
- the frozen-benchmark harness now loads/verifies the saved checkpoint **before reading the frozen dataset**, so future missing/corrupt checkpoint failures do not touch benchmark rows.

The previous validation/stability reproduction remains valid evidence for the locked configuration, but `frozen_benchmark_ready` is temporarily reset to false until the same locked training command is rerun after this persistence fix and the saved directory is verified.

Do not alter the locked model configuration or alpha during this rerun.


### Verified persisted checkpoint — final pre-benchmark gate

The explicit `save_pretrained` rerun completed successfully and the locked model artifact is now physically present at:

- `artifacts/phase3/bge-reranker-v2-m3-locked`
- `model_checkpoint_saved: true`
- persisted files include `config.json`, `model.safetensors`, tokenizer/config files, and Sentence Transformers metadata.

The persisted checkpoint's neural-only validation NDCG is **0.902097**.

Using the already locked alpha **0.5**, the persisted-checkpoint hybrid validation result is:

- Useful@10: **0.988889**
- HighUtility@10: **0.966667**
- Noise@10: **0.011111**
- SevereError@10: **0.011111**
- relation diversity: **2.555556**
- NDCG@10: **0.921145**
- MRR high utility: **1.000000**

This is above both the learned baseline (**0.889148**) and the pretrained-BGE hybrid incumbent (**0.907113**) on validation NDCG while preserving the locked alpha and safety profile.

Persisted-checkpoint leave-one-query-out stability:

- alpha **0.5 selected in 9/9 folds**
- held-out NDCG: **5 wins / 3 ties / 1 loss**
- mean held-out NDCG delta: **+0.031996**
- min / max held-out NDCG delta: **-0.081425 / +0.112721**
- Noise regressions: **0/9**
- SevereError regressions: **0/9**
- diversity regressions: **4/9**
- mean held-out diversity delta: **-0.555556**
- stability gate: **PASS**

The main known validation failure case remains `ห้อง#1`, where the locked hybrid loses NDCG relative to the learned baseline. This is recorded as diagnostic evidence only and does not change the locked configuration.

Decision:

> Artifact persistence is verified. The Phase 3 candidate remains **fine-tuned BGE + learned baseline, alpha=0.5**. The frozen benchmark is now ready for its single intended evaluation. No further validation-driven tuning is allowed before running it.
