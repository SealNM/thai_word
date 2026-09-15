# Thai Words — Writer Lexical Relevance & Utility Ranking Plan

Date: 2026-09-15  
Status: **In progress — schema v3 approved after first 30 human labels**  
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

## Phase 2 — Human-rated writer-relevance dataset

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
