# Semantic Search Research Summary — 2026-09-15

Status: **V2.5 is the validated baseline. Pause ranking/model experiments here.**

Baseline branch: `feat/dictionary-semantic-v2-5-embeddinggemma`

Project objective: Thai lexical search / word repository for fiction writers. The ranking target is **lexical substitutability and writer usefulness**, not generic semantic relatedness.

## Current baseline — V2.5

V2.5 remains the strongest validated architecture after all experiments below.

Architecture:

```text
Thai dictionary (~50k entries / ~52k senses)
        |
        +--> PyThaiNLP lexical pipeline
        |      - newmm tokenization
        |      - Thai stopwords
        |      - custom headword Trie
        |      - sense-level TF-IDF
        |      - forward/reverse dictionary references
        |      - lexical relation tiers
        |
        +--> EmbeddingGemma dense retrieval
               google/embeddinggemma-300m
               native encode_query / encode_document
               256d Matryoshka profile
        |
        v
weighted RRF + protected strong lexical relations
        |
        v
top results
```

Preferred dense artifact:

```text
artifacts/v2/embeddinggemma-300m-256
```

Why it remains the baseline:
- strongest overall balance of semantic recall and lexical validity;
- much faster to build than larger challengers;
- simple runtime;
- no external API required;
- avoids many severe semantic-neighbor regressions introduced by later rerankers.

## Shared qualitative benchmark

The recurring 10-query set is:

- ฝน
- โกรธ
- เดิน
- สวย
- มืด
- รัก
- พูด
- เร็ว
- กลัว
- บ้าน

Important: this set is useful for regression detection but is **too small to justify repeated tuning**. Do not keep modifying heuristics to make these ten queries look better.

## Experiment history

### V1 — lexical baseline

Branch: `feat/dictionary-semantic-v1`

Built:
- PyThaiNLP newmm tokenization;
- custom headword dictionary;
- sense-level TF-IDF;
- direct/reverse dictionary-reference evidence;
- relation-strength heuristics.

Result:
- strong direct dictionary relations;
- weak on semantic matches whose definitions use different wording.

Decision:
- keep lexical system as one half of all later hybrid retrieval.

### V2 — multilingual dense retrieval

Branch: `feat/dictionary-semantic-v2`

Tested multilingual E5 dense retrieval while retaining V1 lexical search.

Result:
- dense retrieval improved semantic recall;
- established the hybrid lexical+dense architecture.

Decision:
- continue hybrid architecture.

### V2.5 — EmbeddingGemma

Branch: `feat/dictionary-semantic-v2-5-embeddinggemma`

Tested:
- `google/embeddinggemma-300m` 768d;
- `google/embeddinggemma-300m` truncated to 256d;
- native asymmetric retrieval methods `encode_query()` / `encode_document()`.

Result:
- best overall retrieval quality;
- 256d kept quality competitive while reducing artifact size;
- fresh Colab T4 build of 52,004 senses took ~260s for 256d;
- selected as baseline.

**Decision: KEEP.**

### Local embedding challengers

Branch: `feat/dictionary-semantic-v26-local-embedding-benchmark`

Compared at 256d:
- V2.5 EmbeddingGemma;
- Qwen3-Embedding-0.6B;
- Snowflake Arctic Embed L v2.0.

Observed:
- Qwen3 had some local wins but semantic regressions, e.g. `เร็ว -> ช้า`;
- Arctic-L helped some queries such as `เดิน / เร็ว` but regressed others such as `ฝน`;
- neither fixed the key `บ้าน` ordering issue;
- Qwen3 index build was ~5.1x slower than EmbeddingGemma on the same T4;
- Arctic-L was ~1.85x slower.

Decision:
- no challenger consistently beat V2.5;
- keep artifacts only for research/recall experiments.

### Gemini Embedding 2 experiment

Branch: `feat/dictionary-semantic-v26-gemini-embedding-2`

Implemented:
- Gemini Embedding 2 dense artifact builder;
- 256d and 768d paths;
- resumable API index building;
- retry/backoff;
- V2 hybrid compatibility.

Project-level outcome:
- it was not promoted over V2.5;
- later embedding comparisons still concluded V2.5 was the preferred baseline.

Do not introduce an API dependency unless a future frozen benchmark demonstrates a material win.

### V3 — teacher labels + EmbeddingGemma fine-tuning

Branch: `feat/dictionary-semantic-v3-teacher-finetune`

Goal:
- generate grounded relation labels;
- fine-tune EmbeddingGemma using positives/hard negatives.

Problem:
- training quality depends heavily on trustworthy relation labels;
- unsafe hard negatives can damage an already-good embedding space.

Decision:
- not promoted;
- V2.5 retained.

### V3.1 — local teacher router

Branch: `feat/dictionary-semantic-v3-1-local-teacher-router`

Tested:
- Qwen local teacher;
- routing only uncertain candidates;
- Gemini audit path;
- compact direct selector.

Real finding:
- high-confidence labels still confused:
  - near-synonym vs associated;
  - subtype/supertype direction;
  - useful related forms vs hard negatives.

Decision:
- pause teacher-generated fine-tuning;
- do not modify EmbeddingGemma space with these labels.

### V4 — pointwise instruction-aware reranking

Branch: `feat/dictionary-semantic-v4-instruction-reranker`

Tested:
- Qwen3-Reranker 0.6B;
- larger 4B capacity control;
- pure rerank;
- fusion;
- protected mode;
- strict substitutability prompts;
- TNC commonness;
- gated commonness;
- capped promotion;
- token-aware familiarity.

Wins:
- could surface useful words such as `งดงาม / ฉับไว / หวาดกลัว / เรือน`.

Persistent failures:
- `เดิน -> วิ่ง`;
- `ฝน -> น้ำตก`;
- `รัก -> คู่รัก`;
- `มืด -> หน้ามืด / เสียสายตา`.

Important lesson:
- raw commonness cannot create relevance;
- TNC is useful only after semantic/lexical validity is established.

Decision:
- V4 did not replace V2.5;
- move research to listwise ranking.

### V5 — listwise / instruction-following ranking

Branch: `feat/dictionary-semantic-v5-listwise-ranker`

Tested multiple directions:

#### Jina reranker v3.5

Strengths:
- improved relative semantic ordering;
- notably improved `เดิน`;
- ~3.2s/query in the pilot.

Weaknesses:
- rare dictionary forms still dominated `บ้าน / สวย / เร็ว`;
- research/non-commercial licensing concern.

#### Qwen3.5 generative listwise

Goal:
- directly follow product ranking instructions.

Result:
- small generative ranking was not reliable enough to become the solution.

#### ContextualAI instruction reranker

Result:
- slower (~8.9s/query);
- still produced lexical-role/association errors;
- did not solve common-vs-rare ordering consistently.

#### Gemma 4 E2B direct judge

Model:
`google/gemma-4-E2B-it-qat-mobile-transformers`

This was the strongest quality-oriented reranking experiment.

Strong wins included:
- `เดิน`: walking words ahead of `วิ่ง`;
- `สวย`: `งดงาม` near/top;
- `เร็ว`: common forms such as `ไว / รวดเร็ว / ฉับไว`;
- `บ้าน`: `บ้านเรือน` moved dramatically upward.

Remaining leaks:
- `ฝน -> ตก / เมฆ`;
- `รัก -> สารภาพ / ที่รัก`;
- `มืด -> บอด`;
- some overly general words such as `ดี`.

Runtime:
- roughly 19–22s/query on top-50 direct ranking;
- too slow for normal interactive search.

Decision:
- valuable teacher/research signal;
- not production runtime;
- later V2.7 distilled this idea and failed the frozen holdout.

### V2.6 — bounded TNC rarity reranker

Branch: `feat/dictionary-semantic-v2-6-lightweight-reranker`

Architecture:
- unchanged V2.5 retrieval;
- PyThaiNLP TNC unigram frequency;
- rarity as a negative prior only;
- structural penalties;
- semantic/relation guards;
- bounded rank movement.

A bug in the first run allowed unlimited downward displacement; it was corrected so movement was bounded both directions.

Corrected result:
- some useful improvements:
  - `เดิน`: ย่าง / ย่างเท้า improved;
  - `สวย`: งดงาม improved;
  - `บ้าน`: หมู่บ้าน improved;
- but improvements were inconsistent;
- TNC rarity could not determine lexical replaceability.

Decision:
- reject V2.6;
- do not keep tuning frequency thresholds.

### V2.7 — Gemma-distilled linear pairwise ranker

Branch: `feat/dictionary-semantic-v2-7-pairwise-ranker`

Training:
- 96 non-holdout anchors;
- 18 V2.5 candidates/anchor;
- Gemma 4 offline teacher top-8;
- 9,984 pairwise examples;
- 19 runtime features;
- linear Logistic Regression;
- frozen 51-word holdout excluded from anchors and candidates;
- validation accuracy: 0.629555.

Some wins:
- `สวย`: งดงาม #8 -> #3;
- `เร็ว`: รวดเร็ว #9 -> #5;
- `บ้าน`: หมู่บ้าน #8 -> #2;
- `พูด`: เอ่ย surfaced strongly.

But major regressions:
- `ฝน -> เมฆ #3`;
- `เดิน -> ชาย`;
- `สวย -> ดี`;
- `รัก -> ชู้สาว / จอด`;
- `พูด -> พจน์ / ปาก`;
- `เร็ว -> สีฆ-`;
- `กลัว -> กระดก`.

Decision:
- **REJECTED**;
- do not tune the 10-query benchmark further;
- keep V2.5.

### PyThaiNLP static word vectors

Branch: `feat/dictionary-semantic-pythainlp-wordvectors-benchmark`

Tested:
- `thai2fit_wv`, 300d;
- raw headword cosine;
- mean word-vector representation of sense definitions.

Coverage:
- headword coverage: 40.0%;
- sense mean coverage: 100%.

Useful examples:
- `โกรธ -> โมโห`;
- `สวย -> งดงาม`;
- `พูด -> พูดจา / กล่าว`;
- `บ้าน -> หมู่บ้าน / บ้านพัก`.

Core failure:
static Word2Vec captures distributional association rather than lexical substitution:
- `ฝน -> พายุ / หิมะ / มรสุม`;
- `เดิน -> วิ่ง / แล่น / ปีน`;
- `มืด -> สว่าง`;
- `เร็ว -> ช้า / ล่าช้า`;
- `กลัว -> เกลียด / โกรธ`.

Sense-mean mode was much worse despite 100% nominal coverage because generic definition vocabulary dominated the averaged vector.

Decision:
- reject thai2fit for this product objective;
- do not proceed to LTW2V merely for larger vocabulary because it does not solve the association-vs-substitution mismatch.

## Consolidated lessons

1. **EmbeddingGemma retrieval is not the main bottleneck.**
   Later embedding swaps did not consistently improve the candidate pool.

2. **Lexical substitution is not generic semantic relevance.**
   Models repeatedly promote antonyms, associated concepts, manners, subtypes, compounds, or other grammatical roles.

3. **Frequency/commonness is secondary.**
   It can break ties among valid substitutes but must never create relevance.

4. **Static word vectors are the wrong objective.**
   Distributional neighbors are often exactly the wrong words for replacement.

5. **Teacher/reranker quality can look impressive while still being unsafe.**
   A few large common-first wins do not compensate for semantic-role regressions.

6. **Do not overfit the ten-query benchmark.**
   Future work should first build a larger human-reviewed frozen benchmark (30–50+ diverse senses) before another ranking architecture experiment.

7. **Rare/literary vocabulary is not inherently bad.**
   Thai Words is for fiction writers. Rare words should stay available; the goal is ordering common/natural forms earlier when equally valid.

## Final decision as of 2026-09-15

**Keep V2.5 unchanged as the primary baseline.**

Do not promote:
- V3/V3.1 teacher fine-tuning;
- V4 rerankers/commonness heuristics;
- V5 runtime rerankers;
- V2.6 rarity reranker;
- V2.7 learned pairwise ranker;
- PyThaiNLP static word vectors;
- local embedding challengers.

Preserve all branches as research history.

## Recommended next step when research resumes

Do **not** begin by trying another embedding/reranker model.

First:
1. build a larger frozen human-rated writer-relevance benchmark;
2. define explicit labels such as:
   - 3 = direct / excellent substitute;
   - 2 = useful near-synonym / register shift;
   - 1 = contextual / narrow alternative;
   - 0 = associated, wrong role, antonym, misleading;
3. include common, literary, archaic, colloquial, and grammatical-role edge cases;
4. measure NDCG@10, MRR, and Unsafe@10;
5. only then evaluate another architecture.

Until that exists, V2.5 should remain the stable reference point.

## Handoff for a new chat

Start from:

```text
repo: SealNM/thai_word
baseline branch: feat/dictionary-semantic-v2-5-embeddinggemma
baseline: V2.5 EmbeddingGemma 300M 256d + PyThaiNLP lexical hybrid
```

Read this file first before proposing another model experiment.

The most important product rule is:

> Rank by lexical usefulness for writers: preserve sense and grammatical role first; commonness only breaks ties among valid substitutes; keep useful rare/literary words lower rather than deleting them.
