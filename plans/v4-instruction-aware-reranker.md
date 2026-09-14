# V4 — V2.5 Candidate Retrieval + Instruction-Aware Reranking

Status: **core implementation in progress**

Branch: `feat/dictionary-semantic-v4-instruction-reranker`

Base: `feat/dictionary-semantic-v3-1-local-teacher-router`

V4 changes the optimization target after the V3/V3.1 experiments showed that relation-ontology labels and hard-negative generation were not reliable enough to improve the already-strong V2.5 embedding space.

## Goal

Optimize directly for the Thai Words product question:

> If a writer searches for this Thai word/sense, which candidate words are most useful to see first as exact, near, literary, archaic, poetic, or register-shifted substitutes?

V4 does **not** fine-tune EmbeddingGemma initially. V2.5 remains the candidate retriever and a cross-encoder reranker improves precision over a small candidate pool.

## Architecture

```text
Thai query / selected sense
          |
          v
     V2.5 hybrid
  lexical + dense
          |
      top 50
          |
          v
 instruction-aware
     reranker
          |
          +-----------------------------+
          |              |              |
       rerank          fusion        protected
       only        V2.5 + rerank    Tier-5 safe prior
          |              |              |
          +--------------+--------------+
                         |
                       top 10
```

## Why this replaces the current V3 direction

V3/V3.1 attempted to classify fine-grained semantic relations and compile them into positive/hard-negative training pairs. Manual inspection found high-confidence ontology mistakes, including:

- near-synonym vs associated
- subtype/supertype direction
- useful related substitute vs unsafe hard negative
- closely related word forms incorrectly pushed apart

These errors are especially dangerous when they are used to modify the global embedding space.

V4 instead asks a narrower, product-aligned question at query time: rank a grounded set of dictionary candidates by usefulness to a writer.

## Phase A — Preserve V2.5 retrieval

- [x] Keep V2.5 hybrid retrieval unchanged.
- [x] Default reranking pool: top 50 V2.5 candidates.
- [x] Reranker sees the selected query sense when available.
- [x] Candidate text includes the candidate word and matched dictionary definition.
- [x] Do not include V2.5 relation labels inside reranker text, so the reranker can be evaluated independently.

## Phase B — Instruction-aware reranker

Primary model:

- `Qwen/Qwen3-Reranker-0.6B`
- Sentence Transformers `CrossEncoder`
- custom Thai Words task instruction instead of the model's default web-search instruction

Default task instruction:

```text
Given a Thai dictionary query and a candidate dictionary entry, score how useful
the candidate is as a substitute or closely usable alternative for a writer.
Prefer exact synonyms and near-synonyms, including literary, archaic, poetic,
formal, colloquial, or register-shifted alternatives when they preserve the core
meaning. Penalize antonyms, words that are only topically associated, accidental
definition overlap, and candidates that substantially change the core meaning.
```

Control model:

- `BAAI/bge-reranker-v2-m3`
- no Thai Words instruction by default
- use only as a comparison model, not as the initial production dependency

## Phase C — Ranking variants

Implement and compare all three without additional model calls:

### V4-A: `rerank`

Pure reranker ordering.

Purpose:
- measure whether the instruction-aware model alone improves V2.5 ordering.

### V4-B: `fusion`

Weighted reciprocal-rank fusion between:
- original V2.5 rank
- reranker rank

The fusion is rank-based rather than mixing raw logits, because different reranker models have incompatible score scales.

Default:
- V2.5 rank weight: 0.35
- reranker rank weight: 1.0
- RRF k: 20

### V4-C: `protected`

Protect only the narrowest high-precision lexical evidence:
- relation tier >= 5
- standalone lexical form
- relation hint is `direct_gloss_or_synonym` or `mutual_definition_reference`

Within the protected and unprotected groups, order by reranker score.

This is the preferred hypothesis going into the first pilot, but it must beat the alternatives in evaluation.

## Phase D — Qualitative pilot

Use the existing cross-domain query set first:

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

For each query compare:
- V2.5 baseline
- V4-A rerank
- V4-B fusion
- V4-C protected

Pilot gates:

- [ ] Run Qwen3-Reranker-0.6B on the 10-query set.
- [ ] Inspect at least top 10 for every query.
- [ ] Confirm that known good V2.5 literary/near-synonym candidates are not systematically lost.
- [ ] Confirm fewer antonyms, merely associated words, and definition-overlap accidents.
- [ ] Record latency and peak practical GPU usage.

## Phase E — Writer-relevance benchmark

Do not use the V3 9-way relation ontology as the main V4 target.

Human relevance labels:

- **3 — excellent**: direct synonym or highly useful literary/register alternative
- **2 — good**: near-synonym useful to a writer with a modest nuance shift
- **1 — contextual**: useful only in narrower contexts or mostly associated
- **0 — unsafe/irrelevant**: antonym, unrelated, misleading, or not meaning-preserving

Target benchmark:
- at least 30–50 diverse query senses
- nouns, verbs, adjectives, emotions, nature, speech, literary vocabulary
- preserve a frozen evaluation set

Metrics:
- NDCG@10
- MRR
- Recall@20 for relevance >= 2
- Unsafe@10 for relevance = 0

## Phase F — Model/control comparison

After Qwen pilot passes:

- [ ] Compare Qwen3-Reranker-0.6B against BGE-reranker-v2-m3.
- [ ] Compare instruction vs no-instruction for Qwen.
- [ ] Compare candidate pools 20 / 50 / 100.
- [ ] Keep the smallest/fastest configuration that preserves ranking quality.

## Decision gate

V4 may replace V2.5 ordering only if:

1. V2.5 candidate recall remains strong.
2. Human-rated NDCG@10 improves materially.
3. Unsafe@10 decreases.
4. Literary and rare useful alternatives are not collapsed into only obvious dictionary aliases.
5. Latency is practical for the intended product deployment.

If reranking improves precision but is too expensive for every production query, keep V2.5 as the default fast path and use V4 as an optional high-quality mode.

## Non-goals for the first V4 pilot

- No EmbeddingGemma fine-tuning.
- No teacher-generated hard negatives.
- No 9-way semantic relation classification.
- No full-dictionary reranking.
- No replacement of V2.5 before human evaluation.
