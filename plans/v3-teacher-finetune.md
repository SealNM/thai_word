# V3 — Teacher Relation Dataset + EmbeddingGemma Fine-tuning

Status: **in progress**
Branch: `feat/dictionary-semantic-v3-teacher-finetune`
Base: `feat/dictionary-semantic-v2-5-embeddinggemma`

## Goal

Adapt `google/embeddinggemma-300m` to Thai Words' actual objective: ranking Thai words that are useful to writers, especially synonyms and near-synonyms, while avoiding antonyms and merely associated words.

V3 must not train on the evaluation holdout.

## Principles

1. Keep V1 lexical relations and V2.5 hybrid ranking as the production-safe baseline.
2. Use an LLM teacher to **label candidate relations**, not to invent the entire lexicon from scratch.
3. Candidate generation comes from dictionary relations + EmbeddingGemma retrieval so the teacher judges grounded dictionary entries.
4. Separate:
   - synonym / near-synonym
   - antonym
   - subtype / supertype
   - manner
   - associated
   - unrelated
   - uncertain
5. Keep a frozen holdout and reject training examples whose anchor or candidate headword is in it.
6. Train EmbeddingGemma with hard negatives using CachedMultipleNegativesRankingLoss.
7. Preserve Matryoshka usefulness at 768 / 512 / 256 / 128 dimensions.
8. Compare V3 against untouched EmbeddingGemma 256d/768d and E5 on the same holdout.

## Phase A — Freeze holdout

- [x] Add `evaluation/v3_holdout_words.json`.
- [ ] Expand/annotate the final human-reviewed benchmark before declaring V3 production-ready.
- [x] Dataset scripts fail closed on direct holdout headword leakage.

## Phase B — Build grounded teacher seeds

Input:
- `artifacts/v1`
- a V2.5 EmbeddingGemma dense index
- holdout words

For each non-holdout dictionary sense:
1. Search the current hybrid system.
2. Keep a candidate pool with lexical/dense signals.
3. Remove self matches and holdout headwords.
4. Write JSONL teacher tasks with definitions and ranking evidence.

Output:
- `artifacts/v3/teacher_seeds.jsonl`

## Phase C — Teacher relation labeling

Default teacher path:
- Gemini structured output.
- Model configurable; never bake an API key into files.
- Low temperature / deterministic classification.
- Teacher must choose labels only for supplied dictionary candidates.
- Optional suggestions may be recorded separately but are **not** trusted as positives automatically.

Output:
- `artifacts/v3/teacher_labels.jsonl`

Each candidate label records:
- relation
- semantic_relatedness (0-4)
- replaceability (0-3)
- confidence (0-1)
- register
- short reason

## Phase D — Validate and compile training data

Validation:
- schema
- duplicate pairs
- anchor/candidate equality
- holdout leakage
- candidate belongs to the dictionary
- confidence bounds
- relation distribution

Training selection:
- positive: synonym / near_synonym with high confidence and useful replaceability
- hard negative: antonym; high-confidence associated/unrelated candidates that were retrieved near the anchor
- subtype/supertype/manner stay graded metadata by default and are not automatically pushed far away

Outputs:
- `artifacts/v3/training_triplets.jsonl`
- `artifacts/v3/graded_pairs.jsonl`
- `artifacts/v3/dataset_report.json`

## Phase E — Fine-tune EmbeddingGemma

Training:
- base: `google/embeddinggemma-300m`
- Sentence Transformers
- CachedMultipleNegativesRankingLoss
- BatchSamplers.NO_DUPLICATES
- query/document prompts from the model
- MatryoshkaLoss at 768, 512, 256, 128
- initial run: 1 epoch
- float32 by default; bf16 only when hardware supports it
- no fp16 assumption

The first V3 run is a conservative full fine-tune. If Colab T4 memory/runtime is not practical, add a PEFT experiment as a separate variant rather than silently changing the baseline.

## Phase F — Evaluate

Build a dense index using the trained model, then compare:
- V1 lexical
- E5-small
- E5-base
- EmbeddingGemma base 768d
- EmbeddingGemma base 256d
- V3 fine-tuned 768d
- V3 fine-tuned 256d

Success criteria:
1. Better synonym/near-synonym ordering on the frozen holdout.
2. Fewer antonyms in top results.
3. No broad collapse where only direct dictionary glosses are returned.
4. 256d remains competitive with 768d.
5. Improvements generalize across noun / verb / adjective / emotion / literary vocabulary.

## Non-goals for first V3

- No book/PDF ingestion yet.
- No multimodal ingestion yet.
- No reranker yet.
- No automatic acceptance of teacher-invented words.
- No production replacement of V2.5 until holdout evaluation passes.
