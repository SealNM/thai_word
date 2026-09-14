# V3.1 — Local Teacher Router

Status: **in progress**

Branch: `feat/dictionary-semantic-v3-1-local-teacher-router`

Base: `feat/dictionary-semantic-v3-teacher-finetune`

V3.1 replaces the V3 Gemini-first labeling strategy while keeping the V3 compiler, fine-tuning, holdout, and evaluation design.

## Architecture

```text
V2.5 candidates
  -> high-confidence dictionary auto-label
  -> route at most 8 uncertain candidates/sense
  -> Qwen3-4B-Instruct-2507 local teacher
  -> Gemini only for low-confidence/uncertain + random audit
  -> V3 compiler
  -> EmbeddingGemma fine-tuning
```

## Implemented

- [x] `thai_v31_router.py`
  - Auto-label only standalone Tier 5 direct glosses.
  - Keep lower tiers for a teacher.
  - Reduce the remaining pool to at most 8 local candidates by default.
  - Preserve dense-nearest, strong lexical, and one lower-ranked candidate for negative diversity.

- [x] `scripts/route_v31_teacher_seeds.py`
  - Converts V3 teacher seeds into routed V3.1 tasks.

- [x] `scripts/generate_v31_local_teacher_labels.py`
  - Default model: `Qwen/Qwen3-4B-Instruct-2507`.
  - 4-bit NF4 with bitsandbytes by default.
  - Deterministic JSON generation.
  - Validates all relation labels.
  - Retries malformed JSON once.
  - Resumable JSONL output.

- [x] `scripts/build_v31_gemini_audit.py`
  - Sends every `uncertain` label to audit.
  - Sends local labels below confidence 0.75 to audit.
  - Adds a deterministic random 2% audit sample.

- [x] `scripts/merge_v31_teacher_labels.py`
  - Replaces audited local judgments with Gemini judgments.

- [x] `scripts/compare_v31_teachers.py`
  - Measures exact relation agreement.
  - Measures coarse positive/negative/graded agreement.
  - Reports relation confusion pairs.

- [x] `tests/test_v31_router.py`
  - Covers safe auto-labeling, routing, audit selection, and merge behavior.

## Outputs

```text
artifacts/v3/teacher_seeds.jsonl
        |
        v
artifacts/v3_1/routed_teacher_seeds.jsonl
        |
        v
artifacts/v3_1/local_teacher_labels.jsonl
        |
        +--> artifacts/v3_1/gemini_audit_seeds.jsonl
                    |
                    v
             Gemini audit
                    |
                    v
artifacts/v3_1/gemini_audit_labels.jsonl
        |
        v
artifacts/v3_1/final_teacher_labels.jsonl
```

## Pilot gates

Before scaling beyond the first 100 senses:

- [ ] Router reduces the 24-candidate source pool substantially.
- [ ] Local Qwen completes the pilot without systematic JSON failures.
- [ ] V3 compiler produces both useful positives and hard negatives.
- [ ] Manually inspect at least 30 anchors.
- [ ] Use a small Gemini audit or same-seed comparison to detect systematic label mistakes.
- [ ] Then scale to 1,000 senses and inspect again before a full run.

## Cost policy

- Dictionary rules: no API cost.
- Qwen local teacher: no API cost; uses Colab GPU.
- Gemini: only audit/fallback or a small teacher comparison set.
- EmbeddingGemma fine-tuning: local GPU.

This keeps Gemini calls to a small fraction of the original V3 plan.


## Pilot status

- [x] 100 V3 seeds routed successfully.
- [x] 2,400 source candidates reduced to 800 local judgments + 3 safe auto-labels.
- [x] First 10 Qwen local tasks completed: 10/10 success, 0 failures.
- [ ] Inspect relation quality/distribution for the 10-task smoke sample.
- [ ] Complete 100-task local pilot if quality is acceptable.
- [ ] Build Gemini audit subset and compare disagreement patterns.


## Throughput optimization after first 10-sense pilot

Observed:
- First 10 Qwen senses completed successfully with 0 failures.
- End-to-end runtime was more than 10 minutes, which is too slow to scale naively.

Changes:
- [x] Compact output schema uses candidate ids and short relation/register codes.
- [x] Removed generated reason text from the local teacher path because it is not used by the training compiler.
- [x] Reduced default output budget from 1400 to 320 tokens per sense.
- [x] Batch 4 senses per generation call by default.
- [x] Added model-load time, generation time, repair count, and senses/minute metrics.
- [x] Keep automatic single-row repair only for malformed compact JSON.

Scaling policy:
- Do **not** plan to label all 50,000+ senses by default.
- Use staged sampling: 100 -> 1,000 -> roughly 5,000-10,000 diverse anchors.
- Evaluate held-out retrieval quality after each dataset expansion.
- Continue adding teacher data only while holdout quality materially improves.
- Full-dictionary labeling is an optional later experiment, not a V3.1 requirement.


## Quality finding from first 10-label sample

The uploaded first-round local teacher output contained:
- 10 anchors
- 80 judgments
- unrelated: 38
- near_synonym: 15
- subtype: 10
- synonym: 5
- associated: 5
- supertype: 4
- antonym: 2
- uncertain: 1
- mean confidence ~0.899
- only 8/80 judgments below confidence 0.75

Manual inspection found important high-confidence ontology mistakes, including confusion between:
- near_synonym vs associated
- subtype/supertype direction
- subtype vs directly replaceable word
- unrelated vs potentially replaceable word

Therefore confidence-only auditing is insufficient, and 9-way local relation classification must not be used directly as the sole training-data source.

### Direct selector path

Added:
- [x] `scripts/select_v31_triplets_local.py`
  - asks Qwen only for 0-2 safe positives and 0-2 hard negatives per anchor
  - uses <=8 routed candidates
  - compact JSON
  - default batch size 8
  - default output budget 96 tokens
  - selection confidence must be >=80
- [x] `scripts/compile_v31_selector_triplets.py`
  - compiles selected positives/hard negatives directly
  - keeps frozen holdout checks

The 9-way relation classifier remains useful for audit/analysis, but the direct selector is now the preferred local path for contrastive training.


## Decision checkpoint — V2.5 remains the best baseline

Status: **V3.1 teacher path paused for redesign**

Based on the first real V3/V3.1 experiments, **V2.5 remains the strongest and most reliable version at this point**.

Why:
- V2.5 EmbeddingGemma retrieval already produced strong qualitative results for the actual Thai Words objective.
- The V3.1 local Qwen teacher path proved technically runnable, but relation quality was not reliable enough for safe contrastive supervision.
- High-confidence local labels could still contain ontology mistakes such as near-synonym vs associated, subtype/supertype direction, and overly aggressive hard-negative choices.
- The faster direct-selector experiment improved throughput substantially, but its first results still selected unsafe hard negatives for highly related forms.
- Therefore, continuing to scale the current teacher pipeline risks degrading a good V2.5 embedding space rather than improving it.

Current decision:
1. **Keep V2.5 as the primary baseline and preferred search architecture.**
2. Do not replace V2.5 with V3/V3.1.
3. Pause large-scale teacher generation and EmbeddingGemma fine-tuning with the current V3.1 labels.
4. Preserve all V3/V3.1 code and experiments as research material.
5. Any future V3.x direction must demonstrate that its training data is safer and more useful than the current teacher approach before fine-tuning is resumed.
6. Future work should consider alternative improvements that retain V2.5 strengths instead of assuming model fine-tuning is necessary.

This checkpoint supersedes the earlier assumption that the next mandatory step after V2.5 was large-scale teacher labeling + EmbeddingGemma fine-tuning.
