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
