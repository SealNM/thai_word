# V2.7 — Pairwise learned reranker distilled from an offline Gemma teacher

Status: implementation ready for pilot. Branch from V2.5, not from V2.6.

## Decision from V2.6

V2.6 proved two things:

1. a bounded rarity prior can improve a few common-first cases;
2. TNC rarity alone cannot reliably decide lexical replaceability.

Examples from the corrected V2.6 pilot:
- \`สวย\`: \`งดงาม\` improved;
- \`เดิน\`: \`ย่าง / ย่างเท้า\` improved;
- \`บ้าน\`: rare forms still dominated;
- \`พูด\`: frequency-driven movement could demote a useful lexical alternative;
- \`เร็ว\`: a rare form could still move upward.

Therefore V2.7 removes heuristic post-ranking from runtime. TNC remains only one learned feature.

## Hypothesis

Use V2.5 for high-recall candidate retrieval, then learn the product-specific ordering policy from offline Gemma rankings.

\`\`\`text
dictionary senses (holdout excluded)
        ↓
V2.5 top-18 candidates
        ↓
Gemma 4 offline teacher
        ↓
top-8 ordered candidate IDs
        ↓
pairwise preferences
        ↓
MaxAbsScaler + linear LogisticRegression
        ↓
tiny runtime utility scorer
        ↓
V2.5 top-30 → V2.7 → top-10
\`\`\`

Gemma is never needed in production search.

## Leakage prevention

\`evaluation/v27_holdout_words.json\` is frozen before teacher generation.

Rules:
- holdout words cannot be teacher anchors;
- holdout words cannot be teacher candidates;
- the original 10-query benchmark (\`ฝน ... บ้าน\`) is fully excluded from training;
- broader reserved evaluation vocabulary is excluded too.

If the training-data validator finds any leaked holdout word, training aborts.

## Runtime feature schema

The same \`thai_pairwise_v27.extract_features()\` function is used while mining training candidates and at runtime.

Features:
- V2.5 reciprocal rank and fusion score;
- protected relation tier / relation tier;
- direct vs weak lexical relation flags;
- lexical score and lexical reciprocal rank;
- dense similarity and dense reciprocal rank;
- lexical/dense availability flags;
- bound-form flag;
- dense-only sense-resolution flag;
- TNC log-frequency and missing flag;
- sense-count, word-length, definition-length log features.

No query-specific word list or hard-coded answer is used.

## Why pairwise linear ranking

For each teacher preference \`A > B\`, train on:
- \`features(A) - features(B)\` → positive;
- the exact inverse → negative.

Use \`MaxAbsScaler\` (linear through the origin) and logistic regression with \`fit_intercept=False\`.

This gives a valid candidate utility function at runtime:

\`\`\`text
utility(A) > utility(B)  <=>  pairwise model prefers A over B
\`\`\`

It is deterministic, tiny, inspectable, and fast.

## Safety guard

V2.7 preserves V2.5's strongest protected direct-relation bucket before applying learned utility. This prevents a model trained on commonness/style preferences from moving a weaker dense-only candidate above an explicit high-tier dictionary relation.

## Files

- \`thai_pairwise_v27.py\` — shared feature extraction + runtime ranker
- \`thai_v27_data.py\` — JSONL / holdout helpers
- \`scripts/build_v27_teacher_seeds.py\` — mine non-holdout V2.5 candidates
- \`scripts/label_v27_with_gemma.py\` — offline Gemma top-k teacher
- \`scripts/train_v27_pairwise.py\` — pair generation, anchor validation split, final fit
- \`scripts/evaluate_v27.py\` — frozen V2.5 vs V2.7 comparison
- \`tests/test_pairwise_v27.py\` — no-model runtime tests
- \`requirements-v27-teacher.txt\` — optional Gemma-only dependencies

## Pilot defaults

Keep the first pilot deliberately small enough to test the hypothesis before scaling:
- 96 sampled dictionary senses;
- 18 V2.5 candidates per anchor;
- Gemma chooses ordered top 8;
- selected-vs-selected pair preferences;
- 3 sampled unselected negatives per selected candidate;
- 20% anchor-level validation split;
- runtime reranks V2.5 top 30.

### 1. Build teacher seeds

\`\`\`bash
python -u scripts/build_v27_teacher_seeds.py \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --sample-size 96 \
  --candidate-count 18 \
  --device cuda \
  --output artifacts/v27/teacher_seeds.jsonl
\`\`\`

### 2. Inspect teacher prompt before loading Gemma

\`\`\`bash
python scripts/label_v27_with_gemma.py \
  --input artifacts/v27/teacher_seeds.jsonl \
  --top-k 8 \
  --dry-run
\`\`\`

### 3. Label with Gemma offline

\`\`\`bash
python -u scripts/label_v27_with_gemma.py \
  --input artifacts/v27/teacher_seeds.jsonl \
  --output artifacts/v27/teacher_labels.jsonl \
  --top-k 8 \
  --device cuda
\`\`\`

The labeler is resumable: existing \`seed_id\` rows are skipped.

### 4. Train linear pairwise ranker

\`\`\`bash
python -u scripts/train_v27_pairwise.py \
  --input artifacts/v27/teacher_labels.jsonl \
  --holdout evaluation/v27_holdout_words.json \
  --output artifacts/v27/pairwise_ranker.joblib \
  --report artifacts/v27/training_report.json
\`\`\`

Validation is split by anchor before pair construction, so pairs from one anchor cannot leak across train/validation.

### 5. Evaluate on the frozen original benchmark

\`\`\`bash
python -u scripts/evaluate_v27.py \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --model artifacts/v27/pairwise_ranker.joblib \
  --candidate-pool 30 \
  --top-k 10 \
  --device cuda \
  --output artifacts/v27/evaluation.json
\`\`\`

## Decision rule — stop if this does not clearly improve

This is the last ranking experiment in the current discussion.

Keep V2.7 only if the frozen holdout shows a clear overall improvement over V2.5 while preserving semantic validity. In particular:
- common direct substitutes should move up where V2.5 over-promotes rare dictionary forms;
- associated or wrong-role words must not become more prominent;
- improvements must appear across multiple held-out query types, not just one or two examples.

If V2.7 is mixed, neutral, or worse, stop here and keep V2.5 as the project baseline rather than tuning this benchmark further.
