# V2.6 — Lightweight bounded reranker

Status: **rejected after corrected 10-query pilot; V2.5 remains the baseline.**

## Baseline decision

V2.5 remains the retrieval baseline. The embedding experiments after V2.5 did not produce a consistently better ordering, so V2.6 does not change the embedding model, dense index, lexical search, or weighted-RRF fusion.

V2.6 tests one narrow hypothesis:

> Can we keep V2.5 semantic recall while moving unusually rare / structurally awkward dictionary forms slightly downward, without giving common words a positive relevance bonus?

## Why this differs from the failed commonness approaches

Earlier commonness fusion could accidentally lift a frequent but semantically weaker word. V2.6 uses frequency only as a negative rarity prior.

Guards:

1. no positive frequency boost;
2. rerank only inside the first semantic window (default top 20 of V2.5 top 50);
3. a candidate can improve by at most 4 positions;
4. a candidate can be displaced downward by at most 4 positions;
5. a weaker lexical-relation tier cannot jump over a stronger one;
6. protected V2.5 relation buckets cannot be crossed;
7. dense similarity must remain within a small tolerance (default 0.03) to pass a neighbor;
8. direct lexical relations cap the rarity penalty so rare but valid synonyms remain visible.

## Frequency source

Use \`pythainlp.corpus.tnc.unigram_word_freqs()\` from the Thai National Corpus (TNC), already available through the existing PyThaiNLP dependency.

For an exact headword missing from TNC, a multi-token form may use a heavily discounted token proxy. This prevents familiar phrases from receiving the same missing-frequency penalty as genuinely obscure forms, while avoiding a positive phrase-frequency bonus.

## Structural penalties

Initial structural diagnostics / penalties:

- bound forms such as \`วัส-\`;
- cross-reference-only definitions beginning with \`ดู...\`;
- one-character forms.

Do not add broad archaic/poetic-style penalties yet. Thai Words is for writers, so rare literary vocabulary can still be useful; the first experiment should only test bounded ordering, not remove stylistic vocabulary.

## Files

- \`thai_reranker_v26.py\` — bounded reranker and TNC frequency model
- \`scripts/evaluate_v26.py\` — V2.5 vs V2.6 side-by-side evaluator with diagnostics
- \`tests/test_reranker_v26.py\` — no-model unit tests for the ranking guards

## Evaluation

Use the same V2.5 dense artifact and existing 10-query evaluation set:

\`\`\`bash
python -u scripts/evaluate_v26.py \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --candidate-pool 50 \
  --rerank-window 20 \
  --max-promotion 4 \
  --max-demotion 4 \
  --dense-similarity-tolerance 0.03 \
  --top-k 10 \
  --device cuda \
  --output artifacts/v26/bounded-rarity.json
\`\`\`

Primary inspection cases:

- \`บ้าน\`: common usable forms should be able to pass unusually rare forms when semantic quality is comparable;
- \`เร็ว\`: \`ไว / รวดเร็ว / ด่วน\`-type candidates should not be blocked by rare dictionary forms;
- \`สวย\`: common direct substitutes should remain high;
- \`ฝน / เดิน / รัก / มืด\`: semantic validity must not regress merely because an associated word is frequent;
- bound forms such as \`วัส-\` should be demoted without hard-coding any query-specific vocabulary.

## Decision rule

Keep V2.6 only if it improves common-first ordering without damaging semantic validity across the shared benchmark. If it does not beat V2.5 consistently, keep V2.5 unchanged and use the diagnostics to decide whether the next step should be a learned pairwise/listwise ranker trained on product-specific preferences.


## First real pilot — invalidated by asymmetric bound bug

The first 10-query pilot exposed an implementation bug in the intended bounded reranking policy. Promotion was capped at +4, but a candidate could be displaced by many independent promotions and therefore fall without a bound. Observable examples included `จรรจา(-9)`, `ลำยอง(-6)`, and `สีฆ-(-6)`.

This means the first pilot cannot be used as the final quality verdict for V2.6.

Correction:
- add `max_demotion=4`;
- block a swap when it would push the displaced candidate below that bound;
- add a regression test where several common candidates try to cascade past one rare candidate;
- rerun the same real benchmark before tuning TNC thresholds or abandoning the approach.

The CPU fallback warning in the first pilot affects latency only; it does not explain the ranking issue.


## Final corrected pilot verdict

After fixing the asymmetric movement bug, the reranker respected the intended +/-4 movement bound.

Observed useful movements included:
- เดิน: ย่าง / ย่างเท้า improved relative to ย่างตีน;
- สวย: งดงาม moved upward strongly;
- บ้าน: หมู่บ้าน improved;
- มืด: มืดมน entered the top 10.

However, the improvement was inconsistent:
- TNC rarity did not reliably distinguish lexical substitutes from merely related words;
- some rare forms still moved upward;
- some useful direct alternatives moved down;
- important common-first failures such as บ้าน remained largely unresolved.

Decision:
- reject V2.6 as a replacement for V2.5;
- do not tune TNC thresholds further on the 10-query benchmark;
- keep the branch as research history.
