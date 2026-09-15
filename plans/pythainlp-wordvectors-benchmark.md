# PyThaiNLP word-vector benchmark

Status: ready for pilot. Branches from V2.5 and does not alter the V2.5 baseline.

## Question

Can PyThaiNLP's Thai-specific static word vectors retrieve more useful Thai lexical alternatives than the current EmbeddingGemma dense component?

PyThaiNLP currently exposes:
- `thai2fit_wv`
- `ltw2v`
- `ltw2v_v1.0_15_window`
- `ltw2v_v1.0_5_window`

The first pilot uses `thai2fit_wv` only. It is small enough to test quickly before downloading the larger LTW2V models.

## Two retrieval modes

1. **Headword cosine**
   - exact dictionary headword vector vs candidate headword vectors;
   - measures what the original static embedding itself considers similar.

2. **Sense-definition mean**
   - build a mean word-vector representation of each `headword: definition` sense;
   - tokenize with PyThaiNLP `newmm`;
   - normalize the mean and rank senses by cosine;
   - deduplicate back to dictionary entries.

The second mode is the fairer comparison with V2.5 because EmbeddingGemma also embeds dictionary senses rather than only headwords.

## Guardrails

- Do not fuse word vectors into V2.5 yet.
- Do not tune against the 10-query benchmark.
- Report vocabulary coverage before judging quality.
- If both raw modes are clearly worse / mostly associative, stop immediately.
- Only test LTW2V if `thai2fit_wv` shows useful signal.

## Run

```bash
pip install -r requirements-wordvectors.txt

python -m unittest tests.test_pythainlp_wordvectors -v

python -u scripts/evaluate_pythainlp_wordvectors.py \
  --dense-index artifacts/v2/embeddinggemma-300m-256 \
  --model thai2fit_wv \
  --top-k 10 \
  --device cpu \
  --output artifacts/pythainlp-wordvectors/thai2fit.json
```

If thai2fit is promising, rerun with one of:

```bash
--model ltw2v
--model ltw2v_v1.0_15_window
--model ltw2v_v1.0_5_window
```

## Decision

This is a challenger benchmark only. V2.5 remains the validated baseline unless a PyThaiNLP vector mode shows clearly better lexical substitution behavior across the shared queries.
