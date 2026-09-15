# PyThaiNLP word-vector benchmark

Status: rejected after thai2fit_wv pilot. V2.5 remains the baseline; LTW2V follow-up is not justified.

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


## Pilot result — thai2fit_wv rejected

Observed coverage:
- headword coverage: 40.0%
- sense-definition mean coverage: 100.0%
- vector size: 300d

The headword mode produced some useful neighbors, for example:
- โกรธ → โมโห
- สวย → งดงาม
- พูด → พูดจา / กล่าว
- บ้าน → หมู่บ้าน / บ้านพัก
- เร็ว → รวดเร็ว
- กลัว → เกรง / หวาดกลัว / เกรงกลัว

However, the dominant behavior is distributional association rather than lexical substitution:
- ฝน → พายุ / หิมะ / อุทกภัย / มรสุม
- เดิน → เดินทาง / วิ่ง / แล่น / ปีน
- มืด → สว่าง / สดใส
- เร็ว → ช้า / ล่าช้า
- กลัว → เกลียด / โกรธ

This is a fundamental mismatch for Thai Words: antonyms, co-occurring concepts, related actions, and topic-neighbors can be close in static Word2Vec space even when they cannot replace the query in a sentence.

The sense-definition mean mode is substantially worse. Although nominal coverage reaches 100%, averaging static vectors over dictionary-definition tokens collapses meaning toward generic definition/context vocabulary and returns largely unrelated entries. Coverage therefore must not be interpreted as semantic quality.

### Decision

- Do not integrate thai2fit_wv into V2.5.
- Do not tune the mean-vector method on the current 10-query set.
- Do not continue to LTW2V in this experiment: a larger static Word2Vec vocabulary may improve OOV coverage but does not address the core association-vs-substitution failure demonstrated by thai2fit_wv.
- Keep V2.5 EmbeddingGemma + lexical PyThaiNLP hybrid as the strongest validated baseline.
