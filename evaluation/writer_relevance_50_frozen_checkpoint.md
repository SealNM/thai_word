# Writer relevance benchmark — frozen checkpoint

Status: approved after human review.

## Dataset

- 50 target senses
- 1,500 labeled candidate pairs
- schema v3
- V2.5 retrieval baseline

## Split

- train: 31 targets / 930 pairs
- validation: 9 targets / 270 pairs
- benchmark: 10 targets / 300 pairs

The original 10 pilot targets stay in train/development because they were used while designing the schema and objective. Candidate pairs are never randomly split across target senses.

## V2.5 baseline

All 50 targets:
- Useful@10: 0.940
- HighUtility@10: 0.900
- Noise@10: 0.060
- SevereError@10: 0.060
- NDCG@10: 0.965084
- MRR high utility: 0.990

Validation:
- Useful@10: 0.9444
- HighUtility@10: 0.9333
- Noise/SevereError: 0.0556
- NDCG@10: 0.955363

Frozen benchmark:
- Useful@10: 0.940
- HighUtility@10: 0.870
- Noise/SevereError: 0.060
- NDCG@10: 0.955487
- MRR high utility: 1.000

## Files

- evaluation/writer_relevance_50_split_manifest.json
- evaluation/writer_relevance_50_metrics_approved.json

The split is frozen before Phase 3 experiments and should not be changed in response to model results.
