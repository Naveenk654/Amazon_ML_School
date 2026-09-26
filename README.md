# Amazon ML Challenge — Business Entity Resolution

This repository is for the Amazon ML Challenge Business Entity Resolution problem.

## Workflow

1. Read `problem_statement/amazon_ml_challenge_problem_statement.pdf`
2. Understand the problem before implementation.
3. Inspect the actual train/test data.
4. Build validation splits from training data.
5. Develop:
   - normalization
   - candidate generation / blocking
   - pairwise feature engineering
   - matching model
   - threshold calibration
   - singleton handling
6. Evaluate with the challenge's macro F0.5 metric.
7. Run inference on the full test set.
8. Produce:
   - `output/matching_results.tsv`
   - `output/candidate_pairs.tsv`
9. Validate the submission using the challenge validator.
10. Document the final methodology.

## Important

Do not use external business identity lookup, geocoding, government registries,
commercial entity-resolution APIs, or internet-based entity augmentation.
Use the provided data and permitted ML/NLP techniques only.

## Production inference (frozen M3 stack)

```bash
pip install -r requirements.txt            # + indic-transliteration only for the diagnostics
export BER_DATA=/path/to/dataset          # contains train/ and test/ TSVs (not committed)
export BER_WORK=/path/to/cache            # normalized parquet + inference staging
export PYTHONPATH=src

# production (M5-EMB) first needs the embedding neighbour lists in $BER_WORK/emb/
# (test_knn.parquet, test_s1_ids.parquet, test_t_ids.parquet) from experiments/modal/emb_full.py (GPU)
python -m business_entity_resolution.pipeline predict --batch 40000   # phase A + B -> output/*.tsv
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir $BER_DATA/test --check-ids
```

The frozen models are in `models/` (LightGBM text files + `meta.json` with
iterations, cascade cut-offs and thresholds; ~5 MB).

Pipeline (`src/business_entity_resolution/stack.py`), per test S1 batch (phase A):
1. `normalize`: country-agnostic name/address normalization.
2. `blocking` (C): exact core-name key, name TF-IDF top-10, address TF-IDF
   top-10, and hybrid name+address top-10 for common names, all within country.
3. **M1b** pairwise LightGBM → sibling features → **M2** second stage.
4. `expansion`: top-5 hybrid neighbours of each M2-predicted target (M4-NE).
5. Features over the enlarged set → **M1b-E** → siblings → **M2-E** (score s2).

Phase B (all S1 at once): `competition`: cross-S1 competition features (each
target vs the other S1 holding it) → **M3** → threshold 0.60 → submission.
`candidate_pairs.tsv` is exactly the pair set scored by the stack.

Validation (EXP001 fold, 200k S1): macro F0.5 0.95871 (C baseline 0.93462).
Experiment history: `experiments/`. Training scripts: `experiments/scripts/`.

## Baseline (EXP001, historical)

`python -m business_entity_resolution.pipeline validate` reproduces the first
baseline (0.9246).
