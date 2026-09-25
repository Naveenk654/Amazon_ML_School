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

## Running the baseline

```bash
pip install -r requirements.txt
export BER_DATA=/path/to/dataset      # contains train/ and test/ TSVs (not committed)
export BER_WORK=/path/to/cache        # normalized parquet, model, reports
export PYTHONPATH=src

python -m business_entity_resolution.pipeline validate   # S1-grouped labeled validation
python -m business_entity_resolution.pipeline predict    # writes output/matching_results.tsv
                                                          #        output/candidate_pairs.tsv
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir $BER_DATA/test
```

Pipeline: `normalize` → `blocking` (3 passes: exact core-name key, name TF-IDF
top-k, address TF-IDF top-k, all within country) → candidate union →
`features` → LightGBM matcher → threshold decision (empty list = no match) →
submission. `candidate_pairs.tsv` is exactly the set the matcher scores.
