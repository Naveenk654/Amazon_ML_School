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
