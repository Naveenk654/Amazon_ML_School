# Test inference — frozen M3 stack (not submitted)

Command: `PYTHONPATH=src python -m business_entity_resolution.pipeline predict`
(phase A then phase B, each in its own process). Models: `models/`.

| | value |
|---|---|
| test S1 | 1,732,544 (all required; same order as test_source1.tsv) |
| scored candidate pairs | 36,997,673 (21.35 per S1; 0 S1 without candidates) |
| predicted pairs (M3 ≥ 0.60) | 5,854,785 |
| empty predictions | 112,620 (6.50%) |
| phase A | 6,288 s, 12.0 GB peak |
| phase B | 134 s, 11.0 GB peak |

By country (France is unseen in training and unlabeled; these are behaviour checks only):
| country | S1 | cands / S1 | pred / S1 | empty-prediction rate |
|---|---|---|---|---|
| US | 663,106 | 20.09 | 3.55 | 5.59% |
| India | 809,986 | 21.55 | 3.26 | 7.40% |
| France | 259,452 | 23.97 | 3.30 | 6.02% |

Reference: the train singleton rate is 5.58%. On validation, M3's empty-prediction
rate is 6.83%.

## Validation of the files
- Official `validate_submission.py --check-ids` with both files: **PASS**.
  matching_results.tsv: 1,732,544 rows (112,620 empty); candidate_pairs.tsv: 1,732,544
  rows (0 empty); 9,969,589 valid S2/S3 IDs loaded.
- Independent checks:
  - no duplicate S1 rows;
  - no duplicate IDs within a list;
  - 0 bad prefixes, 0 S1 self-matches, 0 IDs missing from test S2/S3;
  - matches ⊆ candidates;
  - UTF-8/ASCII, LF line endings.
- Target IDs predicted for more than one S1: 61 (Phase 3 baseline: 40,992).

| file | bytes | SHA-256 |
|---|---|---|
| matching_results.tsv | 97,905,110 | 424e0bb19bbf2a720ee30d0e286cc0898c395de942c26d07d2c97908268c7363 |
| candidate_pairs.tsv | 499,214,156 | ff5023e6c819d9bd8419c660c726e5108c77242a5a8b296d9e1719b332163cc4 |

The Phase 3 baseline files (SHA-256 628eff27…) are backed up outside the repo.
