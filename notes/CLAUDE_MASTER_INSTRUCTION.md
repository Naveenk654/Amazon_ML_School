# Claude Master Instruction — Amazon ML Business Entity Resolution

Read the attached problem statement PDF completely before doing anything else.

## Phase 1 — Understand

Do NOT code or choose a final model immediately.

Explain:
- the business problem
- Source 1, Source 2, Source 3
- training and test data
- ground truth
- noisy names and addresses
- blocking/candidate generation
- matching
- candidate_pairs.tsv
- matching_results.tsv
- singleton entities
- F0.5
- macro averaging
- precision/recall trade-off
- open-set country requirement
- model/license constraints
- submission requirements
- prohibited external data lookup

Clearly distinguish:
1. facts explicitly stated by the PDF
2. things that must be measured from the actual dataset
3. hypotheses/possible approaches

Do not hallucinate dataset statistics.

## Phase 2 — Dataset Audit

After the actual dataset is available, measure:
- row counts for each source
- column structure
- missing values
- country distribution
- name/address length distributions
- duplicate and near-duplicate patterns
- positive-match counts
- matches per Source 1 entity
- singleton rate
- source-pair distribution
- noise patterns
- possible train/validation split strategies

## Phase 3 — Baseline

Build a reproducible baseline:
- normalization
- simple blocking
- interpretable similarity features
- baseline matcher
- threshold selection
- exact challenge-style F0.5 evaluation

## Phase 4 — Improve

Iteratively investigate:
- stronger normalization
- multi-strategy blocking
- candidate recall
- character/token similarities
- TF-IDF retrieval
- pairwise ML models
- threshold calibration
- singleton handling
- error analysis

Every improvement must be validated experimentally.

## Rules

Do not:
- fabricate results
- assume a model is best without validation
- use external business lookup
- use geocoding APIs
- use government business registries
- use commercial entity-resolution services
- add internet-derived business data

The goal is a strong, reproducible, competition-quality solution.
