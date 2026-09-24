# Project Plan

## Stage 0 — Problem Understanding
- [ ] Read full PDF
- [ ] Understand ER formulation
- [ ] Understand outputs
- [ ] Understand F0.5
- [ ] Understand restrictions

## Stage 1 — Dataset Audit
- [ ] Source sizes
- [ ] Missingness
- [ ] Country distribution
- [ ] Name noise
- [ ] Address noise
- [ ] Ground-truth distribution
- [ ] Singleton rate
- [ ] Match multiplicity

## Stage 2 — Validation
- [ ] Design leakage-safe validation split
- [ ] Implement exact macro F0.5
- [ ] Establish baseline

## Stage 3 — Blocking
- [ ] Baseline blocking
- [ ] Candidate recall measurement
- [ ] Multi-block strategy
- [ ] Candidate reduction ratio

## Stage 4 — Matching
- [ ] Feature engineering
- [ ] Baseline classifier
- [ ] Threshold tuning
- [ ] Singleton handling

## Stage 5 — Error Analysis
- [ ] False positives
- [ ] False negatives
- [ ] Missed candidates
- [ ] Wrong multi-match decisions
- [ ] Singleton errors

## Stage 6 — Final Pipeline
- [ ] Train final system
- [ ] Generate test candidates
- [ ] Score candidates
- [ ] Generate outputs
- [ ] Run validator
- [ ] Reproduce from clean environment

## Stage 7 — Documentation
- [ ] Methodology
- [ ] Blocking strategy
- [ ] Model architecture
- [ ] Features
- [ ] Validation results
- [ ] Limitations
