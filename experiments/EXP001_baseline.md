# EXP001 — First end-to-end baseline

## Setup
- Code: `src/business_entity_resolution/` (`pipeline validate`, `pipeline predict`)
- Validation: train S1 split into 5 stratified folds (country × match-count
  bucket), grouped by S1 entity (an S1 and all its targets stay together).
  Folds 0–2 → matcher training (300k S1 sample), fold 3 → threshold tuning
  (150k), fold 4 → reported validation (200k). Blocking pool = **all** 10.3M
  train S2+S3 records, as in the test setting.
- Normalization: NFKC, Latin accent strip, lowercase, dotted-acronym collapse,
  apostrophe removal, punctuation → space; core name = minus legal/filler
  tokens; concat token; address abbreviation canon, leading-zero strip,
  NULL/N/A removal. No transliteration.
- Blocking (within country, data-driven labels):
  1. `name_exact`: sorted core-name key, blocks > 50 targets skipped
  2. `name_tfidf`: hashed word TF-IDF (core + concat token), top-10, df ≤ 20k
  3. `addr_tfidf`: hashed word TF-IDF of normalized address, top-10, df ≤ 20k
- Matcher: LightGBM, 300 rounds, 36 features (string similarities, number
  overlap, blocking cosines/ranks, name frequency, candidate-group context).
  No IDs, row order or country labels as features.
- Decision: score ≥ threshold (tuned on fold 3); empty list otherwise.

## Results (validation fold, 200,000 S1; 691,564 true pairs; 5.62% singletons)

| Blocking | recall | candidates |
|---|---|---|
| name_exact | 0.4649 | 954,939 |
| name_tfidf (top-10) | 0.5129 | 1,979,324 |
| addr_tfidf (top-10) | 0.7820 | 1,996,476 |
| **union** | **0.9036** | 3,913,182 (19.6 / S1, median 19, p95 28) |

- Reduction ratio: 0.999998 (vs all S1×target pairs), 0.999996 within country
- Candidate pair precision: 0.160
- Oracle macro F0.5 on candidates (perfect matcher): **0.9605**

| Matcher / decision | threshold | macro F0.5 | pair P | pair R |
|---|---|---|---|---|
| tuned | 0.65 | **0.9246** | 0.9820 | 0.8508 |
| fixed | 0.50 | 0.9225 | 0.9722 | 0.8636 |

Breakdown (tuned): US 0.9440, India 0.8957; singletons 0.9437 (633 of 11,247
got a false match); 1–2 matches 0.8902; 3–4 0.9298; 5+ 0.9411.

## Loss attribution (val)
- Blocking misses: 66,676 true pairs (9.6%); matcher FN: 36,509; FP: 10,775.
- Recovering all blocking misses alone (same matcher elsewhere) → 0.9685;
  perfect matcher on current candidates → 0.9605; removing all FPs → 0.9377.
- Blocking recall collapses on common names: S1 core name shared by >50
  targets = 24% of true pairs but 55% of blocking misses (recall 0.77–0.80 vs
  0.95 for names in 1–10 targets). Top-k slots are consumed by same-name
  records elsewhere; the exact pass skips those blocks.
- Other misses: Indic-script targets (21% of misses vs 7% overall), empty target
  address (16% vs 4%), typos in address words (word tokens do not match).
