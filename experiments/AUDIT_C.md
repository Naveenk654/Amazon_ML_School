# Error-level audit of C (hybrid blocking, EXP001 matcher, threshold 0.65)

The validation fold (200,000 S1) was regenerated exactly: 3,968,530 candidates
and 637,718 true pairs among them, macro F0.5 0.93462.

## Where the lost F0.5 comes from (total loss 13,076 entity-points)
| entity category | S1 | share of loss |
|---|---|---|
| all predictions correct but incomplete (recall only) | 57,937 | 49.0% |
| empty prediction on a non-singleton | 3,193 | 24.4% (1,580 had a true pair among candidates) |
| mixed (some FP and some FN) | 3,770 | 10.9% |
| complete but with extra FPs | 5,917 | 10.4% |
| singleton falsely matched | 688 | 5.3% |

About 75–80% of the loss is recall; 15–20% is precision.

## Pair-level stages
- Blocking misses: 53,846 true pairs (7.8%). Profile (Phase 4B): 55% India,
  26% Indic-script target name, 20% empty target address, 44% common names.
- Matcher FN (true pair present, score < 0.65): 37,659 (5.9% of blocked true).
  - 58% have no shared address number, 20% have an empty target address,
    68% have a near-identical name (token-set ≥ 90).
  - Recall on Indic targets is 0.881 vs 0.945 for Latin; on empty-address
    targets it is 0.623.
  - **Sibling evidence:** 94% of these FN targets have other predicted-positive
    targets for the same S1. Their similarity to those siblings has median
    86.5 (55.7% ≥ 80), vs 62.5 (5.5% ≥ 80) for true negatives.
- FP: 11,714. 48% of FP targets are true matches of *another* S1 (a one-to-one
  conflict); 52% are unmatched distractors. 54% share a near-identical name,
  and 40% share all address numbers.

## Thresholding / decision rules (tuned on half A, evaluated on half B)
- Absolute threshold: flat optimum at 0.60–0.70 (Δ ≤ 0.0003).
- Forcing the top-1 when the prediction is empty: worse at every level.
- A relative-to-max rule: ≤ +0.0001.
- Expected-F0.5 subset selection: −0.0008.
- Scores are reasonably calibrated.
- **Conclusion: thresholding is not a lever.**

## Candidate yield
- 77% of candidates score < 0.01 and contain only 680 true pairs.
- Address top-k ranks 6–9 have a 3.5% true rate. Name ranks 6–9 have 4.7%.

## Country
India: macro F0.5 0.908 vs US 0.952. India is 40% of S1 but 56% of the loss.
