# M4-NE — Neighbour expansion through the frozen M2 stack (new baseline)

Frozen: C blocking, M1b, M2 (threshold 0.675, cascade p1 ≥ 0.01), validation fold.
Script: `experiments/scripts/m4_ne.py` (`--phase expand`, then `--phase score`).

## Exact change
1. Run C → M1b → M2 as before. Take M2's predicted targets for each S1
   (score ≥ 0.675; inference-time information only).
2. For each predicted target, retrieve its top-5 neighbours (excluding itself) from the existing
   hybrid name+address index, within country. Neighbours not already in C
   become new candidates for that S1 (they get no baseline-pass flags or ranks, only blocking cosines).
3. Rebuild matcher features for the enlarged set, then frozen M1b → sibling features
   → frozen M2 → the same threshold. No retraining, nothing tuned on validation.

Harness check: scoring C alone through the same code reproduces M2 exactly
(0.946030, FP 7,069, FN 79,445).

## Results (validation fold, 200,000 S1)
| | M2 | **M4-NE** |
|---|---|---|
| **macro F0.5** | 0.94603 | **0.95029** |
| Δ (95% CI, paired bootstrap) | — | **+0.00426 [+0.00407, +0.00447]**; halves +0.00412 / +0.00441 |
| pair precision / recall | 0.9886 / 0.8851 | **0.9895 / 0.9001** |
| false positives / false negatives | 7,069 / 79,445 | **6,634 / 69,116** |
| singleton F0.5 | 0.9654 | **0.9688** |
| India / US | 0.9241 / 0.9606 | **0.9287 / 0.9647** |
| common-name / rare-medium S1 | 0.9117 / 0.9568 | **0.9161 / 0.9611** |
| blocking recall | 0.9221 | **0.9485** |
| candidates per S1 (P95) | 19.84 (28) | 21.50 (31) |
| perfect-matcher ceiling | 0.9717 | **0.9784** |
| runtime (val, 200k S1) | — | expand 340 s (incl. 93 s index fit), score 133 s |
| peak memory | — | 9.7 GB (expand), 10.8 GB (score) |

- Expansion added 332,198 candidates with 18,261 true pairs. M2 predicted 15,095 of
  them (14,431 true, 664 FP, 95.6% precision).
- FP on the original candidates also fell (7,069 → 5,970), because the recomputed group
  and sibling context is stronger.

Pair-level recall over all true pairs (precision of predictions), M2 → M4-NE:
| category | recall | precision |
|---|---|---|
| Indic target | 0.6736 → 0.7132 | 0.9612 → 0.9691 |
| Latin target | 0.9016 → 0.9146 | 0.9902 → 0.9907 |
| empty-address target | 0.4050 → 0.4004 | 0.9618 → 0.9646 |
| India | 0.8517 → 0.8668 | 0.9828 → 0.9843 |
| US | 0.9074 → 0.9223 | 0.9923 → 0.9927 |
| common-name S1 | 0.8214 → 0.8341 | 0.9851 → 0.9877 |

Side effect: empty predictions on non-singletons rose 3,255 → 3,405. The group
features change when candidates are added, and the matcher was not trained on
expansion candidates.

## Error analysis after M4-NE (total loss 9,942)
| category | S1 | share of loss |
|---|---|---|
| correct but incomplete | 46,420 | 46.0% (missing pairs: 29,991 never blocked / 29,764 blocked but not predicted) |
| empty prediction, true pair among candidates | 1,741 | 17.5% |
| empty prediction, no true pair among candidates | 1,664 | 16.7% |
| extra FP only | 3,496 | 8.2% |
| mixed | 1,971 | 8.0% |
| singleton FP | 351 | 3.5% |

- Pair errors: blocking misses 35,585; matcher FN 33,531; FP 6,634.
- Remaining blocking misses: 28.9% empty-address targets, 23.5% Indic, 57% India;
  11.9% belong to S1 with no prediction, so expansion cannot reach them.
- Matcher FN: 23.7% empty-address targets; 45% score between 0.4 and 0.675.
