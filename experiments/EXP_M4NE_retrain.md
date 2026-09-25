# M4-NE-R — Retrain the stack on expanded candidates (new baseline)

Baseline: M4-NE = 0.95029 (frozen M1b/M2 scoring C + neighbour expansion).
Code: `src/business_entity_resolution/expansion.py`, `experiments/scripts/seed_scores.py`,
`build_expanded.py`, then the existing `train_matcher.py` / `stage1_scores.py` /
`sibling_shards.py` with `--blocking CE`.

## Pipeline (leak-free by construction)
- Stage A (unchanged, frozen): C → M1b → M2. Its predictions seed the expansion.
- **Seeds:**
  - train: 4-fold out-of-fold M2 (the same S1 folds as the out-of-fold M1b scores,
    M2 configuration, 149 rounds, cascade p1 ≥ 0.01);
  - tune/val: frozen M2.
  - The validation fold is never seen by any model.
- **Expansion:** the top-5 hybrid neighbours of each seed with score ≥ 0.675.
  - Train: +2,201,187 pairs (121,478 true) on 26.3M.
  - Per 200k S1 that is about 332k new pairs and 18.3k true, identical to val (332,198 / 18,261),
    which shows the out-of-fold seeds behave like inference seeds.
  - The val expansion reproduces M4-NE exactly.
- **Stage B (new):**
  - M1b-E trained on expanded train, early-stopped on expanded tune (126 rounds);
  - 4-fold out-of-fold M1b-E scores on train;
  - sibling features;
  - M2-E (cascade p1 ≥ 0.01, 166 rounds; threshold 0.65 tuned on tune).
- Caveat: a second-order stacking dependence (out-of-fold stage-2 inputs come from fold models that
  saw the other folds) affects only training-distribution realism, not validation.

## Results (validation fold, 200,000 S1)
| | M4-NE | **M4-NE-R (M2-E)** |
|---|---|---|
| **macro F0.5** | 0.95029 | **0.95180** |
| Δ (95% CI, paired bootstrap) | — | **+0.00151 [+0.00126, +0.00178]**; halves +0.00152 / +0.00150 |
| threshold (tune) | 0.675 | 0.65 |
| pair precision / recall | 0.9895 / 0.9001 | 0.9868 / **0.9102** |
| FP / FN | 6,634 / 69,116 | 8,439 / **62,082** |
| singleton F0.5 (false matches) | **0.9688 (351)** | 0.9640 (405) |
| India / US | 0.9287 / 0.9647 | **0.9319** / 0.9651 |
| common-name / rare-medium S1 | 0.9161 / 0.9611 | **0.9204** / 0.9617 |
| empty predictions on non-singletons | 3,405 | **3,194** |
| blocking recall / cands per S1 / ceiling | 0.9485 / 21.50 / 0.9784 | same |
| expansion true pairs predicted (of 18,261) | 14,431 (664 FP) | **16,220** (891 FP) |
| M1b-E alone (stage 1 only) | | 0.94075 (vs M1b 0.93633) |

Pair-level recall over all true pairs (precision), M4-NE → M2-E:
| category | recall | precision |
|---|---|---|
| Indic target | 0.7132 → **0.7728** | 0.9691 → 0.9566 |
| Latin target | 0.9146 → 0.9209 | 0.9907 → 0.9888 |
| empty-address target | 0.4004 → 0.4150 | 0.9646 → 0.9474 |
| India | 0.8668 → 0.8840 | 0.9843 → 0.9804 |
| US | 0.9223 → 0.9277 | 0.9927 → 0.9909 |
| common-name S1 | 0.8341 → 0.8549 | 0.9877 → 0.9827 |

**Trade-off:** the gain is recall (+7,034 TP), at the cost of +1,805 FP and a
singleton regression (−0.0048). The macro gain is significant and consistent; the
extra FP are the target for M3.

## Cost (full training pipeline)
| step | time | peak memory |
|---|---|---|
| seeds | 234 s | 5.9 GB |
| expansion (val / tune / train) | 435 s / 391 s / 2,282 s | 11.9 GB |
| M1b-E | 406 s | 10.0 GB |
| out-of-fold M1b-E | 611 s | **12.2 GB** |
| siblings | 376 s | 6.2 GB |
| M2-E | 270 s | 5.5 GB |
| **total** | ≈ 83 min | 12.2 GB (close to the ~14 GB limit) |
