# M1 — Retrain the matcher on C's candidate distribution

Blocking is C (baseline passes + hybrid for common names), frozen. The same
EXP001 validation fold (200,000 S1) is used, and the threshold is tuned on the
tune fold (150,000 S1).
Scripts: `experiments/scripts/gen_matcher_data.py`, `experiments/scripts/train_matcher.py`.

| | C (EXP001 model) | M1a | **M1b (new baseline)** |
|---|---|---|---|
| training candidates | baseline blocking | C | C |
| training S1 | 300,000 | 300,000 (same) | **1,324,101** (all training folds) |
| rounds | 300 fixed | 300 fixed | early stopping on tune logloss → 131 |
| threshold (tuned on tune) | 0.65 | 0.65 | 0.65 |
| **macro F0.5** | 0.93462 | 0.93485 | **0.93633** |
| Δ vs C (paired bootstrap 95% CI) | — | +0.00023 [−0.00003, +0.00046] | **+0.00171 [+0.00146, +0.00195]** |
| pair precision | 0.9809 | 0.9817 | **0.9835** |
| pair recall | 0.8677 | 0.8666 | 0.8670 |
| false positives | 11,714 | 11,160 | **10,072** |
| singleton F0.5 | 0.9388 | 0.9398 | **0.9453** |
| singleton false matches | 688 | 677 | **615** |
| India / US | 0.9080 / 0.9524 | 0.9085 / 0.9524 | 0.9100 / 0.9539 |
| blocking recall / cands per S1 / ceiling | 0.9221 / 19.84 / 0.9717 | same | same |
| training time | — | 60 s | 229 s |
| peak memory (training) | — | 3.7 GB | 11.3 GB |

Data generation: all training-fold S1 took 1,350 s (11.4 GB peak); tune and
val took about 320 s each (~10.5 GB).

Conclusions:
- Retraining on C's distribution alone (M1a) is not significant.
- The gain comes from 4.4× more training S1, and it is almost all precision.
- The threshold optimum is unchanged (flat between 0.60 and 0.70).

## Error analysis after M1b (total loss 12,735)
| entity category | share of loss |
|---|---|
| correct but incomplete | 51.0% |
| empty prediction on a non-singleton | 25.2% (1,575 had a true pair among candidates) |
| mixed | 9.7% |
| extra FP only | 9.2% |
| singleton FP | 4.8% |

- Matcher FN: 38,101. 58% have no shared address number, 20% have an empty
  target address, and 68% have a near-identical name. Sibling similarity
  (median 86.4, 55.7% ≥ 80, vs 5.5% for negatives) is still unexploited.
- FP: 10,072; 43% are owned by another S1.
- Decision rules still give ≤ +0.0001: thresholding is not a lever.
