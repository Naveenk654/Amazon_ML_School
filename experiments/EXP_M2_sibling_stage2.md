# M2 — Second-stage sibling features (new baseline)

Frozen: C blocking, M1b model, validation fold (200,000 S1), no new candidates.
Code: `src/business_entity_resolution/siblings.py`, `experiments/scripts/stage1_scores.py`,
`experiments/scripts/sibling_shards.py`, `experiments/scripts/train_matcher.py --extra S_ --min-p1 0.01`.

## Design (fixed a priori, not tuned on validation)
- **Stage-1 score p1:**
  - train rows: 4-fold out-of-fold, folds by S1, M1b configuration, 131 rounds;
  - tune/val rows: the saved M1b model.
  - Leakage check: out-of-fold train log loss 0.0303 ≈ M1b val 0.0300
    (in-sample would be 0.0296).
- **Sibling features:** for each candidate, the top-3 OTHER candidates of the
  same S1 by p1. Their p1, the target-to-target name token-set,
  address token-set and number Jaccard, and same-source; the max over siblings and
  the max over strong siblings (p1 ≥ 0.5); a p1-weighted mean; and group
  context (max/sum/count of other p1, own rank, gap to max). No labels are used,
  and a candidate is never its own sibling.
- **Stage 2:** LightGBM (M1b parameters) on the M1 features + sibling features. It
  rescores candidates with p1 ≥ 0.01 (a cascade, for memory); below that p1 is
  kept (< threshold). The cascade excludes 537 val true pairs, and M1b missed
  all of them too. Early stopping and threshold on the tune fold.
- **Control M2-ctrl:** identical, but the `sib*` similarity columns are dropped.

## Results (validation fold)
| | M1b | M2-ctrl | **M2** |
|---|---|---|---|
| macro F0.5 | 0.93633 | 0.93950 | **0.94603** |
| Δ vs M1b (95% CI) | — | +0.00317 [+0.00290, +0.00345] | **+0.00970 [+0.00937, +0.01003]** |
| Δ M2 vs M2-ctrl | | | +0.00653 [+0.00622, +0.00681] |
| half-split Δ vs M1b | | | +0.00954 / +0.00986 |
| threshold (tune) | 0.65 | 0.65 | 0.675 |
| pair precision | 0.9835 | 0.9845 | **0.9886** |
| pair recall | 0.8670 | 0.8743 | **0.8851** |
| false positives | 10,072 | 9,537 | **7,069** |
| singleton F0.5 (false matches) | 0.9453 (615) | 0.9613 (435) | **0.9654 (389)** |
| India / US | 0.9100 / 0.9539 | 0.9145 / 0.9562 | **0.9241 / 0.9606** |
| common-name S1 / rare-medium S1 | 0.9000 / 0.9478 | 0.9031 / 0.9510 | **0.9117 / 0.9568** |
| blocking recall / cands per S1 / ceiling | 0.9221 / 19.84 / 0.9717 | same | same |
| stage-2 training time / peak memory | | 60 s / 4.9 GB | 98 s / 5.3 GB |

Extra pipeline cost: out-of-fold stage 1 took 601 s (11.9 GB); sibling
features took 372 s for 33.2M rows (6.0 GB).

**Recovery of M1b's matcher-missed true pairs (38,101):** M2 recovers 15,401
(40.4%) and loses 2,899 M1b true positives, a net +12,502 true positives.
FP: 4,569 removed, 1,566 new (10,072 → 7,069).

Recall / precision among blocked true pairs (M1b → M2):
| category | recall | precision |
|---|---|---|
| Indic target | 0.8772 → 0.9298 | 0.9356 → 0.9612 |
| Latin target | 0.9441 → 0.9617 | 0.9863 → 0.9902 |
| empty-address target | 0.6216 → 0.6182 | 0.9508 → 0.9618 |
| common-name S1 | 0.9377 → 0.9613 | 0.9769 → 0.9851 |

## Error analysis after M2 (total loss 10,794)
| category | share of loss |
|---|---|
| correct but incomplete | 49.8% |
| empty prediction on a non-singleton | 30.2% (3,255 S1; 1,658 have NO true pair among candidates) |
| mixed | 8.7% |
| extra FP only | 7.7% |
| singleton FP | 3.6% |

- Pair-level error sources: blocking misses **53,846** > matcher FN 25,599 > FP 7,069.
- Remaining matcher FN: 29.7% empty-address targets (recall 0.618, not
  improved by M2), 59% no shared house number, 44% India. Sibling similarity
  of FN is now close to that of negatives (median 65 vs 63): the signal is used up.
- FP: 46% are owned by another S1 (a one-to-one conflict).
- Decision rules: expected-F0.5 subset selection +0.0008–0.0009 on both halves
  (the first rule to show a consistent gain); top-1 rescue at 0.4: +0.0005.
