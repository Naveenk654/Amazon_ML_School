# M3 — Cross-S1 competition features (new baseline)

Baseline: M4-NE-R = 0.95180 (frozen). Validation fold unchanged; no new blocking.
Code: `src/business_entity_resolution/stack.py` (full inference stack),
`experiments/scripts/score_stack.py`, `seed_scores.py --blocking CE --model M2E --prefix s2_`,
`comp_features.py`, `train_matcher.py --name M3 --blocking CE --extra S_ Q_ --cascade-prefix s2_ --min-p1 0.01`.

## Design
- **Competitor coverage:**
  - At test time every S1 is scored, so every competitor is visible.
  - The experiment therefore scores ALL 2,206,821 training S1 through the frozen stack:
    train (1.32M), tune (150k), val (200k) and "rest" (537k unsampled fold-3/4 S1,
    used only as competitors).
  - That is 47.5M scored pairs; 93.3% of pairs point at a target that other S1 also hold.
- **Stack check:** `stack.py` run on the validation fold reproduces M2-E exactly (max score
  difference 3e-8, 0 decision changes, macro F0.5 0.95180).
- **Competitor scores** are inference-time stage-2 (M2-E) scores:
  - train: 4-fold out-of-fold M2-E (same S1 folds as before);
  - tune/val/rest: the frozen M2-E. No labels are used.
- **Features** (per candidate (S1, t), over the OTHER S1 holding t): comp_n, comp_n_strong,
  comp_best, comp_margin = s2 − comp_best, comp_rank, comp_sum_other, cosine
  differences to the best competitor, comp_same_key (the best competitor has the same name key),
  s1_n_contested, plus the own s2.
- **Stage 3:** LightGBM (the same parameters) on M1-E features, sibling features and competition features.
  - Cascade on s2 ≥ 0.01; early stopping (91 rounds) and threshold (0.60) on tune.
  - No hard one-to-one assignment.

## Results (validation fold, 200,000 S1)
| | M4-NE-R | **M3** |
|---|---|---|
| **macro F0.5** | 0.95180 | **0.95871** |
| Δ (95% CI, paired bootstrap) | — | **+0.00691 [+0.00661, +0.00723]**; halves +0.00718 / +0.00664 |
| threshold (tune) | 0.65 | 0.60 |
| pair precision / recall | 0.9868 / 0.9102 | **0.9898 / 0.9237** |
| FP / FN | 8,439 / 62,082 | **6,568 / 52,792** |
| cross-S1-conflict FP (target owned by another S1) | 4,287 | **1,386** (3,629 removed, 728 new) |
| unowned-distractor FP | 4,152 | 5,182 |
| TP lost / gained | | 1,161 lost (1,089 had competitors) / **10,451 gained** |
| singleton F0.5 (false matches) | 0.9640 (405) | **0.9659 (384)** |
| empty predictions on non-singletons | 3,194 | **2,799** |
| India / US | 0.9319 / 0.9651 | **0.9425 / 0.9695** |
| common-name / rare-medium S1 | 0.9204 / 0.9617 | **0.9305 / 0.9676** |
| blocking recall / cands per S1 / ceiling | 0.9485 / 21.50 / 0.9784 | same |

Pair-level recall over all true pairs (precision):
| category | recall | precision |
|---|---|---|
| Indic target | 0.7728 → **0.7976** | 0.9566 → **0.9791** |
| Latin target | 0.9209 → 0.9335 | 0.9888 → 0.9905 |
| empty-address target | 0.4150 → **0.4752** | 0.9474 → 0.9499 |
| India | 0.8840 → 0.9005 | 0.9804 → 0.9871 |
| US | 0.9277 → 0.9391 | 0.9909 → 0.9915 |
| common-name S1 | 0.8549 → 0.8688 | 0.9827 → 0.9903 |

- Feature gain: comp_margin 84%, s2 9.5%, p1 2.5%.
- Competition fixes precision (one-to-one conflicts) *and* recall: a target with no strong
  rival can be accepted at a lower own score (threshold 0.60).

## Cost
| step | time | peak memory |
|---|---|---|
| rest scoring (537k S1 through the full stack) | 2,243 s | ~12 GB |
| out-of-fold M2-E | 501 s | 6.3 GB |
| competition features (47.5M pairs) | 143 s | 11.4 GB |
| M3 training | 242 s total (73 s fit) | 6.0 GB |

Stack inference throughput is about 60k S1 per 3.6 min at ~12 GB peak (batch 60k).
