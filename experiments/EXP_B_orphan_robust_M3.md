# Experiment B — Orphan-robust M3 (M3OR)

Frozen: blocking, M1b, M2, expansion, M1b-E, M2-E. Only the stage-3 model changes.
Scripts: `experiments/scripts/comp_dropout.py` (QD_ features), `train_matcher.py --name M3OR
--blocking CE --extra S_ QD_ --cascade-prefix s2_ --min-p1 0.01`, `sim_orphans.py --m3-name M3OR`,
`eval_m3_heldout.py --role rest --phase B --m3-name M3OR`, `cmp_orphan_robust.py`.

## Design
- **Training competition features under randomized competitor dropout:**
  - training S1 are split into 3 random groups;
  - group g uses a population where 19% of the OTHER S1 are removed (seed g), so there are
    3 different removal draws;
  - tune (early stopping, threshold) uses its own 19%-dropout population (seed 3).
- Inputs: out-of-fold stage-2 scores, no labels. The validation fold is untouched (normal Q).
- Result: 91 rounds, threshold 0.625 (M3: 91 rounds, 0.60).
- Engineering: `competition_features` was rewritten in NumPy. It is bit-identical (max abs diff
  0.0 per column; M3 validation 0.958713 reproduced), and phase-B peak dropped 12.2 → 8.7 GB.

## Results
| | M3 | M3OR | Δ (paired bootstrap 95% CI) |
|---|---|---|---|
| **normal validation** macro F0.5 | **0.95871** | 0.95837 | −0.00034 [−0.00050, −0.00017] |
| **19% competitor-dropout validation** | 0.95646 | **0.95728** | **+0.00082 [+0.00062, +0.00101]** |
| 30% dropout | 0.95483 | 0.95653 | +0.0017 |
| degradation normal → 19% | −0.00225 | −0.00109 | halved |
| held-out set (533k S1, normal) | **0.95831** | 0.95797 | −0.0003 |

Breakdowns (normal → 19% dropout):
| | M3 normal | M3OR normal | M3 19% | M3OR 19% |
|---|---|---|---|---|
| precision / recall | 0.9898 / 0.9237 | **0.9912** / 0.9205 | 0.9861 / 0.9247 | **0.9893** / 0.9212 |
| FP / FN | 6,568 / 52,792 | **5,653** / 54,969 | 9,010 / 52,057 | **6,895** / 54,531 |
| singleton F0.5 | 0.9659 | **0.9705** | 0.9598 | **0.9668** |
| India / US | 0.9425 / 0.9695 | 0.9417 / 0.9695 | 0.9399 / 0.9675 | **0.9402 / 0.9686** |
| common-name S1 | 0.9305 | 0.9299 | 0.9282 | 0.9284 |
| competition-heavy S1 | 0.9498 | 0.9495 | 0.9473 | **0.9481** |
| empty-address target recall | 0.4752 | 0.4593 | 0.4917 | 0.4658 |

## Decision (pre-registered rule)
M3OR does NOT significantly improve normal validation (it is significantly worse by 0.0003).
It does reduce degradation under competitor dropout (+0.0008 at 19%). By the rule it is **not
adopted; M3 stays frozen**.

It is a precision-for-recall trade that pays off only when the population has test-like orphan
density. Test has measured +23% target records per S1, so the hidden leaderboard is the
appropriate arbiter.

## Candidate test file (not submitted, production unchanged)
Phase B on the stored test stage-1 outputs, with M3 replaced by M3OR (temporary models dir):
- official validator `--check-ids`: PASS;
- candidate_pairs.tsv is byte-identical to the M3 run;
- 5,797,294 predicted pairs (M3: 5,854,785); empty 6.64% (M3: 6.50%);
- matching_results.tsv: 97,166,667 bytes, SHA-256
  063210cd6c1972507ea2c1b4b1e1b1fde92a4ff0c2dcd3a4d4c92e74be9c8048.
