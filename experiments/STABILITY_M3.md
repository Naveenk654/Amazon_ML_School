# M3 stability check and production-path verification

Both checks run through the PRODUCTION code path
(`stack.score_batch(..., stage3=True)` + `stack.stage3_scores`, via
`experiments/scripts/eval_m3_heldout.py`). Models: `models/` (frozen M3 stack).

## 1. Reproduction on the validation fold (200,000 S1)
The production path reproduces the M3 experiment exactly: macro F0.5 0.958713,
FP 6,568, FN 52,792, precision 0.98982, recall 0.92366.

## 2. Independent held-out set (second validation sample)
532,720 training S1 from folds 3–4 that were never used for training, early stopping
or threshold selection (disjoint from the validation fold).

| | validation fold (200k) | held-out set (533k) |
|---|---|---|
| true pairs | 691,564 | 1,843,829 |
| blocking recall | 0.9485 | 0.9483 |
| M2-E level macro F0.5 | 0.95180 | 0.95117 |
| **M3 macro F0.5** | **0.95871** | **0.95831** |
| M3 − M2-E (95% CI) | +0.00691 [+0.00661, +0.00723] | +0.00714 [+0.00694, +0.00734] |
| M3 precision / recall | 0.9898 / 0.9237 | 0.9899 / 0.9234 |
| M3 singleton F0.5 | 0.9659 | 0.9645 |
| M3 India / US | 0.9425 / 0.9695 | 0.9420 / 0.9692 |

Conclusion: M3 is stable. The held-out score is within 0.0004 of validation, and
the gain over M2-E is the same on both samples.

Cost (held-out set, 533k S1): phase A 2,094 s (12.0 GB peak), phase B 216 s (12.2 GB).
