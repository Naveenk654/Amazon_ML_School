# Competition mode, Phase 1 — train / held-out → test distribution audit (frozen M3)

Scripts: `experiments/scripts/audit_shift.py` (per-pair dump + summary),
`experiments/scripts/sim_orphans.py`. Everything uses the frozen production M3 path;
the val dump reproduces M3 exactly (0.95871). No model was changed.

Scores: local val 0.95871 · held-out 0.95831 · **hidden leaderboard 0.922** (C: 0.906).

## Key statistics (validation fold → test, by country)
| metric | val India | val US | test India | test US | test France |
|---|---|---|---|---|---|
| S1 | 79,945 | 120,055 | 809,986 | 663,106 | 259,452 |
| target records per S1 (whole split) | 4.68 | 4.67 | **5.82** | **5.76** | 5.53 |
| candidates per S1 | 21.90 | 21.24 | 21.55 | 20.09 | 23.97 |
| stage-2 cascade rate (s2 ≥ 0.01) | 0.196 | 0.198 | 0.217 | **0.243** | 0.224 |
| expansion share of cascade rows | 0.072 | 0.057 | 0.072 | 0.044 | **0.160** |
| candidates with s2 in 0.3–0.7 | 0.0090 | 0.0075 | 0.0122 | **0.0141** | 0.0144 |
| cascade rows with M3 in 0.4–0.8 | 0.030 | 0.024 | 0.044 | **0.050** | 0.035 |
| strong candidates per S1 (s2 ≥ 0.5) | 3.19 | 3.29 | 3.26 | 3.51 | 3.43 |
| predictions per S1 | 3.16 | 3.27 | 3.26 | **3.55** | 3.30 |
| empty-prediction rate | 7.5% | 6.4% | 7.4% | 5.6% | 6.0% |
| predictions with a strong competing S1 | 1.0% | 0.3% | 0.8% | 0.3% | **3.2%** |
| predictions whose best competitor has the same name key | 16.7% | 16.3% | 16.8% | 13.7% | **25.5%** |
| M3 margin, 10th percentile (predictions) | 0.88 | 0.97 | 0.84 | 0.90 | **0.75** |
| targets predicted for > 1 S1 | 0 | 0 | 6 | 2 | **53** |
| competitors per cascade row, 90th percentile | 14 | 9 | 10 | 7 | **21** |
| Indic share of predicted targets | 16.1% | 0 | 15.7% | 0 | 0 |
| common-name S1 share | 33% | 18% | 34% | 16% | 18% |

The held-out set (533k S1) matches val on every statistic (to within about ±0.001).

## Orphan-target simulation (validation fold)
Test has +23% target records per S1 in both US and India. That is consistent with about 19%
of the S1 owning test targets being absent from test S1, leaving ownerless "orphan"
targets. Simulation: remove competitor S1 (never the evaluated S1), then recompute the competition
features and M3 only.
| removed competitor S1 | M3 macro F0.5 | FP | FP on orphaned targets |
|---|---|---|---|
| 0% | 0.95871 | 6,568 | 0 |
| 19% | 0.95646 (−0.0023) | 9,010 | 2,436 |
| 30% | 0.95483 (−0.0039) | 10,740 | 4,167 |
M2-E level (no competition features) is unaffected: 0.95180.

## Top 5 shifts (ranked by plausible contribution to the 0.958 → 0.922 gap)
1. **France is unseen and anomalous (15% of test S1).** It has 2.5× the expansion share,
   5–10× the share of predictions with a strong competitor, 25% same-key competitors,
   the lowest margins, 53 of the 61 multiply-assigned targets, and 2× the competitor tail.
   Arithmetic: if US and India score as on val at test mix (0.9547) minus the orphan effect,
   France ≈ **0.75**. This is the largest candidate by far.
2. **Country mix.** India is 55% of US+India S1 in test vs 40% in val, and India scores lower
   (0.942 vs 0.970). Expected cost ≈ **−0.004** with no model error at all.
3. **Harder US/India candidate sets at stage 2.** Ambiguous s2 mass (0.3–0.7) is up
   1.4–1.9×, the cascade rate is up 10–23%, and p1 quantiles are lower. This does not depend on
   competition, so it is a change in the targets themselves (more near-duplicate
   records per S1). Its cost can't be measured without labels.
4. **Orphan-target density (+23% records per S1).** Measured cost −0.0023 at 19%
   orphaning; it inflates FPs specifically in M3's "uncontested target" logic.
5. **More predictions per S1 in test US (+8.5%) with fewer empty predictions (5.6% vs
   6.4%).** These are either more true matches or more FPs. Combined with #3/#4 the
   precision side is the risk (F0.5 weights precision 2×).

Items 2 and 4 are measured at about −0.006 combined. That leaves about 0.03 of the gap to France (#1) and
the unmeasurable US/India hardness (#3).
