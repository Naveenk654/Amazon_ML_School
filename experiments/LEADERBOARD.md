# Leaderboard log (hidden test; public subset)

| # | system | local val | held-out | leaderboard | matching_results.tsv SHA-256 |
|---|---|---|---|---|---|
| 1 | C baseline (Phase 3 EXP001 pipeline) | 0.9246 | — | 0.906 | 628eff27… |
| 2 | M3 (C + M2 siblings + expansion retrain + competition) | 0.95871 | 0.95831 | 0.922 | 424e0bb1… |
| 3 | **M3OR** (M3 with 19% competitor-dropout training) | 0.95837 | 0.95797 | **0.925** | 063210cd… |

Notes:
- M3OR was slightly worse on normal validation (−0.0003) but better under simulated
  test-like orphan density (+0.0008). On hidden test it gained +0.003. Competition-stage
  robustness matters more on test than locally (France, the densest-competition country, is
  the likely driver).
- **Production = M3OR** (`models/M3.txt`, threshold 0.625 in `models/meta.json`).
