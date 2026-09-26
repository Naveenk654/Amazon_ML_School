# Leaderboard log (hidden test; public subset)

| # | system | local val | held-out | leaderboard | matching_results.tsv SHA-256 |
|---|---|---|---|---|---|
| 1 | C baseline (Phase 3 EXP001 pipeline) | 0.9246 | — | 0.906 | 628eff27… |
| 2 | M3 (C + M2 siblings + expansion retrain + competition) | 0.95871 | 0.95831 | 0.922 | 424e0bb1… |
| 3 | M3OR (M3 with 19% competitor-dropout training) | 0.95837 | 0.95797 | 0.925 | 063210cd… |
| — | probe: M3OR with France predictions blanked (diagnostic only) | — | — | 0.797 | — |
| 4 | **M5-EMB** (M3OR stack + multilingual-e5-small top-5 candidates + embedding features) | 0.97263 | pending | **0.936** | 774d041f… |

Notes:
- M3OR was slightly worse on normal validation (−0.0003) but better under simulated
  test-like orphan density (+0.0008). On hidden test it gained +0.003. Competition-stage
  robustness matters more on test than locally (France, the densest-competition country, is
  the likely driver).
- France probe: US+India ≈ 0.927 and France ≈ 0.85 + (France singleton share), ≈ 0.91.
  So most of the local→leaderboard gap was US/India rather than France.
- M5-EMB: +0.0143 on validation and +0.011 on the leaderboard (0.925 → 0.936). Its local gain
  largely carried over to test.
- **Production = M5-EMB** (`models/`, `emb_k: 5`, M3 threshold 0.65). Inference needs
  `work/emb/test_knn.parquet` from `experiments/modal/emb_full.py`. M3OR backup: scratchpad `backup_M3OR/`.
