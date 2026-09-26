# M5-EMB: embedding retrieval (multilingual-e5-small) as extra candidates + features

Baseline: production M3OR (normal validation 0.95837; leaderboard 0.925).
Single change: the embedding neighbour lists from `experiments/modal/emb_full.py`, as
  - candidates: each S1's top-5 neighbours, added after neighbour expansion;
  - features: `emb_cos`, `emb_rank` (NaN when the pair is not in the S1's top-30),
    `emb_top1`, `emb_kth` (30th cosine), `emb_gap`, and `x_emb` (pair added by embeddings).

Stage A (C → M1b → M2 seeds) is frozen and unchanged. Stage B (M1bE, OOF, siblings, M2E)
and M3 (19% competitor dropout, as M3OR) are retrained with identical settings. The
competitor population ("rest" 532k S1) is rescored with the new stack.
Runner: `experiments/scripts/run_m5emb.sh` (`BER_VARIANT=X`). Code: `src/business_entity_resolution/emb.py`.

## Embedding run
- Model: intfloat/multilingual-e5-small (MIT, 118M). Text: `query: name | address`, fp16, max 64 tokens.
- Modal T4, ~6k texts/s. For every train and test S1: top-30 same-country S2/S3 records.
- The train output reproduces the pilot exactly (100% of the pilot's top-30 pairs).
- The expansion code was refactored for reuse; its output is verified identical
  (val: 332,198 new pairs / 18,261 true, as before).

## Candidates
| role | added by embeddings | of which true |
|---|---|---|
| val (200k S1) | 215,083 (1.08 / S1) | 13,799 |
| tune (150k S1) | 161,154 | 10,460 |
| train (1.32M S1) | 1,422,304 | 91,077 |

Validation: blocking recall 0.9485 → **0.9685**; candidates per S1 21.50 → 22.58; best
achievable macro F0.5 0.9784 → **0.9900**.

## Results (validation fold, 200,000 S1, never seen by any model)
| stage | before | **M5-EMB** |
|---|---|---|
| stage 1 (M1bE) | 0.94075 | **0.95506** |
| stage 2 (M2E) | 0.95180 | **0.96608** |
| **final (M3)** | 0.95837 (M3OR) | **0.97263** |
| Δ final, paired bootstrap 95% CI | — | **+0.01426 [+0.01378, +0.01469]**; halves +0.01447 / +0.01405 |
| S1 better / worse | — | 15,882 / 3,780 |
| threshold (tune) | 0.625 | 0.65 |
| pair precision / recall | 0.9912 / 0.9205 | **0.9928 / 0.9413** |
| FP | 5,653 | **4,699** |
| singleton F0.5 | 0.9705 | **0.9748** |
| India / US | 0.9417 / 0.9695 | **0.9626 / 0.9793** |

`emb_cos` is the single most important stage-1 feature (26.6% of gain). The gain is not
only recall from new candidates: precision also rises and FP fall by 17%.

## Memory fixes (results unaffected)
- `train_matcher.py`: the raw float32 matrix is dropped once LightGBM has binned it.
  Peak memory is now 9.7 GB; before the fix the run was killed near the ~14 GB limit.
- `stage1_scores.py`: shards are read per fold instead of all at once. Peak 12.2 → 7.6 GB.
- The rest population is scored in 60k-S1 batches (the test batch size) instead of 150k.

## Held-out ("rest", 532,720 S1, never trained on)
| | M3OR | **M5-EMB** |
|---|---|---|
| macro F0.5 | 0.95797 | **0.97174** |
| India / US | 0.9414 / 0.9691 | **0.9618 / 0.9784** |
| blocking recall | — | 0.9683 |

## Leaderboard
**0.936** (M3OR 0.925), so +0.011 of the +0.014 local gain carried over. Promoted to production.
