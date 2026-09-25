"""Cross-S1 competition features (M3).

For each candidate (S1, target), describe the OTHER S1 that also hold this
target as a candidate, using only stage-2 scores and blocking cosines
(inference-time information; no labels). Pure NumPy (segment sums over a
target-sorted order) so tens of millions of pairs fit in memory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRONG = 0.5
COLUMNS = ["s2", "comp_n", "comp_n_strong", "comp_best", "comp_margin", "comp_rank", "comp_sum_other",
           "comp_dcos_name", "comp_dcos_addr", "comp_same_key", "s1_n_contested"]


def competition_features(s1_row: np.ndarray, t_row: np.ndarray, s2: np.ndarray,
                         cos_name: np.ndarray, cos_addr: np.ndarray,
                         s1_key_id: np.ndarray) -> pd.DataFrame:
    """Arrays are aligned over the whole scored population; returns features aligned too.

    s1_key_id: integer id of each S1's normalized name key (indexed by s1_row).
    """
    n = len(t_row)
    s = np.asarray(s2, np.float32)
    order = np.lexsort((-s, t_row))
    t_s, s_s = t_row[order], s[order]
    start = np.flatnonzero(np.r_[True, t_s[1:] != t_s[:-1]])
    size = np.diff(np.r_[start, n])
    del t_s
    g0 = np.repeat(start, size)
    gsz = np.repeat(size, size)
    rank = np.arange(n) - g0                                   # 0 = best S1 for this target
    comp = np.where(rank == 0, np.where(gsz > 1, g0 + 1, -1), g0)  # best OTHER S1 (sorted pos)
    has = comp >= 0
    ci = np.maximum(comp, 0)
    del comp
    strong = s_s >= STRONG
    gstrong = np.repeat(np.add.reduceat(strong.astype(np.int32), start), size)
    gsum = np.repeat(np.add.reduceat(s_s.astype(np.float64), start), size).astype(np.float32)
    cs = np.where(has, s_s[ci], 0.0).astype(np.float32)
    s1o = s1_row[order]
    out = np.empty((n, len(COLUMNS)), np.float32)

    def put(j, v):                                             # scatter sorted values back
        out[order, j] = v

    put(0, s_s)
    put(1, gsz - 1)
    put(2, gstrong - strong)
    put(3, cs)
    put(4, s_s - cs)
    put(5, rank)
    put(6, gsum - s_s)
    cn = cos_name[order]
    put(7, np.where(has, cn - cn[ci], np.nan))
    del cn
    ca = cos_addr[order]
    put(8, np.where(has, ca - ca[ci], np.nan))
    del ca
    put(9, np.where(has, s1_key_id[s1o] == s1_key_id[s1o[ci]], False))
    del g0, gsz, rank, ci, has, strong, gstrong, gsum, cs, s1o
    contested = ((out[:, 3] > out[:, 0]) & (out[:, 0] >= STRONG)).astype(np.float64)
    per_s1 = np.bincount(s1_row, weights=contested)
    out[:, 10] = per_s1[s1_row]
    return pd.DataFrame(out, columns=COLUMNS)
