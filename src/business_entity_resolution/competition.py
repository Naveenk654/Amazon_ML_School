"""Cross-S1 competition features (M3).

For each candidate (S1, target), describe the OTHER S1 that also hold this
target as a candidate, using only stage-2 scores and blocking cosines
(inference-time information; no labels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRONG = 0.5


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
    g0, gsz = np.repeat(start, size), np.repeat(size, size)
    rank = np.arange(n) - g0                               # 0 = best S1 for this target
    comp = np.where(rank == 0, np.where(gsz > 1, g0 + 1, -1), g0)  # best OTHER S1 (sorted pos)
    has = comp >= 0
    ci = np.maximum(comp, 0)
    cs = np.where(has, s_s[ci], 0.0)
    gstrong = pd.Series((s_s >= STRONG).astype(np.int32)).groupby(g0).transform("sum").to_numpy()
    gsum = pd.Series(s_s).groupby(g0).transform("sum").to_numpy()
    s1o = s1_row[order]
    cn, ca = cos_name[order], cos_addr[order]
    F = pd.DataFrame({
        "s2": s_s,
        "comp_n": (gsz - 1).astype(np.float32),
        "comp_n_strong": (gstrong - (s_s >= STRONG)).astype(np.float32),
        "comp_best": cs.astype(np.float32),
        "comp_margin": (s_s - cs).astype(np.float32),
        "comp_rank": rank.astype(np.float32),
        "comp_sum_other": (gsum - s_s).astype(np.float32),
        "comp_dcos_name": np.where(has, cn - cn[ci], np.nan).astype(np.float32),
        "comp_dcos_addr": np.where(has, ca - ca[ci], np.nan).astype(np.float32),
        "comp_same_key": np.where(has, s1_key_id[s1o] == s1_key_id[s1o[ci]], False).astype(np.float32),
    })
    inv = np.empty(n, np.int64)
    inv[order] = np.arange(n)
    F = F.iloc[inv].reset_index(drop=True)
    contested = ((F["comp_best"] > F["s2"]) & (F["s2"] >= STRONG)).astype(np.float32)
    F["s1_n_contested"] = contested.groupby(s1_row).transform("sum").to_numpy()
    return F
