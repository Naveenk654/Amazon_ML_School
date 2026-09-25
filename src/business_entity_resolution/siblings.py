"""Second-stage "sibling" features (M2).

For each candidate (S1, target), compare the target with the S1's strongest
OTHER candidates, ranked by a stage-1 score p1. Only inference-time
information is used: stage-1 scores and target-to-target similarities. No
labels are used, and a candidate is never its own sibling.
"""
from __future__ import annotations

from multiprocessing import get_context

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .config import N_JOBS

TOP_M = 3        # siblings compared per candidate
STRONG = 0.5     # stage-1 score considered a strong sibling (fixed a priori)


def _sims(args):
    na, nb, aa, ab, ua, ub = args
    out = np.full((len(na), 3), np.nan, np.float32)
    for i in range(len(na)):
        out[i, 0] = fuzz.token_set_ratio(na[i], nb[i])
        if aa[i] and ab[i]:
            out[i, 1] = fuzz.token_set_ratio(aa[i], ab[i])
        if ua[i] and ub[i]:
            sa, sb = set(ua[i].split()), set(ub[i].split())
            out[i, 2] = len(sa & sb) / len(sa | sb)
    return out


def _pair_sims(tg: pd.DataFrame, ta: np.ndarray, tb: np.ndarray, pool) -> np.ndarray:
    step = 100_000
    cols = [tg[c].to_numpy() for c in ("name_core", "addr_norm", "addr_nums")]
    chunks = []
    for lo in range(0, len(ta), step):
        a, b = ta[lo:lo + step], tb[lo:lo + step]
        chunks.append(tuple(x for c in cols for x in (c[a].tolist(), c[b].tolist())))
    res = pool.map(_sims, chunks)
    return np.vstack(res) if res else np.zeros((0, 3), np.float32)


def sibling_features(cand: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """cand: s1_row, t_row, p1 (any order). Returns features aligned with cand."""
    n = len(cand)
    s1 = cand["s1_row"].to_numpy()
    p = cand["p1"].to_numpy().astype(np.float32)
    t = cand["t_row"].to_numpy()
    order = np.lexsort((-p, s1))                      # group by S1, best first
    s_s, p_s, t_s = s1[order], p[order], t[order]
    start = np.flatnonzero(np.r_[True, s_s[1:] != s_s[:-1]])
    size = np.diff(np.r_[start, n])
    g0 = np.repeat(start, size)
    gsz = np.repeat(size, size)
    pos = np.arange(n)
    rank = pos - g0

    # candidate slots = top TOP_M+1 of the group, excluding self
    slots = g0[:, None] + np.arange(TOP_M + 1)[None, :]
    valid = (np.arange(TOP_M + 1)[None, :] < gsz[:, None]) & (slots != pos[:, None])
    first = np.argsort(~valid, axis=1, kind="stable")[:, :TOP_M]
    sib = np.take_along_axis(slots, first, 1)
    sib_ok = np.take_along_axis(valid, first, 1)
    sib = np.where(sib_ok, sib, -1)

    F = {}
    ta = np.repeat(t_s, TOP_M)[sib.ravel() >= 0]
    tb = t_s[sib.ravel()[sib.ravel() >= 0]]
    with get_context("spawn").Pool(N_JOBS) as pool:
        sims = _pair_sims(tg, ta, tb, pool)
    S = np.full((n * TOP_M, 3), np.nan, np.float32)
    S[sib.ravel() >= 0] = sims
    S = S.reshape(n, TOP_M, 3)
    psib = np.where(sib >= 0, p_s[np.maximum(sib, 0)], np.nan)
    src = tg["src"].to_numpy() if "src" in tg else None
    for k in range(TOP_M):
        F[f"sib{k}_p"] = psib[:, k]
        F[f"sib{k}_name"] = S[:, k, 0]
        F[f"sib{k}_addr"] = S[:, k, 1]
        F[f"sib{k}_num"] = S[:, k, 2]
        if src is not None:
            same = np.where(sib[:, k] >= 0, src[t_s] == src[t_s[np.maximum(sib[:, k], 0)]], False)
            F[f"sib{k}_same_src"] = same.astype(np.float32)
    strong = psib >= STRONG
    combo = np.nanmean(S[:, :, :2], axis=2)           # mean of name & address sims
    for j, nm in enumerate(("name", "addr", "num")):
        F[f"sib_max_{nm}"] = np.nanmax(np.where(sib >= 0, S[:, :, j], np.nan), axis=1)
        F[f"sib_strong_max_{nm}"] = np.nanmax(np.where(strong, S[:, :, j], np.nan), axis=1)
    F["sib_strong_max_combo"] = np.nanmax(np.where(strong, combo, np.nan), axis=1)
    w = np.where(sib >= 0, np.nan_to_num(psib), 0)
    F["sib_wmean_combo"] = np.nansum(w * np.nan_to_num(combo), 1) / np.maximum(w.sum(1), 1e-6)

    # group-level stage-1 context (self excluded where it matters)
    gs = pd.Series(p_s).groupby(g0)
    gsum = gs.transform("sum").to_numpy()
    gstrong = pd.Series((p_s >= STRONG).astype(np.int32)).groupby(g0).transform("sum").to_numpy()
    F["p1"] = p_s
    F["p1_rank"] = rank.astype(np.float32)
    F["grp_max_p_other"] = np.where(rank == 0, np.where(gsz > 1, p_s[np.minimum(g0 + 1, n - 1)], 0), p_s[g0])
    F["grp_sum_p_other"] = gsum - p_s
    F["grp_n_strong_other"] = gstrong - (p_s >= STRONG)
    F["p1_gap_to_max_other"] = F["grp_max_p_other"] - p_s

    out = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in F.items()})
    inv = np.empty(n, np.int64)
    inv[order] = np.arange(n)
    return out.iloc[inv].reset_index(drop=True)
