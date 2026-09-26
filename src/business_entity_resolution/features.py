"""Pairwise features for (S1, candidate) pairs.

No entity IDs, row positions or country labels are used as features.
"""
from __future__ import annotations

from multiprocessing import get_context

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .blocking import PASSES
from .config import N_JOBS

EMB_COLS = ["emb_cos", "emb_rank", "emb_top1", "emb_gap", "emb_kth", "x_emb"]


def _string_feats(args):
    n1f, n2f, n1c, n2c, k1, k2, a1, a2, u1, u2 = args
    out = np.zeros((len(n1f), 13), np.float32)
    for i in range(len(n1f)):
        s1n = set(u1[i].split()) if u1[i] else set()
        s2n = set(u2[i].split()) if u2[i] else set()
        inter = len(s1n & s2n)
        union = len(s1n | s2n)
        out[i] = (
            fuzz.ratio(n1f[i], n2f[i]),
            fuzz.token_set_ratio(n1f[i], n2f[i]),
            fuzz.token_sort_ratio(n1c[i], n2c[i]),
            fuzz.partial_ratio(n1c[i], n2c[i]),
            JaroWinkler.similarity(k1[i], k2[i]),
            float(k1[i] == k2[i]),
            float(bool(k1[i]) and bool(k2[i]) and (k1[i] in k2[i] or k2[i] in k1[i])),
            fuzz.token_set_ratio(a1[i], a2[i]) if a2[i] else np.nan,
            fuzz.ratio(a1[i], a2[i]) if a2[i] else np.nan,
            inter / union if union else np.nan,
            inter,
            len(s2n - s1n),
            float(bool(s1n) and s1n == s2n),
        )
    return out


STRING_COLS = [
    "name_ratio", "name_tset", "core_tsort", "core_partial", "concat_jw", "concat_eq",
    "concat_contains", "addr_tset", "addr_ratio", "num_jacc", "num_shared",
    "num_extra_t", "num_equal",
]


def build_features(cand: pd.DataFrame, s1: pd.DataFrame, tg: pd.DataFrame,
                   s1_key_freq: pd.Series, t_key_freq: pd.Series) -> pd.DataFrame:
    """cand has s1_row/t_row positions into s1/tg plus blocking columns."""
    r1 = cand["s1_row"].to_numpy()
    r2 = cand["t_row"].to_numpy()
    fields = ["name_full", "name_core", "name_concat", "addr_norm", "addr_nums"]
    res = []
    step, big = 50_000, 1_000_000  # bounded memory: python strings built per slice
    # spawn: workers must not inherit (copy-on-write) the multi-GB parent heap
    with get_context("spawn").Pool(N_JOBS) as pool:
        for lo in range(0, len(cand), big):
            A, B = s1.iloc[r1[lo:lo + big]], tg.iloc[r2[lo:lo + big]]
            cols = []
            for f in fields:
                cols += [A[f].tolist(), B[f].tolist()]
            chunks = [tuple(c[i:i + step] for c in cols) for i in range(0, len(A), step)]
            res += pool.map(_string_feats, chunks)
            del cols, chunks, A, B
    F = pd.DataFrame(np.vstack(res) if res else np.zeros((0, len(STRING_COLS)), np.float32),
                     columns=STRING_COLS)
    del res
    a = s1[["country", "name_key", "name_core", "addr_nums"]].iloc[r1]
    b = tg[["src", "name_indic", "addr_empty", "name_core"]].iloc[r2]

    for p in PASSES:
        F[p] = cand[p].to_numpy().astype(np.int8)
        F[f"rank_{p}"] = cand[f"rank_{p}"].to_numpy()
    F["cos_name"] = cand["cos_name"].to_numpy()
    F["cos_addr"] = cand["cos_addr"].to_numpy()
    F["t_src3"] = (b["src"].to_numpy() == 3).astype(np.int8)
    F["t_indic"] = b["name_indic"].to_numpy().astype(np.int8)
    F["t_addr_empty"] = b["addr_empty"].to_numpy().astype(np.int8)
    F["s1_ntok"] = a["name_core"].str.count(" ").to_numpy() + 1
    F["t_ntok"] = b["name_core"].str.count(" ").to_numpy() + 1
    F["s1_nnums"] = a["addr_nums"].str.split().str.len().to_numpy()
    # How common the S1 name is (unsupervised; computed on the split's S1 set / target pool).
    k = pd.MultiIndex.from_arrays([a["country"].to_numpy(), a["name_key"].to_numpy()])
    F["s1_key_freq"] = s1_key_freq.reindex(k).to_numpy()
    F["t_key_freq"] = t_key_freq.reindex(k).fillna(0).to_numpy()

    # Candidate-set context for the same S1 (uses only blocking scores).
    g = pd.Series(cand["s1_row"].to_numpy())
    F["n_cands"] = g.map(g.value_counts()).to_numpy()
    for c in ("cos_name", "cos_addr"):
        s = pd.Series(F[c].to_numpy())
        F[f"{c}_grp_rank"] = s.groupby(g).rank(ascending=False, method="min").to_numpy()
        F[f"{c}_gap_to_max"] = (s.groupby(g).transform("max") - s).to_numpy()
    F["cos_sum"] = F["cos_name"] + F["cos_addr"]
    F["cos_sum_grp_rank"] = F["cos_sum"].groupby(g.to_numpy()).rank(ascending=False, method="min").to_numpy()
    for c in EMB_COLS:  # embedding-retrieval features, when the candidate frame carries them (M5-EMB)
        if c in cand.columns:
            F[c] = cand[c].to_numpy().astype(np.float32)
    return F.astype(np.float32)
