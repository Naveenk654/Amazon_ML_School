"""Neighbour expansion (M4-NE).

Records that are near-duplicates of a target already predicted for an S1 are
likely the same business, because S2/S3 hold several records per entity.
For each seed (S1, predicted target) pair, the target's top-k neighbours in
the hybrid name+address index become candidates for that S1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .blocking import EXTRA_PASSES, PASSES, Blocker, addr_doc, name_doc, pair_cosine

KEY = lambda s, t: np.asarray(s, np.int64) << 32 | np.asarray(t, np.int64)


def expand(blocker: Blocker, s1: pd.DataFrame, tg: pd.DataFrame, seeds: pd.DataFrame,
           exclude: np.ndarray, k: int = 5) -> pd.DataFrame:
    """seeds: (s1_row, t_row) predicted pairs. exclude: sorted int64 keys already candidates.

    Returns new candidate rows in the blocking-output schema (no baseline-pass
    flags or ranks), with both blocking cosines computed.
    """
    out = []
    for c, m in blocker.by_country.items():
        sub = seeds[tg["country"].to_numpy()[seeds["t_row"].to_numpy()] == c]
        if sub.empty:
            continue
        q = tg.iloc[sub["t_row"].to_numpy()]
        Qh = sp.hstack([m["name"].transform(name_doc(q)), m["addr"].transform(addr_doc(q))]).tocsr()
        fr = Blocker._topk_frame(m["hybrid_TT"], Qh, k + 1, blocker.cfg, 0)  # +1: the seed itself
        new = pd.DataFrame({"s1_row": sub["s1_row"].to_numpy()[fr["qi"].to_numpy()],
                            "t_row": m["idx"][fr["tj"].to_numpy()]}).drop_duplicates()
        new = new[~np.isin(KEY(new["s1_row"], new["t_row"]), exclude)]
        if new.empty:
            continue
        out.append(_cosines(m, s1, new))
    return _blocking_schema(out)


def _cosines(m: dict, s1: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Add the name/address blocking cosines for (s1_row, t_row) pairs of one country."""
    loc = pd.Series(np.arange(len(m["idx"])), index=m["idx"])
    us = np.unique(new["s1_row"].to_numpy())
    qs = s1.iloc[us]
    qi = pd.Series(np.arange(len(us)), index=us).reindex(new["s1_row"]).to_numpy()
    tj = loc.reindex(new["t_row"]).to_numpy()
    new = new.copy()
    new["cos_name"] = pair_cosine(m["name"].transform(name_doc(qs)), m["name"].T, qi, tj)
    new["cos_addr"] = pair_cosine(m["addr"].transform(addr_doc(qs)), m["addr"].T, qi, tj)
    return new


def with_cosines(blocker: Blocker, s1: pd.DataFrame, tg: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """New candidate pairs from any source -> blocking-output schema with both cosines."""
    cty = tg["country"].to_numpy()[pairs["t_row"].to_numpy()]
    out = [_cosines(m, s1, pairs[cty == c]) for c, m in blocker.by_country.items() if (cty == c).any()]
    return _blocking_schema(out)


def _blocking_schema(out: list) -> pd.DataFrame:
    cols = ["s1_row", "t_row", *PASSES, *EXTRA_PASSES, *[f"rank_{p}" for p in PASSES], "cos_name", "cos_addr"]
    if not out:
        return pd.DataFrame(columns=cols)
    new = pd.concat(out, ignore_index=True)
    for p in (*PASSES, *EXTRA_PASSES):
        new[p] = False
    for p in PASSES:
        new[f"rank_{p}"] = np.nan
    return new[cols]
