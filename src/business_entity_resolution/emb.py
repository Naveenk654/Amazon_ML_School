"""Embedding retrieval (M5-EMB): precomputed top-K neighbours from multilingual-e5-small.

The neighbour lists come from experiments/modal/emb_full.py: for every S1 of a
split, its K=30 nearest S2/S3 records of the same country (cosine of
normalized embeddings of "name | address"). Here they
  - add the top-k neighbours of each S1 as extra candidates, and
  - give every candidate pair embedding features (NaN when the pair is not in
    the S1's top-30, so the model gets a lower bound instead).
Label-free and computed identically for train and test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import work_dir
from .expansion import KEY
from .features import EMB_COLS  # noqa: F401 (re-exported)


class EmbKnn:
    """Neighbour lists restricted to a set of S1 rows (keeps memory proportional to the batch)."""

    def __init__(self, split: str, rows: np.ndarray, path=None):
        path = path or work_dir() / "emb" / f"{split}_knn.parquet"
        rows = np.unique(rows)
        parts = []
        for rg in range(pq.ParquetFile(path).num_row_groups):
            t = pq.ParquetFile(path).read_row_group(rg).to_pandas()
            parts.append(t[np.isin(t["s1_row"].to_numpy(), rows)])
        K = pd.concat(parts, ignore_index=True)
        key = KEY(K["s1_row"], K["t_row"])
        o = np.argsort(key)
        self.key = key[o]
        self.cos = K["cos"].to_numpy(np.float32)[o]
        self.rank = K["rank"].to_numpy(np.int16)[o]
        self.s1 = K["s1_row"].to_numpy()[o]
        self.t = K["t_row"].to_numpy()[o]
        g = pd.DataFrame({"s": self.s1, "c": self.cos})
        self.top1 = g.groupby("s")["c"].max()
        self.kth = g.groupby("s")["c"].min()

    def pairs(self, rows: np.ndarray, k: int) -> pd.DataFrame:
        m = (self.rank < k) & np.isin(self.s1, rows)
        return pd.DataFrame({"s1_row": self.s1[m], "t_row": self.t[m]})

    def features(self, s1_row: np.ndarray, t_row: np.ndarray) -> pd.DataFrame:
        key = KEY(s1_row, t_row)
        i = np.minimum(np.searchsorted(self.key, key), len(self.key) - 1)
        hit = self.key[i] == key if len(self.key) else np.zeros(len(key), bool)
        F = pd.DataFrame({"emb_cos": np.where(hit, self.cos[i], np.nan),
                          "emb_rank": np.where(hit, self.rank[i], np.nan)})
        F["emb_top1"] = self.top1.reindex(s1_row).to_numpy()
        F["emb_kth"] = self.kth.reindex(s1_row).to_numpy()
        F["emb_gap"] = F["emb_top1"] - F["emb_cos"]
        return F.astype(np.float32)


def add_emb(blocker, s1: pd.DataFrame, tg: pd.DataFrame, e: pd.DataFrame, knn: EmbKnn,
            rows: np.ndarray, k: int) -> pd.DataFrame:
    """Append the top-k embedding neighbours not already in e, then attach EMB_COLS to every row.

    e must be sorted by (s1_row, t_row); the result is too.
    """
    from .expansion import with_cosines
    new = knn.pairs(rows, k)
    new = new[~np.isin(KEY(new["s1_row"], new["t_row"]), KEY(e["s1_row"], e["t_row"]))]
    n0 = len(e)
    if len(new):
        new = with_cosines(blocker, s1, tg, new)
        for c in e.columns:
            if c not in new.columns:
                new[c] = False if e[c].dtype == bool else 0
        e = pd.concat([e, new[e.columns]], ignore_index=True)
    x_emb = np.r_[np.zeros(n0, bool), np.ones(len(e) - n0, bool)]
    F = knn.features(e["s1_row"].to_numpy(), e["t_row"].to_numpy())
    for c in F.columns:
        e[c] = F[c].to_numpy()
    e["x_emb"] = x_emb
    return e.sort_values(["s1_row", "t_row"], kind="stable").reset_index(drop=True)
