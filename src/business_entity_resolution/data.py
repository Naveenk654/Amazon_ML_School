"""Loading the challenge TSVs, ground truth and validation folds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEED, data_dir

READ_KW = dict(sep="\t", dtype=str, keep_default_na=False)


def load_sources(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (s1, targets) for 'train' or 'test'.

    targets = S2 + S3 concatenated; `src` (2 or 3) is taken from the file the
    record came from, not parsed from the ID.
    """
    d = data_dir() / split
    s1 = pd.read_csv(d / f"{split}_source1.tsv", **READ_KW)
    parts = []
    for k in (2, 3):
        t = pd.read_csv(d / f"{split}_source{k}.tsv", **READ_KW)
        t["src"] = np.int8(k)
        parts.append(t)
    targets = pd.concat(parts, ignore_index=True)
    return s1, targets


def load_ground_truth(split: str = "train") -> pd.DataFrame:
    """Return long-form (s1_id, t_id) true pairs."""
    gt = pd.read_csv(data_dir() / split / f"{split}_ground_truth.tsv", **READ_KW)
    lists = gt["matched_entity_ids"].map(lambda x: x.split(",") if x else [])
    n = lists.map(len).to_numpy()
    return pd.DataFrame(
        {
            "s1_id": np.repeat(gt["source1_entity_id"].to_numpy(), n),
            "t_id": np.concatenate([np.asarray(l, dtype=object) for l in lists if l]),
        }
    )


def assign_folds(s1: pd.DataFrame, n_matches: np.ndarray, n_folds: int) -> np.ndarray:
    """Stratified (country x match-count bucket) random fold per S1 entity.

    An S1 and all of its true targets share the fold, because every target
    belongs to exactly one S1 (verified in the audit). IDs are not used.
    """
    rng = np.random.default_rng(SEED)
    strata = s1["country"].astype(str) + "|" + np.minimum(n_matches, 6).astype(str)
    fold = np.empty(len(s1), dtype=np.int8)
    for _, idx in pd.Series(np.arange(len(s1))).groupby(strata.to_numpy()):
        idx = rng.permutation(idx.to_numpy())
        fold[idx] = np.arange(len(idx)) % n_folds
    return fold
