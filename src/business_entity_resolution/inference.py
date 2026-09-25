"""Entity-level decisions and submission writing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import as_sets, macro_f05


def decide(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Baseline decision: keep every candidate with score >= threshold.

    An S1 with no candidate above threshold gets an empty list (no-match).
    """
    return pairs.loc[pairs["score"] >= threshold, ["s1_id", "t_id"]]


def tune_threshold(pairs: pd.DataFrame, truth: dict, s1_ids, grid=None) -> tuple[float, pd.DataFrame]:
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2) if grid is None else grid
    rows = []
    for t in grid:
        pred = as_sets(decide(pairs, t))
        rows.append((t, macro_f05(pred, truth, s1_ids)))
    res = pd.DataFrame(rows, columns=["threshold", "macro_f05"])
    return float(res.loc[res["macro_f05"].idxmax(), "threshold"]), res


def write_lists_rows(path, s1_ids: np.ndarray, s1_row: np.ndarray, t_ids: np.ndarray,
                     t_row: np.ndarray, col: str) -> None:
    """Memory-lean writer from integer row positions (one line per S1, input order)."""
    order = np.argsort(s1_row, kind="stable")
    sr, tr = s1_row[order], t_row[order]
    bounds = np.searchsorted(sr, np.arange(len(s1_ids) + 1))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for i, sid in enumerate(s1_ids):
            ts = dict.fromkeys(t_ids[tr[bounds[i]:bounds[i + 1]]])
            f.write(f"{sid}\t{','.join(ts)}\n")
