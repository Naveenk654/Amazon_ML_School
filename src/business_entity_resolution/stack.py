"""Inference through the full model stack (the M4-NE-R pipeline).

    C blocking -> features -> M1b (p1) -> siblings -> M2 (seed)
      -> neighbour expansion of seeds -> features over the enlarged set
      -> M1b-E (p1E) -> siblings -> M2-E (final stage-2 score s2)

Every model is loaded from disk; nothing here uses labels.
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .blocking import Blocker
from .expansion import KEY, expand
from .features import build_features
from .siblings import sibling_features

log = logging.getLogger(__name__)


@dataclass
class StackModels:
    m1b: lgb.Booster
    m1b_iter: int
    m2: lgb.Booster
    m2_cut: float
    m2_thr: float
    m1be: lgb.Booster
    m1be_iter: int
    m2e: lgb.Booster
    m2e_cut: float
    m2e_thr: float

    @classmethod
    def load(cls, work: Path) -> "StackModels":
        j = lambda n: json.loads((work / "exp" / f"{n}.json").read_text())
        b = lambda n: lgb.Booster(model_file=str(work / "models" / f"{n}.txt"))
        return cls(b("M1b"), int(j("M1b")["best_iteration"]), b("M2"), j("M2")["min_p1"],
                   j("M2")["val"]["threshold"], b("M1bE"), int(j("M1bE")["best_iteration"]),
                   b("M2E"), j("M2E")["min_p1"], j("M2E")["val"]["threshold"])


def _stage(cand, s1, tg, s1f, tf, m1, it1, m2, cut):
    """features -> stage-1 p1 -> siblings -> cascaded stage-2 score (memory-lean)."""
    X = build_features(cand, s1, tg, s1f, tf)
    p1 = m1.predict(X[m1.feature_name()], num_iteration=it1)
    s2 = p1.astype(np.float64).copy()
    m = p1 >= cut
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        S = sibling_features(pd.DataFrame({"s1_row": cand["s1_row"].to_numpy(),
                                           "t_row": cand["t_row"].to_numpy(), "p1": p1.astype(np.float32)}), tg)
    if m.any():
        XS = pd.concat([X[m].reset_index(drop=True), S[m].reset_index(drop=True)], axis=1)
        del X, S
        s2[m] = m2.predict(XS[m2.feature_name()])
    return s2


def score_batch(blocker: Blocker, s1, tg, s1f, tf, rows: np.ndarray, M: StackModels, k: int = 5):
    """Return the final enlarged candidate frame for S1 `rows` with column s2."""
    c = blocker.candidates(s1.iloc[rows])
    c["s1_row"] = rows[c["s1_row"].to_numpy()]
    seed = _stage(c, s1, tg, s1f, tf, M.m1b, M.m1b_iter, M.m2, M.m2_cut)
    seeds = c.loc[seed >= M.m2_thr, ["s1_row", "t_row"]]
    new = expand(blocker, s1, tg, seeds, np.sort(KEY(c["s1_row"], c["t_row"])), k)
    e = pd.concat([c, new[c.columns]], ignore_index=True)
    e["x_expand"] = np.r_[np.zeros(len(c), bool), np.ones(len(new), bool)]
    e = e.sort_values(["s1_row", "t_row"], kind="stable").reset_index(drop=True)
    s2 = _stage(e, s1, tg, s1f, tf, M.m1be, M.m1be_iter, M.m2e, M.m2e_cut)
    e["s2"] = s2.astype(np.float32)
    return e
