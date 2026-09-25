"""Inference through the full model stack (M3 pipeline).

    C blocking -> features -> M1b (p1) -> siblings -> M2 (seed)
      -> neighbour expansion of seeds -> features over the enlarged set
      -> M1b-E (p1E) -> siblings -> M2-E (stage-2 score s2)          [per S1 batch]
      -> cross-S1 competition features over ALL scored pairs -> M3   [needs every S1]

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
from .competition import competition_features
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
    m3: lgb.Booster | None = None
    m3_cut: float = 0.0
    m3_thr: float = 0.0

    @classmethod
    def load_dir(cls, d: Path) -> "StackModels":
        """Load the frozen models shipped in a self-contained directory (models/ + meta.json)."""
        meta = json.loads((d / "meta.json").read_text())
        b = lambda n: lgb.Booster(model_file=str(d / f"{n}.txt"))
        return cls(b("M1b"), meta["M1b"]["best_iteration"], b("M2"), meta["M2"]["cascade"],
                   meta["M2"]["threshold"], b("M1bE"), meta["M1bE"]["best_iteration"],
                   b("M2E"), meta["M2E"]["cascade"], meta["M2E"]["threshold"],
                   b("M3"), meta["M3"]["cascade"], meta["M3"]["threshold"])

    @classmethod
    def load(cls, work: Path) -> "StackModels":
        j = lambda n: json.loads((work / "exp" / f"{n}.json").read_text())
        b = lambda n: lgb.Booster(model_file=str(work / "models" / f"{n}.txt"))
        m3 = (b("M3"), j("M3")["min_p1"], j("M3")["val"]["threshold"]) \
            if (work / "models" / "M3.txt").exists() else (None, 0.0, 0.0)
        return cls(b("M1b"), int(j("M1b")["best_iteration"]), b("M2"), j("M2")["min_p1"],
                   j("M2")["val"]["threshold"], b("M1bE"), int(j("M1bE")["best_iteration"]),
                   b("M2E"), j("M2E")["min_p1"], j("M2E")["val"]["threshold"], *m3)


def _stage(cand, s1, tg, s1f, tf, m1, it1, m2, cut, keep_min=None):
    """features -> stage-1 p1 -> siblings -> cascaded stage-2 score (memory-lean).

    With keep_min, also returns the stage-2 feature rows (X + sibling) for
    candidates whose stage-2 score >= keep_min (the stage-3 cascade), with
    column `row` = position in `cand`.
    """
    X = build_features(cand, s1, tg, s1f, tf)
    p1 = m1.predict(X[m1.feature_name()], num_iteration=it1)
    s2 = p1.astype(np.float64).copy()
    m = p1 >= cut
    keep = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        S = sibling_features(pd.DataFrame({"s1_row": cand["s1_row"].to_numpy(),
                                           "t_row": cand["t_row"].to_numpy(), "p1": p1.astype(np.float32)}), tg)
    if m.any():
        XS = pd.concat([X[m].reset_index(drop=True), S[m].reset_index(drop=True)], axis=1)
        del X, S
        s2[m] = m2.predict(XS[m2.feature_name()])
        if keep_min is not None:
            sel = s2[m] >= keep_min                     # s2 >= keep_min implies p1 >= cut here
            keep = XS[sel].reset_index(drop=True)
            keep["row"] = np.flatnonzero(m)[sel]
    return (s2, keep) if keep_min is not None else s2


def score_batch(blocker: Blocker, s1, tg, s1f, tf, rows: np.ndarray, M: StackModels, k: int = 5,
                stage3: bool = False):
    """Return the final enlarged candidate frame for S1 `rows` with column s2.

    stage3=True also returns the stage-3 input rows (see _stage).
    """
    c = blocker.candidates(s1.iloc[rows])
    c["s1_row"] = rows[c["s1_row"].to_numpy()]
    seed = _stage(c, s1, tg, s1f, tf, M.m1b, M.m1b_iter, M.m2, M.m2_cut)
    seeds = c.loc[seed >= M.m2_thr, ["s1_row", "t_row"]]
    new = expand(blocker, s1, tg, seeds, np.sort(KEY(c["s1_row"], c["t_row"])), k)
    e = pd.concat([c, new[c.columns]], ignore_index=True)
    e["x_expand"] = np.r_[np.zeros(len(c), bool), np.ones(len(new), bool)]
    e = e.sort_values(["s1_row", "t_row"], kind="stable").reset_index(drop=True)
    if not stage3:
        e["s2"] = _stage(e, s1, tg, s1f, tf, M.m1be, M.m1be_iter, M.m2e, M.m2e_cut).astype(np.float32)
        return e
    s2, keep = _stage(e, s1, tg, s1f, tf, M.m1be, M.m1be_iter, M.m2e, M.m2e_cut, keep_min=M.m3_cut)
    e["s2"] = s2.astype(np.float32)
    return e, keep


def stage3_scores(P: pd.DataFrame, s1_key_id: np.ndarray, XS3: pd.DataFrame, M: StackModels) -> np.ndarray:
    """Final M3 scores over the whole scored population P.

    P   : s1_row, t_row, s2, cos_name, cos_addr for EVERY scored pair (all S1 of the split)
    XS3 : stage-2 feature rows for the cascade, column `row` = position in P
    Rows outside the cascade keep s2 (always below the M3 threshold).
    """
    Q = competition_features(P["s1_row"].to_numpy(), P["t_row"].to_numpy(), P["s2"].to_numpy(),
                             P["cos_name"].to_numpy(), P["cos_addr"].to_numpy(), s1_key_id)
    out = P["s2"].to_numpy().astype(np.float64).copy()
    feats = M.m3.feature_name()
    for lo in range(0, len(XS3), 1_000_000):          # chunked: bounded memory
        x = XS3.iloc[lo:lo + 1_000_000]
        rows = x["row"].to_numpy()
        F = pd.concat([x.drop(columns="row").reset_index(drop=True), Q.iloc[rows].reset_index(drop=True)], axis=1)
        out[rows] = M.m3.predict(F[feats])
    return out
