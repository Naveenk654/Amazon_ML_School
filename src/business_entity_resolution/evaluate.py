"""Challenge-style evaluation: blocking quality, pair metrics, macro F0.5."""
from __future__ import annotations

import numpy as np
import pandas as pd

BETA2 = 0.25  # beta = 0.5


def macro_f05(pred: dict, truth: dict, s1_ids) -> float:
    """Per-S1 F0.5 averaged over s1_ids; empty/empty scores 1.0 (singleton)."""
    return float(np.mean([f05(pred.get(s, set()), truth.get(s, set())) for s in s1_ids]))


def f05(p: set, t: set) -> float:
    if not t:
        return 1.0 if not p else 0.0
    tp = len(p & t)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(t)
    return (1 + BETA2) * prec * rec / (BETA2 * prec + rec)


def as_sets(pairs: pd.DataFrame, a: str = "s1_id", b: str = "t_id") -> dict:
    return pairs.groupby(a)[b].agg(set).to_dict()


def blocking_report(cand: pd.DataFrame, truth_pairs: pd.DataFrame, s1_ids: np.ndarray,
                    space: float, passes) -> dict:
    """Recall per pass and for the union, candidate volume and reduction ratio.

    cand        : columns s1_id, t_id, <pass flags>
    truth_pairs : true (s1_id, t_id) restricted to s1_ids
    space       : number of possible (S1, target) pairs for these S1 (all-pairs)
    """
    key = cand["s1_id"] + "|" + cand["t_id"]
    tkey = set(truth_pairs["s1_id"] + "|" + truth_pairs["t_id"])
    is_true = key.isin(tkey).to_numpy()
    n_true = len(tkey)
    rep = {"true_pairs": n_true}
    for p in passes:
        m = cand[p].to_numpy()
        rep[f"recall_{p}"] = is_true[m].sum() / n_true
        rep[f"cands_{p}"] = int(m.sum())
    rep["recall_union"] = is_true.sum() / n_true
    per = cand.groupby("s1_id").size().reindex(s1_ids, fill_value=0)
    rep["cands_total"] = int(len(cand))
    rep["cands_per_s1_mean"] = float(per.mean())
    rep["cands_per_s1_median"] = float(per.median())
    rep["cands_per_s1_p95"] = float(per.quantile(0.95))
    rep["s1_with_zero_cands"] = float((per == 0).mean())
    rep["reduction_ratio"] = 1 - len(cand) / space
    rep["pair_precision_of_candidates"] = is_true.sum() / max(len(cand), 1)
    return rep


def oracle_f05(cand: pd.DataFrame, truth: dict, s1_ids) -> float:
    """Macro F0.5 of a perfect matcher restricted to the candidate set (ceiling)."""
    cs = as_sets(cand)
    pred = {s: cs.get(s, set()) & truth.get(s, set()) for s in s1_ids}
    return macro_f05(pred, truth, s1_ids)


def pair_metrics(pred_pairs: pd.DataFrame, truth_pairs: pd.DataFrame) -> dict:
    pk = set(pred_pairs["s1_id"] + "|" + pred_pairs["t_id"])
    tk = set(truth_pairs["s1_id"] + "|" + truth_pairs["t_id"])
    tp = len(pk & tk)
    return {"pair_precision": tp / max(len(pk), 1), "pair_recall": tp / max(len(tk), 1),
            "pred_pairs": len(pk)}


def breakdown(pred: dict, truth: dict, s1: pd.DataFrame) -> pd.DataFrame:
    """Macro F0.5 by country and by true-match-count bucket."""
    d = s1[["entity_id", "country"]].copy()
    d["n_true"] = [len(truth.get(s, ())) for s in d["entity_id"]]
    d["n_pred"] = [len(pred.get(s, ())) for s in d["entity_id"]]
    d["f05"] = [f05(pred.get(s, set()), truth.get(s, set())) for s in d["entity_id"]]
    d["bucket"] = np.where(d["n_true"] == 0, "singleton",
                           np.where(d["n_true"] <= 2, "1-2", np.where(d["n_true"] <= 4, "3-4", "5+")))
    rows = []
    for col in ("country", "bucket"):
        g = d.groupby(col).agg(n=("f05", "size"), macro_f05=("f05", "mean"),
                               mean_pred=("n_pred", "mean"), mean_true=("n_true", "mean"))
        g.index = [f"{col}={i}" for i in g.index]
        rows.append(g)
    return pd.concat(rows)
