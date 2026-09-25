"""Train/evaluate a matcher on pre-generated shards (see gen_matcher_data.py).

    PYTHONPATH=src python experiments/scripts/train_matcher.py --name M1b --blocking C
    PYTHONPATH=src python experiments/scripts/train_matcher.py --name M1a --blocking C \
        --n-train-s1 300000 --fixed-rounds 300

Trains on role=train, early-stops and tunes the threshold on role=tune, reports
on role=val (the EXP001 validation fold). Never overwrites the baseline model.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import resource
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from business_entity_resolution import evaluate as ev
from business_entity_resolution.config import ModelConfig, ValidationConfig, work_dir
from business_entity_resolution.data import load_ground_truth
from business_entity_resolution.pipeline import make_roles

log = logging.getLogger("train_matcher")


def load_role(d, keep_s1=None):
    cands, Xs = [], []
    for cf in sorted(glob.glob(f"{d}/cand_*.parquet")):
        c = pd.read_parquet(cf, columns=["s1_row", "t_row", "y"])
        X = pd.read_parquet(cf.replace("cand_", "X_"))
        if keep_s1 is not None:
            m = np.isin(c["s1_row"].to_numpy(), keep_s1)
            c, X = c[m], X[m]
        cands.append(c.reset_index(drop=True))
        Xs.append(X.to_numpy(np.float32))
        cols = list(X.columns)
    return pd.concat(cands, ignore_index=True), np.vstack(Xs), cols


def f05_report(c, score, thr, truth, ids, singles):
    pred_pairs = c.loc[score >= thr, ["s1_row", "t_row"]]
    pred = ev.as_sets(pred_pairs, "s1_row", "t_row")
    per = np.array([ev.f05(pred.get(s, set()), truth.get(s, set())) for s in ids])
    tp = int(c["y"].to_numpy()[score >= thr].sum())
    n_true = sum(len(v) for v in truth.values())
    return {"threshold": thr, "macro_f05": float(per.mean()),
            "pair_precision": tp / max(len(pred_pairs), 1), "pair_recall": tp / n_true,
            "pred_pairs": len(pred_pairs), "false_positives": len(pred_pairs) - tp,
            "singleton_f05": float(per[singles].mean()),
            "singleton_false_match": int((per[singles] == 0).sum()),
            "pred_empty_rate": float(np.mean([s not in pred for s in ids]))}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--blocking", default="C")
    ap.add_argument("--n-train-s1", type=int, default=None, help="use EXP001's first N train S1")
    ap.add_argument("--fixed-rounds", type=int, default=None, help="no early stopping")
    ap.add_argument("--max-rounds", type=int, default=3000)
    a = ap.parse_args()
    t0 = time.time()
    wd = work_dir()
    base = wd / "matcher_data" / a.blocking

    s1 = pd.read_parquet(wd / "train_s1_norm.parquet", columns=["entity_id", "country"])
    gt = load_ground_truth("train")
    gs = pd.Index(s1["entity_id"]).get_indexer(gt["s1_id"])
    gtt = pd.Index(pd.read_parquet(wd / "train_t_norm.parquet", columns=["entity_id"])["entity_id"]) \
        .get_indexer(gt["t_id"])
    nm = np.bincount(gs, minlength=len(s1))
    roles = make_roles(s1, nm, ValidationConfig(n_train=a.n_train_s1 or 10**9))
    gtr = pd.DataFrame({"s1_row": gs, "t_row": gtt})
    del gt

    keep = np.sort(roles["train"]) if a.n_train_s1 else None
    ctr, Xtr, cols = load_role(base / "train", keep)
    ctu, Xtu, _ = load_role(base / "tune")
    log.info("train pairs %d (S1 %d), tune pairs %d", len(ctr), ctr["s1_row"].nunique(), len(ctu))

    mcfg = ModelConfig()
    dtr = lgb.Dataset(Xtr, label=ctr["y"].to_numpy(), feature_name=cols, free_raw_data=True)
    t1 = time.time()
    if a.fixed_rounds:
        model = lgb.train(mcfg.params, dtr, num_boost_round=a.fixed_rounds)
    else:
        dtu = lgb.Dataset(Xtu, label=ctu["y"].to_numpy(), reference=dtr)
        model = lgb.train(mcfg.params, dtr, num_boost_round=a.max_rounds, valid_sets=[dtu],
                          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)])
    t_train = time.time() - t1
    del dtr, Xtr, ctr

    # threshold on tune (macro F0.5 incl. singletons)
    tune_ids = np.sort(roles["tune"])
    truth_tu = ev.as_sets(gtr[np.isin(gs, tune_ids)], "s1_row", "t_row")
    s_tu = model.predict(Xtu, num_iteration=model.best_iteration or None)
    grid = np.round(np.arange(0.30, 0.901, 0.025), 3)
    curve = []
    for t in grid:
        pred = ev.as_sets(ctu.loc[s_tu >= t, ["s1_row", "t_row"]], "s1_row", "t_row")
        curve.append(ev.macro_f05(pred, truth_tu, tune_ids))
    thr = float(grid[int(np.argmax(curve))])
    del Xtu

    cva, Xva, _ = load_role(base / "val")
    val_ids = np.sort(roles["val"])
    truth_va = ev.as_sets(gtr[np.isin(gs, val_ids)], "s1_row", "t_row")
    s_va = model.predict(Xva, num_iteration=model.best_iteration or None)
    singles = np.array([s not in truth_va for s in val_ids])
    res = f05_report(cva, s_va, thr, truth_va, val_ids, singles)
    country = s1["country"].to_numpy()[val_ids]
    pred = ev.as_sets(cva.loc[s_va >= thr, ["s1_row", "t_row"]], "s1_row", "t_row")
    per = np.array([ev.f05(pred.get(s, set()), truth_va.get(s, set())) for s in val_ids])
    res["by_country"] = {c: float(per[country == c].mean()) for c in np.unique(country)}
    n_true = sum(len(v) for v in truth_va.values())
    res["blocking_recall"] = int(cva["y"].sum()) / n_true
    res["cands_per_s1"] = len(cva) / len(val_ids)
    res["ceiling"] = ev.macro_f05(ev.as_sets(cva[cva["y"] == 1], "s1_row", "t_row"), truth_va, val_ids)

    imp = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    out = {"name": a.name, "blocking": a.blocking, "n_train_s1": int(len(keep) if keep is not None else len(roles["train"])),
           "best_iteration": model.best_iteration or model.current_iteration(),
           "threshold_curve_tune": dict(zip(map(float, grid), map(float, curve))),
           "val": res, "train_time_s": t_train, "total_time_s": time.time() - t0,
           "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
           "feature_gain": (imp / imp.sum()).sort_values(ascending=False).round(4).to_dict()}
    (wd / "models").mkdir(exist_ok=True)
    model.save_model(str(wd / "models" / f"{a.name}.txt"))
    (wd / "exp" / f"{a.name}.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame({"s1_row": cva["s1_row"], "t_row": cva["t_row"], "y": cva["y"], "score": s_va}) \
        .to_parquet(wd / "exp" / f"{a.name}_val_scores.parquet", index=False)
    log.info("RESULT %s", json.dumps({k: v for k, v in out.items() if k != "feature_gain"}))


if __name__ == "__main__":
    main()
