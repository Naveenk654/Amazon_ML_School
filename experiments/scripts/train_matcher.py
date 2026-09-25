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
import pyarrow.parquet as pq

from business_entity_resolution import evaluate as ev
from business_entity_resolution.config import ModelConfig, ValidationConfig, work_dir
from business_entity_resolution.data import load_ground_truth
from business_entity_resolution.pipeline import make_roles

log = logging.getLogger("train_matcher")


def load_role(d, keep_s1=None, extra=(), min_p1=None, drop_prefix=None):
    """Load cand + features into one preallocated float32 array (no vstack copy).

    `extra` = shard prefixes appended column-wise (e.g. 'S_' sibling features).
    """
    files = sorted(glob.glob(f"{d}/cand_*.parquet"))
    cands, masks = [], []
    for cf in files:
        c = pd.read_parquet(cf, columns=["s1_row", "t_row", "y"])
        m = np.isin(c["s1_row"].to_numpy(), keep_s1) if keep_s1 is not None else np.ones(len(c), bool)
        if min_p1 is not None:  # cascade: stage 2 only sees candidates with p1 >= min_p1
            m &= pd.read_parquet(cf.replace("cand_", "p1_"))["p1"].to_numpy() >= min_p1
        cands.append(c[m].reset_index(drop=True))
        masks.append(m)
    cols = list(pq.ParquetFile(files[0].replace("cand_", "X_")).schema_arrow.names)
    for pre in extra:
        cols += list(pq.ParquetFile(files[0].replace("cand_", pre)).schema_arrow.names)
    keep_c = [i for i, c in enumerate(cols) if not (drop_prefix and c.startswith(drop_prefix))]
    cols = [cols[i] for i in keep_c]
    out = np.empty((sum(int(m.sum()) for m in masks), len(cols)), np.float32)
    lo = 0
    for cf, m in zip(files, masks):
        parts = [pd.read_parquet(cf.replace("cand_", "X_"))]
        parts += [pd.read_parquet(cf.replace("cand_", pre)) for pre in extra]
        blk = np.hstack([x.to_numpy(np.float32) for x in parts])[m][:, keep_c]
        out[lo:lo + len(blk)] = blk
        lo += len(blk)
        del parts, blk
    return pd.concat(cands, ignore_index=True), out, cols


def cascade_predict(model, X, p1, min_p1):
    """Stage-2 score where p1 >= min_p1; below the cutoff the stage-1 score is kept."""
    it = model.best_iteration or None
    if p1 is None:
        return model.predict(X, num_iteration=it)
    s = p1.astype(np.float64).copy()
    m = p1 >= min_p1
    s[m] = model.predict(X[m], num_iteration=it)
    return s


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
    ap.add_argument("--extra", nargs="*", default=[], help="extra shard prefixes, e.g. S_")
    ap.add_argument("--min-p1", type=float, default=None,
                    help="cascade: rescore only candidates with stage-1 p1 >= this; others keep p1")
    ap.add_argument("--drop-prefix", default=None, help="drop feature columns with this prefix (ablation)")
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
    ctr, Xtr, cols = load_role(base / "train", keep, a.extra, a.min_p1, a.drop_prefix)
    ctu, Xtu, _ = load_role(base / "tune", extra=a.extra, drop_prefix=a.drop_prefix)
    if a.min_p1 is not None:
        p1_tu = np.concatenate([pd.read_parquet(f.replace("cand_", "p1_"))["p1"].to_numpy()
                                for f in sorted(glob.glob(f"{base / 'tune'}/cand_*.parquet"))])
    log.info("train pairs %d (S1 %d), tune pairs %d", len(ctr), ctr["s1_row"].nunique(), len(ctu))

    mcfg = ModelConfig()
    dtr = lgb.Dataset(Xtr, label=ctr["y"].to_numpy(), feature_name=cols, free_raw_data=True)
    t1 = time.time()
    if a.fixed_rounds:
        model = lgb.train(mcfg.params, dtr, num_boost_round=a.fixed_rounds)
    else:
        es = p1_tu >= a.min_p1 if a.min_p1 is not None else slice(None)
        dtu = lgb.Dataset(Xtu[es], label=ctu["y"].to_numpy()[es], reference=dtr)
        model = lgb.train(mcfg.params, dtr, num_boost_round=a.max_rounds, valid_sets=[dtu],
                          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)])
    t_train = time.time() - t1
    del dtr, Xtr, ctr

    # threshold on tune (macro F0.5 incl. singletons)
    tune_ids = np.sort(roles["tune"])
    truth_tu = ev.as_sets(gtr[np.isin(gs, tune_ids)], "s1_row", "t_row")
    s_tu = cascade_predict(model, Xtu, p1_tu if a.min_p1 is not None else None, a.min_p1)
    grid = np.round(np.arange(0.30, 0.901, 0.025), 3)
    curve = []
    for t in grid:
        pred = ev.as_sets(ctu.loc[s_tu >= t, ["s1_row", "t_row"]], "s1_row", "t_row")
        curve.append(ev.macro_f05(pred, truth_tu, tune_ids))
    thr = float(grid[int(np.argmax(curve))])
    del Xtu

    cva, Xva, _ = load_role(base / "val", extra=a.extra, drop_prefix=a.drop_prefix)
    val_ids = np.sort(roles["val"])
    truth_va = ev.as_sets(gtr[np.isin(gs, val_ids)], "s1_row", "t_row")
    p1_va = np.concatenate([pd.read_parquet(f.replace("cand_", "p1_"))["p1"].to_numpy()
                            for f in sorted(glob.glob(f"{base / 'val'}/cand_*.parquet"))]) \
        if a.min_p1 is not None else None
    s_va = cascade_predict(model, Xva, p1_va, a.min_p1)
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
    out = {"name": a.name, "blocking": a.blocking, "extra": a.extra, "min_p1": a.min_p1, "drop_prefix": a.drop_prefix, "n_train_s1": int(len(keep) if keep is not None else len(roles["train"])),
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
