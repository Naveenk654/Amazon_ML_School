"""Evaluate the frozen M3 stack on a held-out S1 set through the PRODUCTION code path.

    PYTHONPATH=src python experiments/scripts/eval_m3_heldout.py --role val   # must reproduce M3 (0.95871)
    PYTHONPATH=src python experiments/scripts/eval_m3_heldout.py --role rest  # independent held-out set

Phase A (stack.score_batch, stage3=True) scores the role's S1. Phase B
(stack.stage3_scores) runs over the full training-split population: the role's
fresh pairs + every other scored S1 (train out-of-fold s2, tune/val/rest frozen
s2). Reports M3 vs the M2-E level on the same S1.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import resource
import time

import numpy as np
import pandas as pd

from business_entity_resolution import evaluate as ev
from business_entity_resolution.blocking import Blocker
from business_entity_resolution.config import BlockingConfig, ValidationConfig, variant, work_dir
from business_entity_resolution.data import assign_folds, load_ground_truth
from business_entity_resolution.expansion import KEY
from business_entity_resolution.pipeline import key_freqs, make_roles, prepared
from business_entity_resolution.stack import StackModels, score_batch, stage3_scores

P_COLS = ["s1_row", "t_row", "s2", "cos_name", "cos_addr"]


def role_rows(s1, nm, role):
    vcfg = ValidationConfig()
    roles = make_roles(s1, nm, vcfg)
    if role == "val":
        return np.sort(roles["val"])
    fold = assign_folds(s1, nm, vcfg.n_folds)
    return np.setdiff1d(np.flatnonzero(np.isin(fold, (vcfg.tune_fold, vcfg.val_fold))),
                        np.concatenate([roles["tune"], roles["val"]]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["val", "rest"], required=True)
    ap.add_argument("--phase", choices=["A", "B"], required=True)
    ap.add_argument("--batch", type=int, default=60_000)
    ap.add_argument("--m3-name", default=None, help="phase B only: evaluate a candidate M3 variant")
    a = ap.parse_args()
    t0 = time.time()
    wd = work_dir()
    out = wd / "heldout" / f"{a.role}{variant()}"
    out.mkdir(parents=True, exist_ok=True)
    M = StackModels.load(wd)
    if a.m3_name:  # evaluate a candidate M3 variant (work/models/<name>.txt)
        import lightgbm as lgb
        M.m3 = lgb.Booster(model_file=str(wd / "models" / f"{a.m3_name}.txt"))
        info = json.loads((wd / "exp" / f"{a.m3_name}.json").read_text())
        M.m3_thr, M.m3_cut = info["val"]["threshold"], info["min_p1"]

    if a.phase == "A":
        s1, tg = prepared("train")
        s1f, tf = key_freqs(s1, tg)
        gs = pd.Index(s1["entity_id"]).get_indexer(load_ground_truth("train")["s1_id"])
        rows = role_rows(s1, np.bincount(gs, minlength=len(s1)), a.role)
        blocker = Blocker(tg, BlockingConfig(hybrid=True))
        for i, lo in enumerate(range(0, len(rows), a.batch)):
            e, keep = score_batch(blocker, s1, tg, s1f, tf, rows[lo:lo + a.batch], M, stage3=True)
            e[P_COLS].to_parquet(out / f"P_{i:03d}.parquet", index=False)
            keep.to_parquet(out / f"XS3_{i:03d}.parquet", index=False)
            logging.info("batch %d: %d pairs, %d stage-3 rows (%.0fs)", i, len(e), len(keep), time.time() - t0)
        meta = {"n_s1": int(len(rows)), "runtime_s": time.time() - t0,
                "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
        (out / "phaseA.json").write_text(json.dumps(meta, indent=1))
        return

    # Phase B: population = role's fresh pairs + all OTHER scored S1 of the training split
    own = [pd.read_parquet(f) for f in sorted(glob.glob(str(out / "P_*.parquet")))]
    n_own = [len(x) for x in own]
    others = []
    for role in ("train", "tune", "val"):
        if role == a.role:
            continue
        for cf in sorted(glob.glob(str(wd / "matcher_data" / f"CE{variant()}" / role / "cand_*.parquet"))):
            c = pd.read_parquet(cf, columns=["s1_row", "t_row", "cos_name", "cos_addr"])
            c["s2"] = pd.read_parquet(cf.replace("cand_", "s2_"))["seed"].to_numpy()
            others.append(c[P_COLS])
    if a.role != "rest":
        others += [pd.read_parquet(f, columns=P_COLS) for f in sorted(glob.glob(str(wd / "stack" / f"rest{variant()}" / "part_*.parquet")))]
    P = pd.concat(own + others, ignore_index=True)
    del others
    offs = np.r_[0, np.cumsum(n_own)]
    XS3 = []
    for i in range(len(own)):
        x = pd.read_parquet(out / f"XS3_{i:03d}.parquet")
        x["row"] = x["row"].to_numpy() + offs[i]
        XS3.append(x)
    XS3 = pd.concat(XS3, ignore_index=True)
    key_id = pd.factorize(pd.read_parquet(wd / "train_s1_norm.parquet", columns=["name_key"])["name_key"])[0]
    final = stage3_scores(P, key_id, XS3, M)
    del XS3
    n = offs[-1]
    R = P.iloc[:n][["s1_row", "t_row", "s2"]].copy()
    R["m3"] = final[:n]
    s1e = pd.read_parquet(wd / "train_s1_norm.parquet", columns=["entity_id", "country"])
    gt = load_ground_truth("train")
    gs = pd.Index(s1e["entity_id"]).get_indexer(gt["s1_id"])
    gtt = pd.Index(pd.read_parquet(wd / "train_t_norm.parquet", columns=["entity_id"])["entity_id"]).get_indexer(gt["t_id"])
    ids = np.unique(R["s1_row"].to_numpy())
    rows = role_rows(s1e, np.bincount(gs, minlength=len(s1e)), a.role)
    assert np.array_equal(np.sort(rows)[np.isin(np.sort(rows), ids)], ids)
    sel = np.isin(gs, rows)
    gtr = pd.DataFrame({"s1_row": gs[sel], "t_row": gtt[sel]})
    truth = ev.as_sets(gtr, "s1_row", "t_row")
    R["y"] = np.isin(KEY(R["s1_row"], R["t_row"]), np.sort(KEY(gtr["s1_row"], gtr["t_row"]))).astype(np.int8)
    rows = np.sort(rows)
    res = {"role": a.role, "n_s1": int(len(rows)), "pairs": int(n), "true_pairs": int(len(gtr)),
           "blocking_recall": float(R["y"].sum() / len(gtr))}
    per = {}
    for name, col, thr in (("M2E_level", "s2", M.m2e_thr), ("M3", "m3", M.m3_thr)):
        p = R[col].to_numpy() >= thr
        y = R["y"].to_numpy() == 1
        pred = ev.as_sets(R.loc[p, ["s1_row", "t_row"]], "s1_row", "t_row")
        f = np.array([ev.f05(pred.get(s, set()), truth.get(s, set())) for s in rows])
        per[name] = f
        sg = np.array([s not in truth for s in rows])
        cty = s1e["country"].to_numpy()[rows]
        res[name] = {"threshold": thr, "macro_f05": float(f.mean()),
                     "pair_precision": float((p & y).sum() / p.sum()), "pair_recall": float((p & y).sum() / len(gtr)),
                     "FP": int((p & ~y).sum()), "FN": int(len(gtr) - (p & y).sum()),
                     "singleton_f05": float(f[sg].mean()),
                     "by_country": {c: float(f[cty == c].mean()) for c in np.unique(cty)}}
    d = per["M3"] - per["M2E_level"]
    bs = d[np.random.default_rng(0).integers(0, len(d), (1000, len(d)))].mean(1)
    res["delta_M3_vs_M2E"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res["runtime_phaseB_s"] = time.time() - t0
    res["peak_rss_phaseB_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    (out / f"result_{a.m3_name or 'M3'}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
