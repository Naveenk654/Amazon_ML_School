"""Expansion seed scores (stage-2 / M2-level) for every C shard, leak-free.

    PYTHONPATH=src python experiments/scripts/seed_scores.py                       # C / M2 -> seed_
    PYTHONPATH=src python experiments/scripts/seed_scores.py --blocking CE --model M2E --prefix s2_

train      : 4-fold out-of-fold M2 (same S1 folds as stage1_scores.py; each fold
             model = M2 configuration, M2's best iteration, cascade p1 >= min_p1,
             trained on the other folds' X + sibling features).
tune / val : the frozen M2 model (never trained on these S1).
Below the cascade cutoff the seed is p1, as at inference.
Writes seed_XXX.parquet (column `seed`) next to each C cand_XXX.parquet.
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

from business_entity_resolution.config import SEED, ModelConfig, work_dir

log = logging.getLogger("seed")


def read_xs(cf, mask=None):
    X = pd.concat([pd.read_parquet(cf.replace("cand_", "X_")), pd.read_parquet(cf.replace("cand_", "S_"))], axis=1)
    return X if mask is None else X[mask]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocking", default="C")
    ap.add_argument("--model", default="M2")
    ap.add_argument("--prefix", default="seed_")
    a = ap.parse_args()
    t0 = time.time()
    wd = work_dir()
    base = wd / "matcher_data" / a.blocking
    info = json.loads((wd / "exp" / f"{a.model}.json").read_text())
    cut, rounds = info["min_p1"], int(info["best_iteration"])
    m2 = lgb.Booster(model_file=str(wd / "models" / f"{a.model}.txt"))
    feats = m2.feature_name()

    for role in ("tune", "val"):
        for cf in sorted(glob.glob(f"{base}/{role}/cand_*.parquet")):
            p1 = pd.read_parquet(cf.replace("cand_", "p1_"))["p1"].to_numpy().astype(np.float64)
            m = p1 >= cut
            seed = p1.copy()
            seed[m] = m2.predict(read_xs(cf, m)[feats])
            pd.DataFrame({"seed": seed.astype(np.float32)}).to_parquet(cf.replace("cand_", a.prefix), index=False)

    files = sorted(glob.glob(f"{base}/train/cand_*.parquet"))
    s1r = [pd.read_parquet(f, columns=["s1_row"])["s1_row"].to_numpy() for f in files]
    ys = [pd.read_parquet(f, columns=["y"])["y"].to_numpy() for f in files]
    p1s = [pd.read_parquet(f.replace("cand_", "p1_"))["p1"].to_numpy().astype(np.float64) for f in files]
    uniq = np.unique(np.concatenate(s1r))  # identical fold assignment to stage1_scores.py
    fold_of = pd.Series(np.random.default_rng(SEED + 1).integers(0, 4, len(uniq)), index=uniq)
    folds = [fold_of.reindex(r).to_numpy() for r in s1r]
    masks = [p >= cut for p in p1s]
    Xs = [read_xs(f, m)[feats].to_numpy(np.float32) for f, m in zip(files, masks)]
    fcut = [f[m] for f, m in zip(folds, masks)]
    ycut = [y[m] for y, m in zip(ys, masks)]
    seeds = [p.copy() for p in p1s]
    for k in range(4):
        n = sum(int((f != k).sum()) for f in fcut)
        Xtr = np.empty((n, len(feats)), np.float32)
        lo = 0
        for X, f in zip(Xs, fcut):
            b = X[f != k]
            Xtr[lo:lo + len(b)] = b
            lo += len(b)
        ytr = np.concatenate([y[f != k] for y, f in zip(ycut, fcut)])
        mdl = lgb.train(ModelConfig().params, lgb.Dataset(Xtr, label=ytr, feature_name=feats),
                        num_boost_round=rounds)
        del Xtr, ytr
        for i, (X, f) in enumerate(zip(Xs, fcut)):
            sel = f == k
            if sel.any():
                idx = np.flatnonzero(masks[i])[sel]
                seeds[i][idx] = mdl.predict(X[sel])
        log.info("fold %d done (%.0fs)", k, time.time() - t0)
    for f, s in zip(files, seeds):
        pd.DataFrame({"seed": s.astype(np.float32)}).to_parquet(f.replace("cand_", a.prefix), index=False)
    meta = {"rounds": rounds, "cut": cut, "runtime_s": time.time() - t0,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    (base / f"{a.prefix}meta.json").write_text(json.dumps(meta, indent=1))
    log.info("done %s", meta)


if __name__ == "__main__":
    main()
