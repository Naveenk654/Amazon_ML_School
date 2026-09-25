"""Stage-1 (M1b-style) scores for every shard, without in-sample leakage.

    PYTHONPATH=src python experiments/scripts/stage1_scores.py --blocking C --model M1b

train role : K-fold out-of-fold scores (folds by S1; each fold model uses the
             M1b configuration and M1b's best iteration, trained on the others).
tune / val : scores of the saved M1b model (out-of-sample by construction).
Writes p1_XXX.parquet (column p1) next to each cand_XXX.parquet.
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

log = logging.getLogger("stage1")


def shards(d):
    return sorted(glob.glob(f"{d}/cand_*.parquet"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocking", default="C")
    ap.add_argument("--model", default="M1b")
    ap.add_argument("--folds", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    wd = work_dir()
    base = wd / "matcher_data" / a.blocking
    info = json.loads((wd / "exp" / f"{a.model}.json").read_text())
    rounds = int(info["best_iteration"])
    model = lgb.Booster(model_file=str(wd / "models" / f"{a.model}.txt"))

    for role in ("tune", "val"):
        for cf in shards(base / role):
            X = pd.read_parquet(cf.replace("cand_", "X_")).to_numpy(np.float32)
            p = model.predict(X, num_iteration=rounds)
            pd.DataFrame({"p1": p.astype(np.float32)}).to_parquet(cf.replace("cand_", "p1_"), index=False)

    # train: out-of-fold
    files = shards(base / "train")
    s1r = [pd.read_parquet(f, columns=["s1_row"])["s1_row"].to_numpy() for f in files]
    ys = [pd.read_parquet(f, columns=["y"])["y"].to_numpy() for f in files]
    Xs = [pd.read_parquet(f.replace("cand_", "X_")).to_numpy(np.float32) for f in files]
    cols = list(pd.read_parquet(files[0].replace("cand_", "X_")).columns)
    uniq = np.unique(np.concatenate(s1r))
    fold_of = pd.Series(np.random.default_rng(SEED + 1).integers(0, a.folds, len(uniq)), index=uniq)
    folds = [fold_of.reindex(r).to_numpy() for r in s1r]
    oof = [np.zeros(len(r), np.float32) for r in s1r]
    params = ModelConfig().params
    for k in range(a.folds):
        n = sum(int((f != k).sum()) for f in folds)
        Xtr = np.empty((n, Xs[0].shape[1]), np.float32)  # preallocated: no vstack copy
        lo = 0
        for X, f in zip(Xs, folds):
            b = X[f != k]
            Xtr[lo:lo + len(b)] = b
            lo += len(b)
            del b
        ytr = np.concatenate([y[f != k] for y, f in zip(ys, folds)])
        m = lgb.train(params, lgb.Dataset(Xtr, label=ytr, feature_name=cols), num_boost_round=rounds)
        del Xtr, ytr
        for i, (X, f) in enumerate(zip(Xs, folds)):
            if (f == k).any():
                oof[i][f == k] = m.predict(X[f == k])
        log.info("fold %d done (%.0fs)", k, time.time() - t0)
    for f, p in zip(files, oof):
        pd.DataFrame({"p1": p}).to_parquet(f.replace("cand_", "p1_"), index=False)
    meta = {"model": a.model, "rounds": rounds, "folds": a.folds, "runtime_s": time.time() - t0,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    (base / "stage1_meta.json").write_text(json.dumps(meta, indent=1))
    log.info("done %s", meta)


if __name__ == "__main__":
    main()
