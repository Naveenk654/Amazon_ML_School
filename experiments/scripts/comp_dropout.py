"""Experiment B: competition features under randomized competitor-S1 dropout (orphan-robust M3).

    PYTHONPATH=src python experiments/scripts/comp_dropout.py --drop 0.19

Writes QD_XXX.parquet next to each CE shard:
  train : training S1 split into 3 random groups; group g's features come from a
          population where a random `drop` share of the OTHER S1 is removed (seed g).
  tune  : its own dropout population (seed 3), matching the test-like deployment.
  val   : normal population (identical to Q_), for normal-validation reporting.
Inputs: inference-time stage-2 scores only (out-of-fold for train); no labels.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import resource
import time

import numpy as np
import pandas as pd

from business_entity_resolution.competition import competition_features
from business_entity_resolution.config import SEED, work_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", type=float, default=0.19)
    a = ap.parse_args()
    t0 = time.time()
    wd = work_dir()
    base = wd / "matcher_data" / "CE"
    files, parts = [], []
    for role in ("train", "tune", "val"):
        for cf in sorted(glob.glob(f"{base}/{role}/cand_*.parquet")):
            c = pd.read_parquet(cf, columns=["s1_row", "t_row", "cos_name", "cos_addr"])
            c["s2"] = pd.read_parquet(cf.replace("cand_", "s2_"))["seed"].to_numpy()
            files.append((role, cf, len(c)))
            parts.append(c)
    parts += [pd.read_parquet(f, columns=["s1_row", "t_row", "cos_name", "cos_addr", "s2"])
              for f in sorted(glob.glob(str(wd / "stack" / "rest" / "part_*.parquet")))]
    P = pd.concat(parts, ignore_index=True)
    del parts
    s1r = P["s1_row"].to_numpy()
    key_id = pd.factorize(pd.read_parquet(wd / "train_s1_norm.parquet", columns=["name_key"])["name_key"])[0]
    offs = np.r_[0, np.cumsum([n for _, _, n in files])]
    role_of = np.concatenate([np.full(n, r, dtype=object) for r, _, n in files])
    n_own = offs[-1]
    uniq = np.unique(s1r)
    train_s1 = np.unique(s1r[:n_own][role_of == "train"])
    tune_s1 = np.unique(s1r[:n_own][role_of == "tune"])
    group = pd.Series(np.random.default_rng(SEED + 11).integers(0, 3, len(train_s1)), index=train_s1)
    QD = None
    runs = [(g, train_s1[group.to_numpy() == g]) for g in range(3)] + [(3, tune_s1)]
    for seed, keep_s1 in runs:
        rng = np.random.default_rng(SEED + 100 + seed)
        cand_drop = np.setdiff1d(uniq, keep_s1)
        gone = cand_drop[rng.random(len(cand_drop)) < a.drop]
        live = ~np.isin(s1r, gone)
        idx = np.flatnonzero(live)
        Q = competition_features(s1r[idx], P["t_row"].to_numpy()[idx], P["s2"].to_numpy()[idx],
                                 P["cos_name"].to_numpy()[idx], P["cos_addr"].to_numpy()[idx], key_id)
        if QD is None:
            qcols = list(Q.columns)
            QD = np.full((n_own, Q.shape[1]), np.nan, np.float32)
        target = np.isin(s1r[idx], keep_s1) & (idx < n_own)
        QD[idx[target]] = Q.to_numpy(np.float32)[target]
        logging.info("dropout seed %d: removed %d S1, filled %d rows (%.0fs)", seed, len(gone), int(target.sum()), time.time() - t0)
        del Q, live, idx
        gc.collect()
    # val: normal population (no dropout) = original Q_
    for (role, cf, n), lo in zip(files, offs[:-1]):
        if role == "val":
            pd.read_parquet(cf.replace("cand_", "Q_")).to_parquet(cf.replace("cand_", "QD_"), index=False)
        else:
            blk = pd.DataFrame(QD[lo:lo + n], columns=qcols)
            assert not blk["s2"].isna().any(), f"unfilled rows in {cf}"
            blk.to_parquet(cf.replace("cand_", "QD_"), index=False)
    meta = {"drop": a.drop, "runtime_s": time.time() - t0,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    (base / "comp_dropout_meta.json").write_text(json.dumps(meta, indent=1))
    logging.info("done %s", meta)


if __name__ == "__main__":
    main()
