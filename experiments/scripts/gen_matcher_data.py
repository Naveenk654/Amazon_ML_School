"""Generate candidates + matcher features for a validation role under a blocking config.

    PYTHONPATH=src python experiments/scripts/gen_matcher_data.py --role train --blocking C
    PYTHONPATH=src python experiments/scripts/gen_matcher_data.py --role tune  --blocking C
    PYTHONPATH=src python experiments/scripts/gen_matcher_data.py --role val   --blocking C

Writes parquet shards to $BER_WORK/matcher_data/<blocking>/<role>/ :
  cand_XXX.parquet  (s1_row, t_row, y, blocking flags/cosines)
  X_XXX.parquet     (matcher features, same row order)
Roles come from the deterministic EXP001 split; --role train uses ALL
training-fold S1 (not only the 300k sample used by EXP001).
"""
from __future__ import annotations

import argparse
import json
import logging
import resource
import time

import numpy as np
import pandas as pd

from business_entity_resolution.blocking import Blocker
from business_entity_resolution.config import BlockingConfig, ValidationConfig, work_dir
from business_entity_resolution.data import load_ground_truth
from business_entity_resolution.features import build_features
from business_entity_resolution.pipeline import generate, key_freqs, make_roles, prepared

BLOCKING = {"E0": {}, "C": {"hybrid": True}}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["train", "tune", "val"], required=True)
    ap.add_argument("--blocking", choices=list(BLOCKING), default="C")
    ap.add_argument("--batch", type=int, default=200_000)
    a = ap.parse_args()
    t0 = time.time()
    out = work_dir() / "matcher_data" / a.blocking / a.role
    out.mkdir(parents=True, exist_ok=True)

    s1, tg = prepared("train")
    s1f, tf = key_freqs(s1, tg)
    gt = load_ground_truth("train")
    gs = pd.Index(s1["entity_id"]).get_indexer(gt["s1_id"])
    gtt = pd.Index(tg["entity_id"]).get_indexer(gt["t_id"])
    del gt
    vcfg = ValidationConfig(n_train=10**9)  # same permutation; train = all training-fold S1
    rows = np.sort(make_roles(s1, np.bincount(gs, minlength=len(s1)), vcfg)[a.role])
    gkey = np.sort(gs.astype(np.int64) << 32 | gtt)

    blocker = Blocker(tg, BlockingConfig(**BLOCKING[a.blocking]))
    n_pairs = 0
    for i, lo in enumerate(range(0, len(rows), a.batch)):
        r = rows[lo:lo + a.batch]
        cand = generate(blocker, s1, tg, r)
        k = cand["s1_row"].to_numpy().astype(np.int64) << 32 | cand["t_row"].to_numpy()
        cand["y"] = np.isin(k, gkey).astype(np.int8)
        X = build_features(cand, s1, tg, s1f, tf)
        cand.to_parquet(out / f"cand_{i:03d}.parquet", index=False)
        X.to_parquet(out / f"X_{i:03d}.parquet", index=False)
        n_pairs += len(cand)
        del cand, X
    meta = {"role": a.role, "blocking": a.blocking, "n_s1": int(len(rows)), "n_pairs": n_pairs,
            "runtime_s": time.time() - t0,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    logging.info("done %s", meta)


if __name__ == "__main__":
    main()
