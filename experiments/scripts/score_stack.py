"""Score S1 through the frozen M4-NE-R stack (src/business_entity_resolution/stack.py).

    PYTHONPATH=src python experiments/scripts/score_stack.py --role val    # must reproduce M2E
    PYTHONPATH=src python experiments/scripts/score_stack.py --role rest   # competitors for M3

rest = folds 3-4 S1 not sampled into tune/val. They are used ONLY as cross-S1
competitors (never trained on or evaluated). Output: work/stack/<role>/part_XXX.parquet
with s1_row, t_row, s2, cos_name, cos_addr, x_expand, y.
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
from business_entity_resolution.data import assign_folds, load_ground_truth
from business_entity_resolution.expansion import KEY
from business_entity_resolution.pipeline import key_freqs, make_roles, prepared
from business_entity_resolution.stack import StackModels, score_batch


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["val", "rest"], required=True)
    ap.add_argument("--batch", type=int, default=150_000)
    a = ap.parse_args()
    t0 = time.time()
    wd = work_dir()
    out = wd / "stack" / a.role
    out.mkdir(parents=True, exist_ok=True)
    s1, tg = prepared("train")
    s1f, tf = key_freqs(s1, tg)
    gt = load_ground_truth("train")
    gs = pd.Index(s1["entity_id"]).get_indexer(gt["s1_id"])
    gkey = np.sort(KEY(gs, pd.Index(tg["entity_id"]).get_indexer(gt["t_id"])))
    del gt
    vcfg = ValidationConfig()
    nm = np.bincount(gs, minlength=len(s1))
    roles = make_roles(s1, nm, vcfg)
    if a.role == "val":
        rows = np.sort(roles["val"])
    else:
        fold = assign_folds(s1, nm, vcfg.n_folds)
        rows = np.setdiff1d(np.flatnonzero(np.isin(fold, (vcfg.tune_fold, vcfg.val_fold))),
                            np.concatenate([roles["tune"], roles["val"]]))
    M = StackModels.load(wd)
    blocker = Blocker(tg, BlockingConfig(hybrid=True))
    n = 0
    for i, lo in enumerate(range(0, len(rows), a.batch)):
        e = score_batch(blocker, s1, tg, s1f, tf, rows[lo:lo + a.batch], M)
        e["y"] = np.isin(KEY(e["s1_row"], e["t_row"]), gkey).astype(np.int8)
        e[["s1_row", "t_row", "s2", "cos_name", "cos_addr", "x_expand", "y"]] \
            .to_parquet(out / f"part_{i:03d}.parquet", index=False)
        n += len(e)
        logging.info("batch %d: %d S1 -> %d pairs (%.0fs)", i, len(rows[lo:lo + a.batch]), len(e), time.time() - t0)
    meta = {"role": a.role, "n_s1": int(len(rows)), "pairs": n, "runtime_s": time.time() - t0,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    logging.info("done %s", meta)


if __name__ == "__main__":
    main()
