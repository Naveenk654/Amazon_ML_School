"""Build the expanded (C + neighbour expansion) matcher dataset for one role.

    PYTHONPATH=src python experiments/scripts/build_expanded.py --role train|tune|val

Seeds come from seed_XXX.parquet (see seed_scores.py): out-of-fold M2 for
train, frozen M2 for tune/val. Seeds are pairs with seed >= the M2 threshold.
Writes matcher_data/CE<variant>/<role>/cand_XXX.parquet + X_XXX.parquet (features
recomputed over the enlarged candidate set, as at inference). With BER_VARIANT=X
(M5-EMB) the top-k embedding neighbours are added after expansion, as in stack.score_batch.
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

from business_entity_resolution.blocking import Blocker
from business_entity_resolution.config import BlockingConfig, variant, work_dir
from business_entity_resolution.data import load_ground_truth
from business_entity_resolution.emb import EmbKnn, add_emb
from business_entity_resolution.expansion import KEY, expand
from business_entity_resolution.features import build_features
from business_entity_resolution.pipeline import key_freqs, prepared
from business_entity_resolution.stack import EMB_K


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["train", "tune", "val"], required=True)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    wd = work_dir()
    thr = json.loads((wd / "exp" / "M2.json").read_text())["val"]["threshold"]
    src, dst = wd / "matcher_data" / "C" / a.role, wd / "matcher_data" / f"CE{variant()}" / a.role
    emb_k = EMB_K.get(variant(), 0)
    dst.mkdir(parents=True, exist_ok=True)
    s1, tg = prepared("train")
    s1f, tf = key_freqs(s1, tg)
    gt = load_ground_truth("train")
    gkey = np.sort(KEY(pd.Index(s1["entity_id"]).get_indexer(gt["s1_id"]),
                       pd.Index(tg["entity_id"]).get_indexer(gt["t_id"])))
    del gt
    blocker = Blocker(tg, BlockingConfig(hybrid=True))
    stats = {"new_pairs": 0, "new_true": 0, "c_pairs": 0, "emb_k": emb_k, "emb_pairs": 0, "emb_true": 0}
    for cf in sorted(glob.glob(f"{src}/cand_*.parquet")):
        c = pd.read_parquet(cf)
        seed = pd.read_parquet(cf.replace("cand_", "seed_"))["seed"].to_numpy()
        seeds = c.loc[seed >= thr, ["s1_row", "t_row"]]
        new = expand(blocker, s1, tg, seeds, np.sort(KEY(c["s1_row"], c["t_row"])), a.k)
        new["y"] = np.isin(KEY(new["s1_row"], new["t_row"]), gkey).astype(np.int8)
        stats["c_pairs"] += len(c)
        stats["new_pairs"] += len(new)
        stats["new_true"] += int(new["y"].sum())
        e = pd.concat([c, new[c.columns]], ignore_index=True)
        e["x_expand"] = np.r_[np.zeros(len(c), bool), np.ones(len(new), bool)]
        e = e.sort_values(["s1_row", "t_row"], kind="stable").reset_index(drop=True)
        if emb_k:
            rows = np.unique(c["s1_row"].to_numpy())
            e = add_emb(blocker, s1, tg, e, EmbKnn("train", rows), rows, emb_k)
            e["y"] = np.isin(KEY(e["s1_row"], e["t_row"]), gkey).astype(np.int8)
            stats["emb_pairs"] += int(e["x_emb"].sum())
            stats["emb_true"] += int(e.loc[e["x_emb"], "y"].sum())
        X = build_features(e, s1, tg, s1f, tf)
        name = cf.rsplit("/", 1)[1]
        e.to_parquet(dst / name, index=False)
        X.to_parquet(dst / name.replace("cand_", "X_"), index=False)
        logging.info("%s: C %d + new %d (true %d) (%.0fs)", name, len(c), len(new), int(new["y"].sum()), time.time() - t0)
    stats.update(role=a.role, runtime_s=time.time() - t0,
                 peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    (dst / "meta.json").write_text(json.dumps(stats, indent=1))
    logging.info("done %s", stats)


if __name__ == "__main__":
    main()
