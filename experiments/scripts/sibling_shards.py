"""Build M2 sibling-feature shards (S_XXX.parquet) from stage-1 scores (p1_XXX.parquet).

    PYTHONPATH=src python experiments/scripts/sibling_shards.py --blocking C
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import resource
import time
import warnings

import pandas as pd

from business_entity_resolution.config import work_dir
from business_entity_resolution.siblings import sibling_features


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN nanmax rows
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocking", default="C")
    a = ap.parse_args()
    t0 = time.time()
    wd = work_dir()
    tg = pd.read_parquet(wd / "train_t_norm.parquet", columns=["name_core", "addr_norm", "addr_nums", "src"])
    base = wd / "matcher_data" / a.blocking
    n = 0
    for role in ("tune", "val", "train"):
        for cf in sorted(glob.glob(f"{base}/{role}/cand_*.parquet")):
            c = pd.read_parquet(cf, columns=["s1_row", "t_row"])
            c["p1"] = pd.read_parquet(cf.replace("cand_", "p1_"))["p1"].to_numpy()
            sibling_features(c, tg).to_parquet(cf.replace("cand_", "S_"), index=False)
            n += len(c)
            logging.info("%s %s rows=%d (%.0fs)", role, cf.rsplit("/", 1)[1], len(c), time.time() - t0)
    meta = {"rows": n, "runtime_s": time.time() - t0,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    (base / "sibling_meta.json").write_text(json.dumps(meta, indent=1))
    logging.info("done %s", meta)


if __name__ == "__main__":
    main()
