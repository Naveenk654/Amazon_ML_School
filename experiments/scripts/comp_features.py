"""M3: cross-S1 competition features (inference-time scores only, no labels).

    PYTHONPATH=src python experiments/scripts/comp_features.py

Population = every scored S1: train (out-of-fold M2-E scores, s2_XXX), tune/val
(frozen M2-E, s2_XXX) and rest (work/stack/rest, competitors only). For each
candidate (S1, target), features describe the OTHER S1 that also hold this
target as a candidate. Writes Q_XXX.parquet next to each CE shard (train/tune/val).
"""
from __future__ import annotations

import glob
import json
import logging
import resource
import time

import numpy as np
import pandas as pd

from business_entity_resolution.config import variant, work_dir

STRONG = 0.5


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    t0 = time.time()
    wd = work_dir()
    base = wd / "matcher_data" / f"CE{variant()}"
    parts, owners = [], []  # owners: (role, shard file, n rows) for scattering back
    for role in ("train", "tune", "val"):
        for cf in sorted(glob.glob(f"{base}/{role}/cand_*.parquet")):
            c = pd.read_parquet(cf, columns=["s1_row", "t_row", "cos_name", "cos_addr"])
            c["s2"] = pd.read_parquet(cf.replace("cand_", "s2_"))["seed"].to_numpy()
            parts.append(c)
            owners.append((cf, len(c)))
    for pf in sorted(glob.glob(str(wd / "stack" / f"rest{variant()}" / "part_*.parquet"))):
        parts.append(pd.read_parquet(pf, columns=["s1_row", "t_row", "cos_name", "cos_addr", "s2"]))
        owners.append((None, len(parts[-1])))
    P = pd.concat(parts, ignore_index=True)
    del parts
    n = len(P)
    s1k = pd.read_parquet(wd / "train_s1_norm.parquet", columns=["name_key"])["name_key"]
    key_id = pd.factorize(s1k)[0]
    logging.info("population pairs %d (%.0fs)", n, time.time() - t0)

    t = P["t_row"].to_numpy()
    s = P["s2"].to_numpy().astype(np.float32)
    order = np.lexsort((-s, t))
    t_s, s_s = t[order], s[order]
    start = np.flatnonzero(np.r_[True, t_s[1:] != t_s[:-1]])
    size = np.diff(np.r_[start, n])
    g0, gsz = np.repeat(start, size), np.repeat(size, size)
    rank = np.arange(n) - g0                               # 0 = best S1 for this target
    top1, top2 = g0, np.where(gsz > 1, g0 + 1, -1)
    comp = np.where(rank == 0, top2, top1)                 # best OTHER S1 row (sorted pos)
    has = comp >= 0
    cs = np.where(has, s_s[np.maximum(comp, 0)], 0.0)
    gstrong = pd.Series((s_s >= STRONG).astype(np.int32)).groupby(g0).transform("sum").to_numpy()
    gsum = pd.Series(s_s).groupby(g0).transform("sum").to_numpy()
    s1o = P["s1_row"].to_numpy()[order]
    cn = P["cos_name"].to_numpy()[order]
    ca = P["cos_addr"].to_numpy()[order]
    ci = np.maximum(comp, 0)
    F = pd.DataFrame({
        "s2": s_s,
        "comp_n": (gsz - 1).astype(np.float32),
        "comp_n_strong": (gstrong - (s_s >= STRONG)).astype(np.float32),
        "comp_best": cs.astype(np.float32),
        "comp_margin": (s_s - cs).astype(np.float32),
        "comp_rank": rank.astype(np.float32),
        "comp_sum_other": (gsum - s_s).astype(np.float32),
        "comp_dcos_name": np.where(has, cn - cn[ci], np.nan).astype(np.float32),
        "comp_dcos_addr": np.where(has, ca - ca[ci], np.nan).astype(np.float32),
        "comp_same_key": np.where(has, key_id[s1o] == key_id[s1o[ci]], False).astype(np.float32),
    })
    inv = np.empty(n, np.int64)
    inv[order] = np.arange(n)
    F = F.iloc[inv].reset_index(drop=True)
    # S1-level: how many of this S1's strong candidates face a stronger competitor
    contested = ((F["comp_best"] > F["s2"]) & (F["s2"] >= STRONG)).astype(np.float32)
    F["s1_n_contested"] = contested.groupby(P["s1_row"].to_numpy()).transform("sum").to_numpy()
    lo = 0
    for cf, m in owners:
        if cf is not None:
            F.iloc[lo:lo + m].reset_index(drop=True).to_parquet(cf.replace("cand_", "Q_"), index=False)
        lo += m
    meta = {"population_pairs": n, "targets_with_competition": float((gsz > 1).mean()),
            "runtime_s": time.time() - t0,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    (base / "comp_meta.json").write_text(json.dumps(meta, indent=1))
    logging.info("done %s", meta)


if __name__ == "__main__":
    main()
