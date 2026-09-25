"""Phase 1 diagnostic: does a test-like ORPHAN-target density explain the val -> test gap?

    PYTHONPATH=src python experiments/scripts/sim_orphans.py --drop 0.19

Test has 5.76 targets per S1 vs 4.67 in train (+23% in US and India alike),
consistent with ~19% of the S1 that own test targets being absent (their
targets become ownerless "orphans"). Simulation: randomly remove a fraction of
the COMPETITOR S1 (never the evaluated val S1) from the scored population;
their targets stay in the candidate pools (as in test) but lose their owner's
competing score. Recompute competition features and the frozen M3 only.
No model is changed or retrained.
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from business_entity_resolution import evaluate as ev
from business_entity_resolution.competition import competition_features
from business_entity_resolution.config import SEED, models_dir, work_dir
from business_entity_resolution.stack import StackModels

P_COLS = ["s1_row", "t_row", "s2", "cos_name", "cos_addr"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", type=float, nargs="+", default=[0.0, 0.19, 0.30])
    a = ap.parse_args()
    wd = work_dir()
    M = StackModels.load_dir(models_dir())
    d = wd / "heldout" / "val"
    own = [pd.read_parquet(f) for f in sorted(glob.glob(str(d / "P_*.parquet")))]
    offs = np.r_[0, np.cumsum([len(x) for x in own])]
    others = []
    for role in ("train", "tune"):
        for cf in sorted(glob.glob(str(wd / "matcher_data" / "CE" / role / "cand_*.parquet"))):
            c = pd.read_parquet(cf, columns=["s1_row", "t_row", "cos_name", "cos_addr"])
            c["s2"] = pd.read_parquet(cf.replace("cand_", "s2_"))["seed"].to_numpy()
            others.append(c[P_COLS])
    others += [pd.read_parquet(f, columns=P_COLS) for f in sorted(glob.glob(str(wd / "stack" / "rest" / "part_*.parquet")))]
    O = pd.concat(others, ignore_index=True)
    del others
    Pown = pd.concat(own, ignore_index=True)
    XS3 = []
    for i in range(len(own)):
        x = pd.read_parquet(d / f"XS3_{i:03d}.parquet")
        x["row"] = x["row"].to_numpy() + offs[i]
        XS3.append(x)
    XS3 = pd.concat(XS3, ignore_index=True)
    key_id = pd.factorize(pd.read_parquet(wd / "train_s1_norm.parquet", columns=["name_key"])["name_key"])[0]
    country = pd.read_parquet(wd / "train_s1_norm.parquet", columns=["country"])["country"].to_numpy()
    gv = pd.read_parquet(wd / "audit" / "C_val_gt.parquet")
    truth = ev.as_sets(gv, "s1_row", "t_row")
    ids = np.load(wd / "audit" / "val_rows.npy")
    ga = pd.read_parquet(wd / "audit" / "all_gt_rows.parquet")
    owner = pd.Series(ga["s1_row"].to_numpy(), index=ga["t_row"].to_numpy())
    comp_s1 = np.unique(O["s1_row"].to_numpy())
    feats = M.m3.feature_name()
    res = {}
    for f in a.drop:
        rng = np.random.default_rng(SEED + 7)
        gone = comp_s1[rng.random(len(comp_s1)) < f]
        keep = ~np.isin(O["s1_row"].to_numpy(), gone)
        P = pd.concat([Pown, O[keep]], ignore_index=True)
        Q = competition_features(P["s1_row"].to_numpy(), P["t_row"].to_numpy(), P["s2"].to_numpy(),
                                 P["cos_name"].to_numpy(), P["cos_addr"].to_numpy(), key_id)
        n = offs[-1]
        final = Pown["s2"].to_numpy().astype(np.float64).copy()
        rows = XS3["row"].to_numpy()
        F = pd.concat([XS3.drop(columns="row").reset_index(drop=True), Q.iloc[rows].reset_index(drop=True)], axis=1)
        final[rows] = M.m3.predict(F[feats])
        out = {}
        for name, sc, thr in (("M2E_level", Pown["s2"].to_numpy(), M.m2e_thr), ("M3", final, M.m3_thr)):
            p = sc >= thr
            pr = Pown.loc[p, ["s1_row", "t_row"]]
            pred = ev.as_sets(pr, "s1_row", "t_row")
            per = np.array([ev.f05(pred.get(s, set()), truth.get(s, set())) for s in ids])
            k = pr["s1_row"].to_numpy().astype(np.int64) << 32 | pr["t_row"].to_numpy()
            y = np.isin(k, gv["s1_row"].to_numpy().astype(np.int64) << 32 | gv["t_row"].to_numpy())
            own_t = owner.reindex(pr["t_row"].to_numpy()).to_numpy()
            orphan_fp = (~y) & np.isin(own_t, gone)
            cty = country[ids]
            out[name] = {"macro_f05": round(float(per.mean()), 5), "FP": int((~y).sum()),
                         "FP_on_orphaned_targets": int(orphan_fp.sum()),
                         "India": round(float(per[cty == "India"].mean()), 5), "US": round(float(per[cty == "US"].mean()), 5)}
        res[f"drop={f}"] = {"removed_competitor_S1": int(len(gone)), **out}
        print(f, json.dumps(res[f"drop={f}"]), flush=True)
    (wd / "shift" / "orphan_sim.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
