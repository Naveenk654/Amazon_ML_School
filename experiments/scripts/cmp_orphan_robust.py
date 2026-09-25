"""Experiment B comparison: M3 vs M3OR on validation, normal and with 19% competitor dropout.

    PYTHONPATH=src python experiments/scripts/cmp_orphan_robust.py
Paired bootstrap per S1 + breakdowns (country, common-name, empty-address targets,
competition-heavy S1). Same dropout draw as sim_orphans.py (seed SEED+7).
"""
from __future__ import annotations

import glob
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from business_entity_resolution import evaluate as ev
from business_entity_resolution.competition import competition_features
from business_entity_resolution.config import SEED, models_dir, work_dir
from business_entity_resolution.stack import StackModels

P_COLS = ["s1_row", "t_row", "s2", "cos_name", "cos_addr"]


def main() -> None:
    wd = work_dir()
    base = StackModels.load_dir(models_dir())
    info = json.loads((wd / "exp" / "M3OR.json").read_text())
    models = {"M3": (base.m3, base.m3_thr),
              "M3OR": (lgb.Booster(model_file=str(wd / "models" / "M3OR.txt")), info["val"]["threshold"])}
    d = wd / "heldout" / "val"
    own = [pd.read_parquet(f) for f in sorted(glob.glob(str(d / "P_*.parquet")))]
    offs = np.r_[0, np.cumsum([len(x) for x in own])]
    Pown = pd.concat(own, ignore_index=True)
    XS3 = pd.concat([pd.read_parquet(d / f"XS3_{i:03d}.parquet").assign(
        row=lambda x, i=i: x["row"].to_numpy() + offs[i]) for i in range(len(own))], ignore_index=True)
    others = []
    for role in ("train", "tune"):
        for cf in sorted(glob.glob(str(wd / "matcher_data" / "CE" / role / "cand_*.parquet"))):
            c = pd.read_parquet(cf, columns=["s1_row", "t_row", "cos_name", "cos_addr"])
            c["s2"] = pd.read_parquet(cf.replace("cand_", "s2_"))["seed"].to_numpy()
            others.append(c[P_COLS])
    others += [pd.read_parquet(f, columns=P_COLS) for f in sorted(glob.glob(str(wd / "stack" / "rest" / "part_*.parquet")))]
    O = pd.concat(others, ignore_index=True)
    del others
    s1 = pd.read_parquet(wd / "train_s1_norm.parquet", columns=["country", "name_core", "name_key"])
    tg = pd.read_parquet(wd / "train_t_norm.parquet", columns=["country", "name_core", "addr_empty"])
    key_id = pd.factorize(s1["name_key"])[0]
    gv = pd.read_parquet(wd / "audit" / "C_val_gt.parquet")
    truth = ev.as_sets(gv, "s1_row", "t_row")
    gk = np.sort(gv["s1_row"].to_numpy().astype(np.int64) << 32 | gv["t_row"].to_numpy())
    ids = np.load(wd / "audit" / "val_rows.npy")
    cty = s1["country"].to_numpy()[ids]
    kc = tg.groupby(["country", "name_core"]).size()
    common = kc.reindex(pd.MultiIndex.from_arrays([cty, s1["name_core"].to_numpy()[ids]])).fillna(0).to_numpy() > 50
    sg = np.array([s not in truth for s in ids])
    comp_s1 = np.unique(O["s1_row"].to_numpy())
    rows = XS3["row"].to_numpy()
    res, per = {}, {}
    for drop in (0.0, 0.19):
        gone = comp_s1[np.random.default_rng(SEED + 7).random(len(comp_s1)) < drop]
        P = pd.concat([Pown, O[~np.isin(O["s1_row"].to_numpy(), gone)]], ignore_index=True)
        Q = competition_features(P["s1_row"].to_numpy(), P["t_row"].to_numpy(), P["s2"].to_numpy(),
                                 P["cos_name"].to_numpy(), P["cos_addr"].to_numpy(), key_id)
        Qr = Q.iloc[rows].reset_index(drop=True)
        heavy_s1 = np.unique(Pown["s1_row"].to_numpy()[rows][(Qr["comp_n_strong"].to_numpy() > 0)
                                                               | (Qr["comp_same_key"].to_numpy() > 0)])
        heavy = np.isin(ids, heavy_s1)
        F = pd.concat([XS3.drop(columns="row").reset_index(drop=True), Qr], axis=1)
        del P, Q
        for name, (m, thr) in models.items():
            sc = Pown["s2"].to_numpy().astype(np.float64).copy()
            sc[rows] = m.predict(F[m.feature_name()])
            p = sc >= thr
            pr = Pown.loc[p, ["s1_row", "t_row"]]
            pred = ev.as_sets(pr, "s1_row", "t_row")
            f = np.array([ev.f05(pred.get(s, set()), truth.get(s, set())) for s in ids])
            k = pr["s1_row"].to_numpy().astype(np.int64) << 32 | pr["t_row"].to_numpy()
            y = np.isin(k, gk)
            emp_t = tg["addr_empty"].to_numpy()[gv["t_row"].to_numpy()]
            emp_hit = np.isin(gv["s1_row"].to_numpy().astype(np.int64) << 32 | gv["t_row"].to_numpy(), k)
            per[(drop, name)] = f
            res[f"drop={drop}|{name}"] = {
                "macro_f05": round(float(f.mean()), 5), "precision": round(float(y.mean()), 5),
                "recall": round(float(y.sum() / len(gv)), 5), "FP": int((~y).sum()), "FN": int(len(gv) - y.sum()),
                "singleton_f05": round(float(f[sg].mean()), 5),
                "India": round(float(f[cty == "India"].mean()), 5), "US": round(float(f[cty == "US"].mean()), 5),
                "common_name_S1": round(float(f[common].mean()), 5),
                "competition_heavy_S1": round(float(f[heavy].mean()), 5), "competition_heavy_n": int(heavy.sum()),
                "empty_addr_target_recall": round(float(emp_hit[emp_t].mean()), 5)}
            print(f"drop={drop} {name}", json.dumps(res[f"drop={drop}|{name}"]), flush=True)
        dlt = per[(drop, "M3OR")] - per[(drop, "M3")]
        bs = dlt[np.random.default_rng(0).integers(0, len(dlt), (1000, len(dlt)))].mean(1)
        res[f"drop={drop}|delta"] = {"mean": round(float(dlt.mean()), 5),
                                     "ci95": [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]}
        print(f"drop={drop} delta M3OR-M3", res[f"drop={drop}|delta"], flush=True)
    (wd / "exp" / "M3OR_compare.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
