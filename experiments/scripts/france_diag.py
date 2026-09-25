"""Experiment A: France failure-mechanism diagnostic on frozen-M3 test predictions (no labels).

    PYTHONPATH=src python experiments/scripts/france_diag.py

Input: work/shift/test.parquet (audit_shift.py dump of the exact production M3 path).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from business_entity_resolution.config import models_dir, work_dir
from business_entity_resolution.stack import StackModels

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)


def city_of(addr: pd.Series) -> pd.Series:
    """Second-to-last comma part for FR/US S1 ('..., City, Region|ST'); last-but-one otherwise."""
    parts = addr.str.split(",")
    return parts.str[-2].str.strip().str.lower().fillna("")


def main() -> None:
    wd = work_dir()
    M = StackModels.load_dir(models_dir())
    cols = ["s1_row", "t_row", "s2", "m3", "comp_n_strong", "comp_same_key", "name_tset", "addr_tset", "num_jacc"]
    D = pd.read_parquet(wd / "shift" / "test.parquet", columns=cols)
    D = D[D["s2"].to_numpy() >= M.m3_cut].reset_index(drop=True)  # stage-3 cascade holds every prediction
    s1 = pd.read_parquet(wd / "test_s1_norm.parquet",
                         columns=["business_name", "business_address", "country", "name_core", "name_key", "addr_nums"])
    tg = pd.read_parquet(wd / "test_t_norm.parquet",
                         columns=["business_name", "business_address", "country", "name_core", "addr_nums", "addr_empty"])
    D["country"] = s1["country"].to_numpy()[D["s1_row"].to_numpy()]
    D["pred"] = D["m3"] >= M.m3_thr
    D["pred_s2"] = D["s2"] >= M.m2e_thr

    print("== 1. S1 name templates / collisions (within country)")
    s1["city"] = city_of(s1["business_address"])
    for c, g in s1.groupby("country"):
        key_n = g.groupby("name_key").size()
        kc = g.groupby(["name_key", "city"]).size()
        dup_key = g["name_key"].map(key_n) > 1
        dup_key_city = pd.Series(list(zip(g["name_key"], g["city"])), index=g.index).map(kc) > 1
        vocab = pd.Series(" ".join(g["name_core"]).split()).value_counts()
        print(f"  {c:7s} S1 {len(g):7d} | share sharing name_key {dup_key.mean():.3f} | sharing name_key AND city "
              f"{dup_key_city.mean():.3f} | distinct cities {g['city'].nunique():6d} | name vocab {len(vocab):6d} "
              f"| tokens/name {g['name_core'].str.split().str.len().mean():.2f}")
    fr = s1[s1["country"] == "France"]
    print("  France top cities:", fr["city"].value_counts().head(8).to_dict())
    print("  France most repeated core names:", fr["name_core"].value_counts().head(8).to_dict())

    print("\n== 2. Prediction-level profile by country")
    rows = []
    for c, d in D.groupby("country"):
        p = d[d["pred"]]
        a = s1.iloc[p["s1_row"].to_numpy()]
        t = tg.iloc[p["t_row"].to_numpy()]
        nj = p["num_jacc"].to_numpy()
        rows.append({
            "country": c,
            "pred_per_s1": round(len(p) / int((s1["country"] == c).sum()), 3),
            "M3_accepts_s2<0.65 (competition-driven accept)": round(float((p["s2"] < M.m2e_thr).mean()), 4),
            "M3_rejects_s2>=0.65 (competition veto) per 100 preds": round(float(((~d["pred"]) & d["pred_s2"]).sum() / len(p) * 100), 3),
            "pred: name_tset>=95 & num_jacc==0": round(float(((p["name_tset"] >= 95) & (nj == 0)).mean()), 4),
            "pred: addr_tset<80": round(float((p["addr_tset"] < 80).mean()), 4),
            "pred: num_jacc<1 (numbers differ)": round(float((nj < 1).mean()), 4),
            "pred: name_tset<70": round(float((p["name_tset"] < 70).mean()), 4),
            "pred: target addr empty": round(float(t["addr_empty"].mean()), 4),
            "pred: has same-key competitor": round(float(p["comp_same_key"].mean()), 4),
            "pred: comp_n_strong>0": round(float((p["comp_n_strong"] > 0).mean()), 4),
            "pred: m3 in [0.6,0.8)": round(float((p["m3"] < 0.8).mean()), 4),
            "s1 with >=7 preds": round(float((p.groupby("s1_row").size() >= 7).mean()), 4),
        })
    print(pd.DataFrame(rows).set_index("country").T.to_string())

    print("\n== 3. Targets predicted for >1 S1 (France)")
    pf = D[(D["country"] == "France") & D["pred"]]
    multi = pf.groupby("t_row")["s1_row"].agg(list)
    multi = multi[multi.map(len) > 1]
    print("  count:", len(multi))
    for t, ss in list(multi.items())[:4]:
        print(f"  T: {tg['business_name'].iat[t]} | {tg['business_address'].iat[t]}")
        for s in ss:
            m3 = pf[(pf["s1_row"] == s) & (pf["t_row"] == t)]["m3"].iat[0]
            print(f"     S1 ({m3:.2f}): {s1['business_name'].iat[s]} | {s1['business_address'].iat[s]}")

    print("\n== 4. France risky predictions: near-identical name, NO shared address number (sample)")
    r = pf[(pf["name_tset"] >= 95) & (pf["num_jacc"] == 0)].sample(8, random_state=0)
    for _, x in r.iterrows():
        print(f"  ({x.m3:.2f} s2 {x.s2:.2f}) {s1['business_name'].iat[x.s1_row]} | {s1['business_address'].iat[x.s1_row]}")
        print(f"        -> {tg['business_name'].iat[x.t_row]} | {tg['business_address'].iat[x.t_row] or '<EMPTY>'}")

    print("\n== 5. France S1 with many predictions (sample)")
    many = pf.groupby("s1_row").size()
    s = many[many >= 8].index[:2]
    for sr in s:
        print(f"  S1: {s1['business_name'].iat[sr]} | {s1['business_address'].iat[sr]}  ({many[sr]} preds)")
        for t in pf[pf["s1_row"] == sr]["t_row"].to_numpy()[:10]:
            print(f"     -> {tg['business_name'].iat[t]} | {tg['business_address'].iat[t] or '<EMPTY>'}")

    print("\n== 6. France address formats (S1 vs targets)")
    ft = tg[tg["country"] == "France"]
    print("  S1 last part top:", fr["business_address"].str.split(",").str[-1].str.strip().value_counts().head(5).to_dict())
    print("  target last part top:", ft["business_address"].str.split(",").str[-1].str.strip().value_counts().head(8).to_dict())
    print("  France S1 with number in address:", fr["addr_nums"].ne("").mean().round(4),
          " targets:", ft["addr_nums"].ne("").mean().round(4))


if __name__ == "__main__":
    main()
