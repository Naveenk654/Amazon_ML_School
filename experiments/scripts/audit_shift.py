"""Phase 1 (competition mode): train/held-out -> test distribution audit of the frozen M3 stack.

    PYTHONPATH=src python experiments/scripts/audit_shift.py --pop val|rest|test   # per-pair dump
    PYTHONPATH=src python experiments/scripts/audit_shift.py --summarize            # comparison tables

Uses the stored phase-A outputs of the exact production path (heldout/<role>/
and test_stage/: P_*.parquet all scored pairs, XS3_*.parquet stage-3 inputs).
Stage 3 is recomputed with the frozen models over the correct competitor
population (whole training split for val/rest, whole test split for test).
No labels are used for test; no model is changed.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging

import numpy as np
import pandas as pd

from business_entity_resolution.competition import competition_features
from business_entity_resolution.config import models_dir, work_dir
from business_entity_resolution.stack import StackModels

P_COLS = ["s1_row", "t_row", "s2", "cos_name", "cos_addr"]
X_KEEP = ["p1", "name_tset", "name_ratio", "addr_tset", "num_jacc", "t_indic", "t_addr_empty",
          "name_exact", "name_tfidf", "addr_tfidf", "cos_name", "cos_addr"]


def dump(pop: str) -> None:
    wd = work_dir()
    M = StackModels.load_dir(models_dir())
    d = wd / ("test_stage" if pop == "test" else f"heldout/{pop}")
    own = [pd.read_parquet(f) for f in sorted(glob.glob(str(d / "P_*.parquet")))]
    offs = np.r_[0, np.cumsum([len(x) for x in own])]
    others = []
    if pop != "test":  # competitor population = whole training split (as in eval_m3_heldout.py)
        for role in ("train", "tune", "val"):
            if role == pop:
                continue
            for cf in sorted(glob.glob(str(wd / "matcher_data" / "CE" / role / "cand_*.parquet"))):
                c = pd.read_parquet(cf, columns=["s1_row", "t_row", "cos_name", "cos_addr"])
                c["s2"] = pd.read_parquet(cf.replace("cand_", "s2_"))["seed"].to_numpy()
                others.append(c[P_COLS])
        if pop != "rest":
            others += [pd.read_parquet(f, columns=P_COLS) for f in sorted(glob.glob(str(wd / "stack" / "rest" / "part_*.parquet")))]
    P = pd.concat(own + others, ignore_index=True)
    del others
    split = "test" if pop == "test" else "train"
    key_id = pd.factorize(pd.read_parquet(wd / f"{split}_s1_norm.parquet", columns=["name_key"])["name_key"])[0]
    Q = competition_features(P["s1_row"].to_numpy(), P["t_row"].to_numpy(), P["s2"].to_numpy(),
                             P["cos_name"].to_numpy(), P["cos_addr"].to_numpy(), key_id)
    n = offs[-1]
    out = P.iloc[:n].reset_index(drop=True)
    Qn = Q.iloc[:n].reset_index(drop=True)
    for c in ("comp_n", "comp_n_strong", "comp_best", "comp_margin", "comp_rank", "comp_same_key"):
        out[c] = Qn[c].to_numpy()
    final = out["s2"].to_numpy().astype(np.float64).copy()
    extra = {c: np.full(n, np.nan, np.float32) for c in X_KEEP if c not in ("cos_name", "cos_addr")}
    feats = M.m3.feature_name()
    for i in range(len(own)):
        x = pd.read_parquet(d / f"XS3_{i:03d}.parquet")
        rows = x["row"].to_numpy() + offs[i]
        F = pd.concat([x.drop(columns="row").reset_index(drop=True), Q.iloc[rows].reset_index(drop=True)], axis=1)
        final[rows] = M.m3.predict(F[feats])
        for c in extra:
            extra[c][rows] = x[c].to_numpy()
    out["m3"] = final.astype(np.float32)
    for c, v in extra.items():
        out[c] = v
    od = wd / "shift"
    od.mkdir(exist_ok=True)
    out.to_parquet(od / f"{pop}.parquet", index=False)
    logging.info("dumped %s: %d pairs", pop, n)


def summarize() -> None:
    wd = work_dir()
    M = StackModels.load_dir(models_dir())
    thr3 = M.m3_thr
    rep = {}
    for pop in ("val", "rest", "test"):
        split = "test" if pop == "test" else "train"
        D = pd.read_parquet(wd / "shift" / f"{pop}.parquet")
        s1 = pd.read_parquet(wd / f"{split}_s1_norm.parquet", columns=["country", "name_core", "business_name"]
                             if False else ["country", "name_core"])
        tg = pd.read_parquet(wd / f"{split}_t_norm.parquet", columns=["country", "name_core", "name_indic", "addr_empty"])
        kc = tg.groupby(["country", "name_core"]).size()
        D["country"] = s1["country"].to_numpy()[D["s1_row"].to_numpy()]
        D["t_indic_all"] = tg["name_indic"].to_numpy()[D["t_row"].to_numpy()]
        D["t_empty_all"] = tg["addr_empty"].to_numpy()[D["t_row"].to_numpy()]
        D["pred"] = D["m3"] >= thr3
        ids = np.unique(D["s1_row"].to_numpy())
        # S1 universe for this population (all S1 of the role; every one has >= 1 candidate here)
        cty_s1 = pd.Series(s1["country"].to_numpy()[ids], index=ids)
        fr = kc.reindex(pd.MultiIndex.from_arrays([s1["country"].to_numpy()[ids], s1["name_core"].to_numpy()[ids]])).fillna(0).to_numpy()
        common = pd.Series(fr > 50, index=ids)
        rep[pop] = {}
        for c in ["ALL", *sorted(D["country"].unique())]:
            m = np.ones(len(D), bool) if c == "ALL" else (D["country"].to_numpy() == c)
            d = D[m]
            sid = ids if c == "ALL" else ids[cty_s1.to_numpy() == c]
            g = d.groupby("s1_row")
            npred = g["pred"].sum().reindex(sid, fill_value=0)
            nstrong = (d["s2"] >= 0.5).groupby(d["s1_row"]).sum().reindex(sid, fill_value=0)
            casc = d["s2"] >= 0.01
            pr = d[d["pred"]]
            tcount = pr.groupby("t_row").size()
            qs = lambda s: [round(float(x), 4) for x in np.nanpercentile(s, [10, 50, 90])] if len(s) else None
            r = {
                "n_s1": int(len(sid)),
                "cands_per_s1": round(len(d) / len(sid), 3),
                "cascade_rate(s2>=0.01)": round(float(casc.mean()), 4),
                "expansion_share_of_cascade_rows": round(float(((d["name_exact"] == 0) & (d["name_tfidf"] == 0) & (d["addr_tfidf"] == 0))[casc].mean()), 4),
                "strong_cands_per_s1(s2>=0.5)": round(float(nstrong.mean()), 3),
                "p1_q10_50_90(cascade)": qs(d.loc[casc, "p1"]),
                "s2_share>=0.5": round(float((d["s2"] >= 0.5).mean()), 4),
                "s2_share_0.3-0.7": round(float(((d["s2"] >= 0.3) & (d["s2"] < 0.7)).mean()), 4),
                "m3_share_0.4-0.8(cascade)": round(float(((d["m3"] >= 0.4) & (d["m3"] < 0.8))[casc].mean()), 4),
                "pred_per_s1": round(float(npred.mean()), 3),
                "empty_pred_rate": round(float((npred == 0).mean()), 4),
                "targets_pred_for_>1_S1": int((tcount > 1).sum()),
                "comp_n_q(cascade)": qs(d.loc[casc, "comp_n"]),
                "comp_best_q(predicted)": qs(pr["comp_best"]),
                "comp_margin_q(predicted)": qs(pr["comp_margin"]),
                "comp_rank>0_share(predicted)": round(float((pr["comp_rank"] > 0).mean()), 4),
                "comp_n_strong>0_share(predicted)": round(float((pr["comp_n_strong"] > 0).mean()), 4),
                "comp_same_key_share(predicted)": round(float(pr["comp_same_key"].mean()), 4),
                "pred_name_tset_q": qs(pr["name_tset"]),
                "pred_addr_tset_q": qs(pr["addr_tset"]),
                "pred_num_jacc_q": qs(pr["num_jacc"]),
                "pred_t_indic_share": round(float(pr["t_indic_all"].mean()), 4),
                "pred_t_empty_addr_share": round(float(pr["t_empty_all"].mean()), 4),
                "cand_t_indic_share": round(float(d["t_indic_all"].mean()), 4),
                "cand_t_empty_addr_share": round(float(d["t_empty_all"].mean()), 4),
                "top1_cos_name_q": qs(d.groupby("s1_row")["cos_name"].max()),
                "top1_cos_addr_q": qs(d.groupby("s1_row")["cos_addr"].max()),
                "common_name_s1_share": round(float(common.reindex(sid).mean()), 4),
            }
            rep[pop][c] = r
    (wd / "shift" / "summary.json").write_text(json.dumps(rep, indent=1))
    keys = list(rep["val"]["ALL"])
    cols = [(p, c) for p in rep for c in rep[p]]
    print("metric".ljust(38) + "".join(f"{p}:{c}"[:18].rjust(19) for p, c in cols))
    for k in keys:
        print(k[:38].ljust(38) + "".join(str(rep[p][c].get(k))[:18].rjust(19) for p, c in cols))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", choices=["val", "rest", "test"])
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else dump(a.pop)


if __name__ == "__main__":
    main()
