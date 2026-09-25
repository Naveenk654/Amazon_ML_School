"""End-to-end baseline.

    python -m business_entity_resolution.pipeline validate   # labeled S1-grouped validation
    python -m business_entity_resolution.pipeline predict    # test inference -> output/*.tsv

Stages: normalize -> blocking (3 passes) -> candidate union -> features ->
LightGBM matcher -> threshold decision (empty list = no-match) -> submission.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from . import evaluate as ev
from .blocking import PASSES, Blocker
from .config import BlockingConfig, ModelConfig, ValidationConfig, SEED, models_dir, out_dir, work_dir
from .data import assign_folds, load_ground_truth, load_sources
from .features import build_features
from .inference import decide, tune_threshold, write_lists_rows
from .model import predict as model_predict, train as model_train
from .normalize import normalize_frame

log = logging.getLogger("ber")
NORM_COLS = ["name_full", "name_core", "name_concat", "addr_norm", "addr_nums",
             "name_indic", "addr_empty", "name_key"]


def prepared(split: str):
    """Normalized (s1, targets), cached as parquet in the work dir."""
    p1, pt = work_dir() / f"{split}_s1_norm.parquet", work_dir() / f"{split}_t_norm.parquet"
    if p1.exists() and pt.exists():
        cols = ["entity_id", "country", *NORM_COLS]
        return pd.read_parquet(p1, columns=cols), pd.read_parquet(pt, columns=cols + ["src"])
    s1, tg = load_sources(split)
    t0 = time.time()
    s1, tg = normalize_frame(s1), normalize_frame(tg)
    log.info("normalized %s in %.0fs", split, time.time() - t0)
    s1.to_parquet(p1, index=False)
    tg.to_parquet(pt, index=False)
    return s1, tg


def key_freqs(s1: pd.DataFrame, tg: pd.DataFrame):
    """Name-key frequency per (country, key): unsupervised name specificity."""
    return (s1.groupby(["country", "name_key"]).size(),
            tg.groupby(["country", "name_key"]).size())


def generate(blocker: Blocker, s1: pd.DataFrame, tg: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    t0 = time.time()
    cand = blocker.candidates(s1.iloc[rows])
    cand["s1_row"] = rows[cand["s1_row"].to_numpy()]
    log.info("blocking: %d S1 -> %d candidates in %.0fs", len(rows), len(cand), time.time() - t0)
    return cand


def with_ids(cand: pd.DataFrame, s1: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Attach string IDs (only for the subsets that need them, to save memory)."""
    out = cand.copy()
    out["s1_id"] = s1["entity_id"].to_numpy()[out["s1_row"].to_numpy()]
    out["t_id"] = tg["entity_id"].to_numpy()[out["t_row"].to_numpy()]
    return out


def within_country_space(s1_sub: pd.DataFrame, tg: pd.DataFrame) -> tuple[float, float]:
    tc = tg["country"].value_counts()
    within = float(s1_sub["country"].map(tc).fillna(0).sum())
    return within, float(len(s1_sub)) * len(tg)


def make_roles(s1: pd.DataFrame, n_match: np.ndarray, vcfg: ValidationConfig) -> dict:
    """S1 row positions per role; deterministic so experiments reuse EXP001's split."""
    fold = assign_folds(s1, n_match, vcfg.n_folds)
    rng = np.random.default_rng(SEED)
    return {
        "train": rng.permutation(np.flatnonzero(np.isin(fold, vcfg.train_folds)))[: vcfg.n_train],
        "tune": rng.permutation(np.flatnonzero(fold == vcfg.tune_fold))[: vcfg.n_tune],
        "val": rng.permutation(np.flatnonzero(fold == vcfg.val_fold))[: vcfg.n_val],
    }


def cmd_validate(args) -> None:
    bcfg, vcfg, mcfg = BlockingConfig(), ValidationConfig(), ModelConfig()
    s1, tg = prepared("train")
    s1f, tf = key_freqs(s1, tg)
    gt = load_ground_truth("train")
    gt_s1 = pd.Index(s1["entity_id"]).get_indexer(gt["s1_id"])
    gt_t = pd.Index(tg["entity_id"]).get_indexer(gt["t_id"])
    assert (gt_s1 >= 0).all() and (gt_t >= 0).all()
    roles = make_roles(s1, np.bincount(gt_s1, minlength=len(s1)), vcfg)
    all_rows = np.sort(np.concatenate(list(roles.values())))
    in_roles = np.isin(gt_s1, all_rows)
    gt = gt[in_roles].reset_index(drop=True)
    gt_key = gt_s1[in_roles].astype(np.int64) << 32 | gt_t[in_roles]
    truth = ev.as_sets(gt)
    del gt_s1, gt_t

    t0 = time.time()
    blocker = Blocker(tg, bcfg)  # pool = ALL train S2+S3, as in the test setting
    log.info("indexes fitted in %.0fs", time.time() - t0)
    cand = generate(blocker, s1, tg, all_rows)
    del blocker
    ckey = cand["s1_row"].to_numpy().astype(np.int64) << 32 | cand["t_row"].to_numpy()
    cand["y"] = np.isin(ckey, gt_key).astype(np.int8)
    del ckey

    t0 = time.time()
    X = build_features(cand, s1, tg, s1f, tf)
    log.info("features: %s in %.0fs", X.shape, time.time() - t0)

    report = {"config": {"blocking": vars(bcfg), "validation": vars(vcfg),
                         "model_rounds": mcfg.num_boost_round}}
    role_code = np.full(len(s1), -1, np.int8)
    names = list(roles)
    for i, r in enumerate(names):
        role_code[roles[r]] = i
    crole = np.array(names + [""], dtype=object)[role_code[cand["s1_row"].to_numpy()]]

    # Blocking quality per role
    for r, rows in roles.items():
        ids = s1["entity_id"].to_numpy()[rows]
        tp = gt[gt["s1_id"].isin(set(ids))]
        within, full = within_country_space(s1.iloc[rows], tg)
        m = crole == r
        cr = with_ids(cand[m], s1, tg)
        rep = ev.blocking_report(cr, tp, ids, full, PASSES)
        rep["reduction_ratio_within_country"] = 1 - m.sum() / within
        rep["oracle_macro_f05"] = ev.oracle_f05(cr, truth, ids)
        del cr
        report[f"blocking_{r}"] = rep
        log.info("blocking[%s]: %s", r, json.dumps(rep, default=float))

    # Train matcher on train-role pairs only
    mtr = crole == "train"
    t0 = time.time()
    model = model_train(X[mtr], cand["y"].to_numpy()[mtr], mcfg)
    log.info("trained in %.0fs", time.time() - t0)
    cand["score"] = model_predict(model, X)

    ids_tune = s1["entity_id"].to_numpy()[roles["tune"]]
    ids_val = s1["entity_id"].to_numpy()[roles["val"]]
    thr, curve = tune_threshold(with_ids(cand[crole == "tune"], s1, tg), truth, ids_tune)
    report["threshold_curve_tune"] = curve.to_dict("list")
    report["threshold"] = thr

    pv = with_ids(cand[crole == "val"], s1, tg)
    tpv = gt[gt["s1_id"].isin(set(ids_val))]
    for name, t in (("tuned", thr), ("fixed_0.5", 0.5)):
        pred_pairs = decide(pv, t)
        pred = ev.as_sets(pred_pairs)
        res = {"threshold": t, "macro_f05": ev.macro_f05(pred, truth, ids_val),
               **ev.pair_metrics(pred_pairs, tpv),
               "pred_empty_rate": float(np.mean([s not in pred for s in ids_val])),
               "true_singleton_rate": float(np.mean([s not in truth for s in ids_val]))}
        report[f"val_{name}"] = res
        log.info("VAL[%s]: %s", name, json.dumps(res))
    pred = ev.as_sets(decide(pv, thr))
    bd = ev.breakdown(pred, truth, s1.iloc[roles["val"]])
    report["val_breakdown"] = bd.round(4).reset_index().to_dict("list")
    print(bd.round(4).to_string())

    imp = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    report["feature_gain"] = (imp / imp.sum()).sort_values(ascending=False).round(4).to_dict()

    # Error decomposition on val (why do we lose F0.5?)
    report["val_errors"] = error_decomposition(pv, tpv, pred, truth, ids_val)

    wd = work_dir()
    model.save_model(str(wd / "baseline_model.txt"))
    (wd / "baseline_threshold.json").write_text(json.dumps({"threshold": thr}))
    (wd / "baseline_validation.json").write_text(json.dumps(report, indent=1, default=float))
    pv[["s1_id", "t_id", "y", "score", *PASSES, "cos_name", "cos_addr"]] \
        .to_parquet(wd / "val_scored_pairs.parquet", index=False)
    log.info("report written to %s", wd / "baseline_validation.json")


def error_decomposition(pv, tpv, pred, truth, ids) -> dict:
    """Split lost F0.5 into blocking misses vs matcher errors."""
    cset = ev.as_sets(pv)
    lost = {"blocking_miss_pairs": 0, "matcher_fn_pairs": 0, "fp_pairs": 0,
            "fp_on_singletons_entities": 0, "fn_entities_all_missed": 0}
    for s in ids:
        t, p, c = truth.get(s, set()), pred.get(s, set()), cset.get(s, set())
        lost["blocking_miss_pairs"] += len(t - c)
        lost["matcher_fn_pairs"] += len((t & c) - p)
        lost["fp_pairs"] += len(p - t)
        if not t and p:
            lost["fp_on_singletons_entities"] += 1
        if t and not (p & t):
            lost["fn_entities_all_missed"] += 1
    lost["true_pairs"] = len(tpv)
    return lost


def cmd_predict(args) -> None:
    """Test inference with the frozen M3 stack (see stack.py).

    Phase A (this process): every test S1, in batches, through blocking ->
    M1b -> M2 -> neighbour expansion -> M1b-E -> M2-E; stores all scored pairs
    and the stage-3 input rows. Phase B (fresh process, for memory): cross-S1
    competition features over ALL pairs -> M3 -> submission files.
    """
    if args.phase in ("all", "A"):
        predict_phase_a(args)
    if args.phase == "all":
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "business_entity_resolution.pipeline", "predict", "--phase", "B"],
                       check=True)
    elif args.phase == "B":
        predict_phase_b(args)


P_COLS = ["s1_row", "t_row", "s2", "cos_name", "cos_addr"]


def _stage_dir():
    d = work_dir() / "test_stage"
    d.mkdir(exist_ok=True)
    return d


def predict_phase_a(args) -> None:
    import resource

    from .stack import StackModels, score_batch

    t0 = time.time()
    d = _stage_dir()
    for f in d.glob("*.parquet"):
        f.unlink()
    M = StackModels.load_dir(models_dir())
    s1, tg = prepared("test")
    s1f, tf = key_freqs(s1, tg)
    blocker = Blocker(tg, BlockingConfig(hybrid=True))
    rows_all = np.arange(len(s1))
    n_pairs = 0
    for i, lo in enumerate(range(0, len(s1), args.batch)):
        e, keep = score_batch(blocker, s1, tg, s1f, tf, rows_all[lo:lo + args.batch], M, stage3=True)
        e[P_COLS].to_parquet(d / f"P_{i:03d}.parquet", index=False)
        keep.to_parquet(d / f"XS3_{i:03d}.parquet", index=False)
        n_pairs += len(e)
        log.info("batch %d: S1 %d-%d -> %d pairs (%.0fs)", i, lo, min(lo + args.batch, len(s1)), len(e), time.time() - t0)
    meta = {"n_s1": len(s1), "pairs": n_pairs, "runtime_s": time.time() - t0,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    (d / "phaseA.json").write_text(json.dumps(meta, indent=1))
    log.info("phase A done %s", meta)


def predict_phase_b(args) -> None:
    import resource

    from .stack import StackModels, stage3_scores

    t0 = time.time()
    d = _stage_dir()
    M = StackModels.load_dir(models_dir())
    files = sorted(d.glob("P_*.parquet"))
    parts = [pd.read_parquet(f) for f in files]
    offs = np.r_[0, np.cumsum([len(p) for p in parts])]
    P = pd.concat(parts, ignore_index=True)
    del parts
    XS3 = []
    for i, f in enumerate(files):
        x = pd.read_parquet(str(f).replace("P_", "XS3_"))
        x["row"] = x["row"].to_numpy() + offs[i]
        XS3.append(x)
    XS3 = pd.concat(XS3, ignore_index=True)
    s1 = pd.read_parquet(work_dir() / "test_s1_norm.parquet", columns=["entity_id", "country", "name_key"])
    tg_ids = pd.read_parquet(work_dir() / "test_t_norm.parquet", columns=["entity_id"])["entity_id"].to_numpy()
    final = stage3_scores(P, pd.factorize(s1["name_key"])[0], XS3, M)
    del XS3
    od = out_dir()
    ids = s1["entity_id"].to_numpy()
    sr, tr = P["s1_row"].to_numpy(), P["t_row"].to_numpy()
    keep = final >= M.m3_thr
    # candidate_pairs.tsv = exactly the set scored by the matcher stack
    write_lists_rows(od / "candidate_pairs.tsv", ids, sr, tg_ids, tr, "candidate_entity_ids")
    write_lists_rows(od / "matching_results.tsv", ids, sr[keep], tg_ids, tr[keep], "matched_entity_ids")
    country = s1["country"].to_numpy()
    n_c = np.bincount(sr, minlength=len(ids))
    n_p = np.bincount(sr[keep], minlength=len(ids))
    stats = {"model": "M3", "threshold": M.m3_thr, "s1": len(ids), "candidates": len(P),
             "pred_pairs": int(keep.sum()), "empty_pred_rate": float((n_p == 0).mean()),
             "zero_candidate_s1": int((n_c == 0).sum()), "by_country": {},
             "runtime_s": time.time() - t0,
             "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2}
    for c in pd.unique(country):
        m = country == c
        stats["by_country"][str(c)] = {"s1": int(m.sum()), "cands_per_s1": float(n_c[m].mean()),
                                       "pred_per_s1": float(n_p[m].mean()),
                                       "empty_pred_rate": float((n_p[m] == 0).mean())}
    (work_dir() / "test_inference_stats.json").write_text(json.dumps(stats, indent=1))
    log.info("phase B done %s", json.dumps(stats))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "predict"])
    ap.add_argument("--batch", type=int, default=60_000, help="S1 rows per scoring batch")
    ap.add_argument("--phase", choices=["all", "A", "B"], default="all", help="predict: run phase A, B or both")
    args = ap.parse_args()
    {"validate": cmd_validate, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
