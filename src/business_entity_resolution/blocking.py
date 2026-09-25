"""Candidate generation: several simple, individually measurable passes.

All passes operate *within* a country value taken from the data (open set, no
hard-coded labels); the audit found zero cross-country true pairs.

Passes
------
name_exact : identical order-insensitive core-name key (skips huge blocks)
name_tfidf : top-k cosine over IDF-weighted core-name tokens (+ concat token)
addr_tfidf : top-k cosine over IDF-weighted normalized address tokens

No pass compares all pairs: exact is a hash join, the TF-IDF passes use a
sparse top-k matrix product over a vocabulary pruned of very frequent tokens.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import sp_matmul_topn

from .config import N_JOBS, BlockingConfig

log = logging.getLogger(__name__)

PASSES = ("name_exact", "name_tfidf", "addr_tfidf")
# Experimental passes (Phase 4A). Off by default; their flags are reported for
# blocking analysis but are NOT matcher features (the EXP001 model is unchanged).
EXTRA_PASSES = ("x_char_name", "x_hybrid", "x_addr_expand", "x_empty_addr_name")


def name_doc(df: pd.DataFrame) -> list[str]:
    return [f"{c} #{k}" if k else c for c, k in zip(df["name_core"], df["name_concat"])]


def addr_doc(df: pd.DataFrame) -> list[str]:
    return df["addr_norm"].tolist()


class TfidfIndex:
    """IDF-weighted, L2-normalized binary token vectors fitted on the target pool.

    Tokens are hashed (2^24 buckets, deterministic murmurhash) so no Python
    vocabulary is held in memory; near-unique tokens such as concatenated names
    are therefore cheap.
    """

    N_FEATURES = 2 ** 24

    def __init__(self, target_docs: list[str], max_df: int, char_ngrams: int = 0):
        kw = (dict(analyzer="char_wb", ngram_range=(char_ngrams, char_ngrams))
              if char_ngrams else dict(analyzer=str.split))
        self.hv = HashingVectorizer(n_features=self.N_FEATURES, alternate_sign=False,
                                    norm=None, binary=True, dtype=np.float32, **kw)
        X = self.hv.transform(target_docs).tocsr()
        df = np.bincount(X.indices, minlength=self.N_FEATURES)
        n = X.shape[0]
        idf = (np.log((n + 1) / (df + 1)) + 1).astype(np.float32)
        idf[df > max_df] = 0  # prune very frequent tokens from blocking vectors
        self.idf = idf
        self.T = self._weight(X)
        self.TT = self.T.T.tocsr()

    def _weight(self, X: sp.csr_matrix) -> sp.csr_matrix:
        X = X.tocsr(copy=True)
        X.data = self.idf[X.indices] * X.data
        X.eliminate_zeros()
        norm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
        norm[norm == 0] = 1
        X.data = X.data / np.repeat(norm, np.diff(X.indptr)).astype(np.float32)
        return X

    def transform(self, docs: list[str]) -> sp.csr_matrix:
        return self._weight(self.hv.transform(docs).tocsr())


def pair_cosine(Q: sp.csr_matrix, T: sp.csr_matrix, qi: np.ndarray, tj: np.ndarray,
                chunk: int = 2_000_000) -> np.ndarray:
    """Row-wise cosine for explicit (query row, target row) pairs."""
    out = np.empty(len(qi), np.float32)
    for a in range(0, len(qi), chunk):
        b = a + chunk
        out[a:b] = np.asarray(Q[qi[a:b]].multiply(T[tj[a:b]]).sum(1)).ravel()
    return out


class Blocker:
    """Fits per-country indexes on the target pool; queries any S1 subset."""

    def __init__(self, targets: pd.DataFrame, cfg: BlockingConfig):
        self.cfg = cfg
        self.tg = targets
        self.by_country = {}
        for c, idx in targets.groupby("country").indices.items():
            sub = targets.iloc[idx]
            log.info("fit indexes country=%s targets=%d", c, len(idx))
            m = dict(
                idx=idx,
                name=TfidfIndex(name_doc(sub), cfg.name_max_df),
                addr=TfidfIndex(addr_doc(sub), cfg.addr_max_df),
                key_counts=sub["name_key"].value_counts(),
                addr_empty=sub["addr_empty"].to_numpy(),
            )
            if cfg.char_name:
                m["char"] = TfidfIndex(sub["name_core"].tolist(), cfg.char_max_df, char_ngrams=3)
            if cfg.hybrid:
                # [name | w*addr] stacked: dot product = cos_name + w * cos_addr
                m["hybrid_TT"] = sp.vstack([m["name"].TT, m["addr"].TT]).tocsr()
            self.by_country[c] = m

    @staticmethod
    def _topk_frame(TT, Q, k, cfg, code, rows=None):
        C_i, C_j, C_r = [], [], []
        for a in range(0, Q.shape[0], cfg.chunk_rows):
            C = sp_matmul_topn(Q[a:a + cfg.chunk_rows], TT, top_n=k, threshold=cfg.min_score,
                               sort=True, n_threads=N_JOBS).tocsr()
            cnt = np.diff(C.indptr)
            C_i.append(np.repeat(np.arange(C.shape[0]) + a, cnt))
            C_j.append(C.indices)
            C_r.append(np.arange(C.nnz) - np.repeat(C.indptr[:-1], cnt))
        i = np.concatenate(C_i) if C_i else np.array([], np.int64)
        if rows is not None:
            i = rows[i]
        return pd.DataFrame({"qi": i.astype(np.int32),
                             "tj": (np.concatenate(C_j) if C_j else i).astype(np.int32),
                             "pass": np.int8(code),
                             "rank": (np.concatenate(C_r) if C_r else i).astype(np.int16)})

    def candidates(self, s1: pd.DataFrame) -> pd.DataFrame:
        """Union of all enabled passes for the given S1 rows.

        Returns one row per (s1 row position, target row position) with a
        boolean flag per pass, ranks for the baseline passes, and both TF-IDF
        cosines for every pair.
        """
        cfg = self.cfg
        all_passes = PASSES + EXTRA_PASSES
        frames = []
        for c, qidx in s1.groupby("country").indices.items():
            if c not in self.by_country:
                continue  # no targets in this country -> no candidates
            m = self.by_country[c]
            q = s1.iloc[qidx]
            tkeys = self.tg["name_key"].to_numpy()[m["idx"]]
            qkeys = q["name_key"].to_numpy()
            # target frequency of the S1 name key (unsupervised "commonness")
            freq = m["key_counts"].reindex(qkeys).fillna(0).to_numpy()
            parts = []
            # pass 1: exact core-name key, skipping oversized blocks
            ok = m["key_counts"][m["key_counts"] <= cfg.exact_max_block].index
            lk = pd.DataFrame({"qi": np.arange(len(q)), "name_key": qkeys})
            rk = pd.DataFrame({"tj": np.arange(len(tkeys)), "name_key": tkeys})
            ex = lk[lk["name_key"].isin(ok)].merge(rk, on="name_key")[["qi", "tj"]].astype(np.int32)
            ex["pass"] = np.int8(0)
            ex["rank"] = np.int16(0)
            parts.append(ex)
            # passes 2-3: TF-IDF top-k
            Qn = m["name"].transform(name_doc(q))
            Qa = m["addr"].transform(addr_doc(q))
            parts.append(self._topk_frame(m["name"].TT, Qn, cfg.name_topk, cfg, 1))
            parts.append(self._topk_frame(m["addr"].TT, Qa, cfg.addr_topk, cfg, 2))
            # A: adaptive name top-k (still the name_tfidf pass, deeper for common names)
            for lo, k in cfg.adaptive_k:
                rows = np.flatnonzero(freq >= lo)
                if len(rows):
                    parts.append(self._topk_frame(m["name"].TT, Qn[rows], k, cfg, 1, rows))
            # B: char 3-gram name retrieval for common names
            if cfg.char_name:
                rows = np.flatnonzero(freq >= cfg.char_min_freq)
                if len(rows):
                    Qc = m["char"].transform(q["name_core"].iloc[rows].tolist())
                    parts.append(self._topk_frame(m["char"].TT, Qc, cfg.char_topk, cfg, 3, rows))
            # C: hybrid name+address retrieval for common names
            if cfg.hybrid:
                rows = np.flatnonzero(freq >= cfg.hybrid_min_freq)
                if len(rows):
                    Qh = sp.hstack([Qn[rows], cfg.hybrid_w_addr * Qa[rows]]).tocsr()
                    parts.append(self._topk_frame(m["hybrid_TT"], Qh, cfg.hybrid_topk, cfg, 4, rows))
            # D: address-ranked expansion inside the (skipped) exact-name block
            if cfg.addr_expand:
                sel = (freq > cfg.exact_max_block) & (freq <= cfg.expand_max_block) & (freq >= cfg.expand_min_freq)
                parts += list(self._expand_by_address(lk[sel], rk, Qa, m["addr"].T, cfg))
            # E1: same-key targets with EMPTY address (address passes can never find them)
            if cfg.empty_addr_name:
                sel = freq > cfg.exact_max_block
                rk_e = rk[m["addr_empty"]]
                e = lk[sel].merge(rk_e, on="name_key")[["qi", "tj"]]
                e = e.groupby("qi").head(cfg.empty_addr_max).astype(np.int32)
                e["pass"] = np.int8(6)
                e["rank"] = np.int16(0)
                parts.append(e)

            allp = pd.concat(parts, ignore_index=True)
            ranks = allp.groupby(["qi", "tj", "pass"])["rank"].min().unstack("pass")
            ranks = ranks.reindex(columns=range(len(all_passes)))
            ranks.columns = list(all_passes)
            del allp, parts
            u = ranks[list(PASSES)].add_prefix("rank_")
            for p in all_passes:
                u[p] = ranks[p].notna().to_numpy()
            u = u.reset_index()
            qi = u["qi"].to_numpy()
            tj = u["tj"].to_numpy()
            u["cos_name"] = pair_cosine(Qn, m["name"].T, qi, tj)
            u["cos_addr"] = pair_cosine(Qa, m["addr"].T, qi, tj)
            u["s1_row"] = qidx[qi]
            u["t_row"] = m["idx"][tj]
            frames.append(u.drop(columns=["qi", "tj"]))
            log.info("country=%s s1=%d candidates=%d", c, len(q), len(u))
        cols = ["s1_row", "t_row", *all_passes, *[f"rank_{p}" for p in PASSES], "cos_name", "cos_addr"]
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
        return out[cols]

    @staticmethod
    def _expand_by_address(lk, rk, Qa, Ta, cfg, chunk_s1: int = 20_000):
        """For common-name S1s, rank every same-key target by address cosine, keep top-k."""
        for a in range(0, len(lk), chunk_s1):
            pr = lk.iloc[a:a + chunk_s1].merge(rk, on="name_key")[["qi", "tj"]]
            if pr.empty:
                continue
            pr["s"] = pair_cosine(Qa, Ta, pr["qi"].to_numpy(), pr["tj"].to_numpy())
            pr = pr[pr["s"] > 0].sort_values(["qi", "s"], ascending=[True, False])
            pr = pr.groupby("qi").head(cfg.expand_topk)
            yield pd.DataFrame({"qi": pr["qi"].to_numpy(np.int32), "tj": pr["tj"].to_numpy(np.int32),
                                "pass": np.int8(5), "rank": np.int16(0)})
