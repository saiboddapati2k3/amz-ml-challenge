"""Phase 3b: LightGBM pair classifier with out-of-fold predictions.

Folds are over S1 entities (src.eval.splits.assign_folds: stratified by country and
match-count bucket), so every candidate of one S1 lands in the same fold. Early stopping
uses an inner S1-grouped holdout carved from the training folds -- the OOF fold itself is
never seen during fitting, so OOF scores are safe for calibration and decision tuning.

    python -m src.train --feat artifacts/feat/train_dev.parquet --s1-ids artifacts/splits/dev_s1.parquet --tag dev
    python -m src.train ... --loco        # leave-one-country-out instead of K folds
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from src.eval.blocking_recall import truth_pairs
from src.eval.splits import assign_folds

ID_COLS = ["s1_eid", "cand_eid", "y"]
# Candidate-side competition features depend on how many S1 the candidate file covers (near
# constant on a 1% dev sample, large on a full split): excluded until trained on full-split data.
EXCLUDE = ["cand_n_s1", "cand_rank", "cand_ratio", "cand_nm_gap"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, verbose=-1, seed=42, num_threads=0)
MAX_ROUNDS = 3000


def s1_folds(ids: pl.DataFrame, truth: str, k: int, seed: int) -> pl.DataFrame:
    """(s1_eid, fold, country) for every S1 in ids (eid, country)."""
    n_true = (truth_pairs(truth).group_by("s1_eid").agg(n=pl.len()))
    d = ids.rename({"eid": "s1_eid"}).join(n_true, on="s1_eid", how="left").with_columns(pl.col("n").fill_null(0))
    s = d["s1_eid"].to_list()
    folds = assign_folds(s, dict(zip(s, d["country"].to_list())),
                         {e: [0] * n for e, n in zip(s, d["n"].to_list())}, k=k, seed=seed)
    return d.select("s1_eid", "country", fold=pl.Series([folds[e] for e in s], dtype=pl.Int16))


def fit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, names, seed: int = 42):
    """Fit with early stopping on a 10% S1-grouped inner holdout."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    hold = np.isin(groups, rng.choice(ug, size=max(1, len(ug) // 10), replace=False))
    dtr = lgb.Dataset(X[~hold], y[~hold], feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(X[hold], y[hold], reference=dtr)
    return lgb.train(PARAMS, dtr, MAX_ROUNDS, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def run(feat: pl.DataFrame, folds: pl.DataFrame, out_dir: str, tag: str, loco: bool, log=print):
    feat = feat.join(folds, on="s1_eid")
    names = [c for c in feat.columns if c not in ID_COLS + EXCLUDE + ["fold", "country"]]
    X = feat.select(names).to_numpy().astype(np.float32)
    y = feat["y"].to_numpy()
    g = feat["s1_eid"].to_numpy()
    split = feat["country"].to_numpy() if loco else feat["fold"].to_numpy()
    oof = np.full(len(y), np.nan, dtype=np.float32)
    gain = np.zeros(len(names))
    meta = []
    for k in np.unique(split):
        t0 = time.time()
        te = split == k
        m = fit(X[~te], y[~te], g[~te], names)
        oof[te] = m.predict(X[te], num_iteration=m.best_iteration)
        os.makedirs(out_dir, exist_ok=True)
        m.save_model(os.path.join(out_dir, f"model_{tag}_{k}.txt"), num_iteration=m.best_iteration)
        gain += m.feature_importance("gain")
        meta.append({"split": str(k), "best_iter": m.best_iteration, "n_train": int((~te).sum()),
                     "n_test": int(te.sum()), "sec": round(time.time() - t0, 1)})
        log(f"  split {k}: best_iter {m.best_iteration}, {time.time() - t0:.0f}s")
    os.makedirs(out_dir, exist_ok=True)
    res = feat.select("s1_eid", "cand_eid", "src", "country", "fold", "y").with_columns(p=pl.Series(oof))
    res.write_parquet(os.path.join(out_dir, f"oof_{tag}.parquet"))
    imp = pl.DataFrame({"feature": names, "gain": gain / gain.sum()}).sort("gain", descending=True)
    imp.write_csv(os.path.join(out_dir, f"importance_{tag}.csv"))
    with open(os.path.join(out_dir, f"meta_{tag}.json"), "w") as fh:
        json.dump({"params": PARAMS, "features": names, "splits": meta}, fh, indent=1)
    return res, imp


def load_parts(pattern: str, folds: pl.DataFrame, log=print):
    """Feature parts -> (float32 X, id frame, names) without ever holding a full frame: rows are
    copied part by part into one preallocated array (the 10% sample is ~9M rows x 73)."""
    paths = sorted(glob.glob(pattern))
    schema = pl.read_parquet_schema(paths[0])
    names = [c for c in schema if c not in ID_COLS + EXCLUDE + ["fold", "country"]]
    keep = folds.select("s1_eid", "country", "fold")
    n = sum(pl.scan_parquet(p).join(keep.lazy().select("s1_eid"), on="s1_eid", how="semi")
              .select(pl.len()).collect().item() for p in paths)
    X = np.empty((n, len(names)), dtype=np.float32)
    ids, o = [], 0
    for p in paths:
        extra = [c for c in ("s1_eid", "cand_eid", "src", "y") if c not in names]  # 'src' is also a feature
        d = pl.read_parquet(p, columns=names + extra).join(keep, on="s1_eid")
        X[o:o + d.height] = d.select(names).to_numpy().astype(np.float32, copy=False)
        ids.append(d.select("s1_eid", "cand_eid", "src", "country", "fold", "y"))
        o += d.height
    assert o == n
    log(f"  loaded {n:,} rows x {len(names)} features from {len(paths)} parts ({X.nbytes / 2**30:.1f} GB)")
    return X, pl.concat(ids), names


def run_lean(pattern: str, folds: pl.DataFrame, out_dir: str, tag: str, log=print):
    """K-fold like run(), for large samples: the binned LightGBM dataset is built once and every
    fold trains on a subset of it, so no fold copies X."""
    X, ids, names = load_parts(pattern, folds, log)
    y = ids["y"].to_numpy().astype(np.float32)
    g = ids["s1_eid"].to_numpy()
    fold = ids["fold"].to_numpy()
    full = lgb.Dataset(X, label=y, feature_name=names, free_raw_data=False,
                       params={"max_bin": PARAMS["max_bin"], "verbose": -1}).construct()
    oof = np.full(len(y), np.nan, dtype=np.float32)
    gain = np.zeros(len(names))
    meta = []
    for k in np.unique(fold):
        t0 = time.time()
        tr = np.flatnonzero(fold != k)
        rng = np.random.default_rng(42)
        ug = np.unique(g[tr])
        hold = np.isin(g[tr], rng.choice(ug, size=max(1, len(ug) // 10), replace=False))
        dtr, dva = full.subset(tr[~hold].tolist()), full.subset(tr[hold].tolist())
        m = lgb.train(PARAMS, dtr, MAX_ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        te = np.flatnonzero(fold == k)
        oof[te] = m.predict(X[te], num_iteration=m.best_iteration)
        os.makedirs(out_dir, exist_ok=True)
        m.save_model(os.path.join(out_dir, f"model_{tag}_{k}.txt"), num_iteration=m.best_iteration)
        gain += m.feature_importance("gain")
        meta.append({"split": str(k), "best_iter": m.best_iteration, "n_train": int(len(tr)),
                     "n_test": int(len(te)), "sec": round(time.time() - t0, 1)})
        log(f"  split {k}: best_iter {m.best_iteration}, {time.time() - t0:.0f}s")
    res = ids.with_columns(p=pl.Series(oof))
    res.write_parquet(os.path.join(out_dir, f"oof_{tag}.parquet"))
    imp = pl.DataFrame({"feature": names, "gain": gain / gain.sum()}).sort("gain", descending=True)
    imp.write_csv(os.path.join(out_dir, f"importance_{tag}.csv"))
    with open(os.path.join(out_dir, f"meta_{tag}.json"), "w") as fh:
        json.dump({"params": PARAMS, "features": names, "splits": meta}, fh, indent=1)
    return res, imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True, help="one parquet file, or a directory of feature parts")
    ap.add_argument("--s1-ids", required=True)
    ap.add_argument("--truth", default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--out", default="artifacts/model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loco", action="store_true")
    args = ap.parse_args()
    lg = lambda m: print(m, flush=True)
    folds = s1_folds(pl.read_parquet(args.s1_ids).select("eid", "country"), args.truth, args.k, args.seed)
    os.makedirs(args.out, exist_ok=True)
    folds.write_parquet(os.path.join(args.out, f"folds_{args.tag}.parquet"))
    if os.path.isdir(args.feat):
        assert not args.loco, "--loco needs a single feature file"
        res, imp = run_lean(os.path.join(args.feat, "*.parquet"), folds, args.out, args.tag, lg)
    else:
        res, imp = run(pl.read_parquet(args.feat), folds, args.out, args.tag, args.loco, lg)
    from sklearn.metrics import average_precision_score, roc_auc_score
    lg(f"OOF AUC {roc_auc_score(res['y'], res['p']):.5f}  AP {average_precision_score(res['y'], res['p']):.5f}")
    with pl.Config(tbl_rows=15, ascii_tables=True):
        lg(imp.head(15))


if __name__ == "__main__":
    main()
