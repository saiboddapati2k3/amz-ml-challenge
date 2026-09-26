"""Phase 4 (on OOF): compare decision rules by exact macro F0.5 over ALL evaluated S1.

Calibration is cross-fitted: isotonic maps for fold k are fit on the OOF scores of the
other folds (separately for S2 and S3 candidates), so no score is calibrated on itself.
Rule parameters (thresholds, one-owner mode) are chosen here on OOF and only then frozen
for test.

    python -m src.eval.decide_oof --oof artifacts/model/oof_dev.parquet --s1-ids artifacts/splits/dev_s1.parquet
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl
from sklearn.isotonic import IsotonicRegression

from src.decide import apply_one_owner, decide_expected_f, decide_threshold
from src.eval.blocking_recall import truth_pairs
from src.eval.score import breakdown


def calibrate_oof(oof: pl.DataFrame) -> np.ndarray:
    p, y = oof["p"].to_numpy(), oof["y"].to_numpy()
    fold, src = oof["fold"].to_numpy(), oof["src"].to_numpy()
    out = np.empty_like(p)
    for s in np.unique(src):
        for k in np.unique(fold):
            tr, te = (fold != k) & (src == s), (fold == k) & (src == s)
            if te.any():
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[tr], y[tr])
                out[te] = iso.predict(p[te])
    return out


def truth_dict(path: str, ids: List[int]) -> Dict[int, List[int]]:
    tp = truth_pairs(path).join(pl.DataFrame({"s1_eid": ids}), on="s1_eid")
    t = {s: [] for s in ids}
    for a, b in zip(tp["s1_eid"].to_list(), tp["cand_eid"].to_list()):
        t[a].append(b)
    return t


def ece(p: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--s1-ids", required=True)
    ap.add_argument("--truth", default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--miss-prob", type=float, default=0.0)
    args = ap.parse_args()

    oof = pl.read_parquet(args.oof)
    ids_df = pl.read_parquet(args.s1_ids).select("eid", "country")
    ids = ids_df["eid"].to_list()
    truth = truth_dict(args.truth, ids)
    by_country = {c: g["eid"].to_list() for (c,), g in ids_df.group_by("country")}

    oof = oof.with_columns(pc=pl.Series(calibrate_oof(oof)))
    y = oof["y"].to_numpy()
    print(f"ECE raw {ece(oof['p'].to_numpy(), y):.4f}   isotonic {ece(oof['pc'].to_numpy(), y):.4f}")

    base = oof.select(s1_id="s1_eid", cand_id="cand_eid", p="p", pc="pc").to_pandas()
    rows = []

    def report(name, pred):
        r = {"rule": name, **{k: v for k, v in breakdown(pred, truth, ids).items()
                              if k in ("macro_f05", "micro_precision", "micro_recall")}}
        for c, cid in by_country.items():
            r[f"f05_{c}"] = breakdown(pred, truth, cid)["macro_f05"]
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

    report("top1 always", {s: [c] for s, c in base.sort_values("p").groupby("s1_id")["cand_id"].last().items()})
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        report(f"threshold p>={tau}", decide_threshold(base, ids, "p", tau, tau, 0.0))
    for oo in ("none", "soft", "hard"):
        d = apply_one_owner(base, oo, col="pc")
        report(f"expF iso one_owner={oo}", decide_expected_f(d, ids, "pc", miss_prob=args.miss_prob))
    report("expF raw", decide_expected_f(base, ids, "p", miss_prob=args.miss_prob))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
