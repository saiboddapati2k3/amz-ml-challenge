"""Decision rules at full density: one-owner mode x threshold, scored on train S1 outside the model's
training data.

The one-owner rule needs every competing S1, so it can't be measured on a 1-2% sample (dev,
lockbox). Here the scores cover the WHOLE train split (src.predict --split train --no-outputs on
the full-train candidates); candidate statistics use all of them, and the macro F0.5 is computed on
the requested S1 only (exclude the S1 the model was trained on).

    python -m src.eval.owner_eval --scores artifacts/score/train_full_n1_v3h1 \
        --ids artifacts/splits/tune_s1.parquet --taus 0.5,0.6,0.7,0.75,0.8,0.9
"""
from __future__ import annotations

import argparse
import os

import polars as pl

from src.eval.blocking_recall import truth_pairs
from src.predict import OWNER_MODES, cand_stats, owner_adjust


def macro_f05(pred: pl.DataFrame, n_true: pl.DataFrame, ids: pl.DataFrame) -> pl.DataFrame:
    """Per-S1 F0.5 (competition rules) from predicted pairs with a y column. -> (s1_eid, country, f)."""
    per = pred.group_by("s1_eid").agg(n_pred=pl.len(), tp=pl.col("y").sum())
    d = (ids.join(n_true, on="s1_eid", how="left").join(per, on="s1_eid", how="left")
            .with_columns(pl.col("n_true", "n_pred", "tp").fill_null(0)))
    f = (pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
           .when((pl.col("n_pred") == 0) | (pl.col("tp") == 0)).then(0.0)
           .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred"))))
    return d.with_columns(f=f).select("s1_eid", "country", "f", "n_pred", "tp", "n_true")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="score dir covering the whole split")
    ap.add_argument("--ids", required=True, help="parquet (eid, country): S1 to evaluate")
    ap.add_argument("--truth", default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--taus", default="0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9")
    ap.add_argument("--modes", default=",".join(OWNER_MODES))
    ap.add_argument("--mix", default="US:663106,India:809986", help="test S1 counts for the test-mix row")
    ap.add_argument("--out", default=None, help="optional csv of all rows")
    args = ap.parse_args()
    taus = [float(t) for t in args.taus.split(",")]
    scores = pl.scan_parquet(os.path.join(args.scores, "part-*.parquet"))
    stats = cand_stats(scores)
    ids = pl.read_parquet(args.ids).select(pl.col("eid").alias("s1_eid"), "country")
    tp = truth_pairs(args.truth).join(ids.select("s1_eid"), on="s1_eid")
    n_true = tp.group_by("s1_eid").agg(n_true=pl.len())
    pairs = (scores.filter(pl.col("p") >= min(taus) * 0.5).select("s1_eid", "cand_eid", "p")
                   .join(ids.lazy().select("s1_eid"), on="s1_eid", how="semi").collect(engine="streaming")
                   .join(tp.select("s1_eid", "cand_eid", y=pl.lit(1, pl.Int32)), on=["s1_eid", "cand_eid"], how="left")
                   .with_columns(pl.col("y").fill_null(0)))
    mix = {k: int(v) for k, v in (x.split(":") for x in args.mix.split(","))}
    rows = []
    for mode in args.modes.split(","):
        adj = owner_adjust(pairs, stats, mode)
        for tau in taus:
            f = macro_f05(adj.filter(pl.col("p") >= tau), n_true, ids)
            by = {c: v for c, v in f.group_by("country").agg(pl.col("f").mean()).iter_rows()}
            row = {"mode": mode, "tau": tau, "ALL": f["f"].mean(), **by,
                   "test_mix": sum(by.get(c, 0) * n for c, n in mix.items()) / sum(mix.values()),
                   "precision": f["tp"].sum() / max(f["n_pred"].sum(), 1), "recall": f["tp"].sum() / max(f["n_true"].sum(), 1)}
            rows.append(row)
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    res = pl.DataFrame(rows)
    with pl.Config(tbl_rows=60, tbl_cols=20, float_precision=5, tbl_width_chars=200):
        print(res.sort("test_mix", descending=True))
    if args.out:
        res.write_csv(args.out)


if __name__ == "__main__":
    main()
