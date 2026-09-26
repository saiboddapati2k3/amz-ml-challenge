"""Oracle ceiling: macro F0.5 of a perfect matcher restricted to the blocker's candidates.

Per S1 the oracle predicts truth & candidates, so precision is 1 and
F0.5 = 1.25 r / (0.25 + r) with r = retrieved / true; singletons score 1.

    python -m src.eval.oracle --cand artifacts/cand/train_dev_v2.parquet --s1-ids artifacts/splits/dev_s1.parquet
"""
from __future__ import annotations

import argparse

import polars as pl

from src.block import read_cand
from src.eval.blocking_recall import truth_pairs


def oracle_table(cand: pl.DataFrame, tp: pl.DataFrame, ids: pl.DataFrame) -> pl.DataFrame:
    """One row per S1 in `ids`: n_true, n_found, oracle F0.5."""
    hit = (tp.join(cand.select("s1_eid", "cand_eid").with_columns(found=pl.lit(True)),
                   on=["s1_eid", "cand_eid"], how="left")
             .group_by("s1_eid").agg(n_true=pl.len(), n_found=pl.col("found").fill_null(False).sum()))
    r = pl.col("n_found") / pl.col("n_true")
    return (ids.rename({"eid": "s1_eid"}).join(hit, on="s1_eid", how="left")
               .with_columns(pl.col("n_true", "n_found").fill_null(0))
               .with_columns(f=pl.when(pl.col("n_true") == 0).then(1.0)
                               .otherwise(1.25 * r / (0.25 + r)).fill_nan(0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--s1-ids", required=True)
    ap.add_argument("--truth", default="dataset/train/train_ground_truth.tsv")
    args = ap.parse_args()
    ids = pl.read_parquet(args.s1_ids).select("eid", "country")
    tp = truth_pairs(args.truth).join(ids.select(pl.col("eid").alias("s1_eid")), on="s1_eid")
    t = oracle_table(read_cand(args.cand), tp, ids)
    aggs = dict(n=pl.len(), oracle_f05=pl.col("f").mean(),
                singleton_share=(pl.col("n_true") == 0).mean(),
                zero_found=((pl.col("n_true") > 0) & (pl.col("n_found") == 0)).mean(),
                partial=((pl.col("n_found") > 0) & (pl.col("n_found") < pl.col("n_true"))).mean())
    with pl.Config(tbl_rows=50, tbl_cols=20, ascii_tables=True):
        print(t.with_columns(country=pl.lit("ALL")).group_by("country").agg(**aggs))
        print(t.group_by("country").agg(**aggs).sort("n", descending=True))


if __name__ == "__main__":
    main()
