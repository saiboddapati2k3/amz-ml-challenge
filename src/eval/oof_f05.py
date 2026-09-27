"""Macro F0.5 of out-of-fold scores at several thresholds, per country (competition rules: every
S1 of --ids counts, true matches the blocker missed are misses).

    python -m src.eval.oof_f05 --oof artifacts/model/oof_m10.parquet --ids artifacts/splits/train10_s1.parquet
    python -m src.eval.oof_f05 --oof artifacts/model/oof_m10.parquet --ids artifacts/splits/dev_s1.parquet
"""
from __future__ import annotations

import argparse

import polars as pl

from src.eval.blocking_recall import truth_pairs
from src.eval.owner_eval import macro_f05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="parquet with s1_eid, cand_eid, y, p")
    ap.add_argument("--ids", required=True, help="parquet (eid, country): S1 to score")
    ap.add_argument("--truth", default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--taus", default="0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9")
    args = ap.parse_args()
    ids = pl.read_parquet(args.ids).select(pl.col("eid").alias("s1_eid"), "country")
    oof = pl.read_parquet(args.oof, columns=["s1_eid", "cand_eid", "y", "p"]).join(ids.select("s1_eid"), on="s1_eid")
    n_true = truth_pairs(args.truth).join(ids.select("s1_eid"), on="s1_eid").group_by("s1_eid").agg(n_true=pl.len())
    rows = []
    for tau in (float(t) for t in args.taus.split(",")):
        f = macro_f05(oof.filter(pl.col("p") >= tau), n_true, ids)
        by = dict(f.group_by("country").agg(pl.col("f").mean()).iter_rows())
        rows.append({"tau": tau, "ALL": f["f"].mean(), **by})
    with pl.Config(tbl_rows=30, float_precision=5):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
