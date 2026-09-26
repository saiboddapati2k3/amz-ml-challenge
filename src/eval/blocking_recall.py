"""Blocking recall on train: share of ground-truth (S1, S2/S3) pairs present in the candidates.

    python -m src.eval.blocking_recall --cand artifacts/cand/train_dev.parquet --s1-ids artifacts/splits/dev_s1.parquet
"""
from __future__ import annotations

import argparse
import sys

import polars as pl

from src.block import read_cand
from src.prep import encode_id_expr, load


def truth_pairs(path: str) -> pl.DataFrame:
    gt = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return (gt.select(pl.col("source1_entity_id").alias("a"),
                      pl.col("matched_entity_ids").fill_null("").str.split(",").alias("b"))
              .explode("b").with_columns(pl.col("b").str.strip_chars())
              .filter(pl.col("b").str.len_chars() > 0)
              .select(encode_id_expr("a").alias("s1_eid"), encode_id_expr("b").alias("cand_eid"),
                      pl.col("b").str.slice(1, 1).cast(pl.Int8).alias("src")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--s1-ids", default=None)
    ap.add_argument("--truth", default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--norm", default="artifacts/norm")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # polars tables use box-drawing chars; cp1252 consoles fail

    cand = read_cand(args.cand)
    tp = truth_pairs(args.truth)
    if args.s1_ids:
        ids = pl.read_parquet(args.s1_ids).select("eid", "country")
    else:
        ids = load(args.norm, "train", 1, ["eid", "country"])
    tp = tp.join(ids.rename({"eid": "s1_eid"}), on="s1_eid")
    hit = tp.join(cand.select("s1_eid", "cand_eid", "r_all"), on=["s1_eid", "cand_eid"], how="left")
    hit = hit.with_columns(found=pl.col("r_all").is_not_null())

    n1 = ids.height
    print(f"S1 evaluated: {n1:,}   truth pairs: {tp.height:,}   candidates: {cand.height:,} "
          f"({cand.height / max(n1, 1):.1f} per S1)")
    print(f"pair recall: {hit['found'].mean():.4f}")
    print(hit.group_by("src").agg(recall=pl.col("found").mean(), n=pl.len()).sort("src"))
    print(hit.group_by("country").agg(recall=pl.col("found").mean(), n=pl.len()).sort("n", descending=True))
    for k in (5, 10, 20, 40):
        print(f"  recall@r_all<={k}: {(hit['r_all'] <= k).fill_null(False).mean():.4f}")
    # entity-level: share of S1 with all true matches retrieved
    ent = hit.group_by("s1_eid").agg(pl.col("found").all())
    print(f"S1 with every true match retrieved: {ent['found'].mean():.4f} (of {ent.height:,} with matches)")


if __name__ == "__main__":
    main()
