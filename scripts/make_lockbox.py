"""Lockbox holdout: train S1 entities disjoint from the dev sample, for honest milestone scores.

Stratified by country x match-count bucket (0 / 1 / 2 / 3+) with the same seeded stable ranking
as src.eval.splits.assign_folds, so its singleton share matches train. The lockbox is scored
only at milestones and is never used to choose or tune anything (see STATUS.md).

    python -m scripts.make_lockbox --n 45000
"""
import argparse
import os

import polars as pl

from src.eval.blocking_recall import truth_pairs
from src.eval.splits import _stable_rank
from src.io_utils import ID_BASE

ap = argparse.ArgumentParser()
ap.add_argument("--norm", default="artifacts/norm")
ap.add_argument("--truth", default="dataset/train/train_ground_truth.tsv")
ap.add_argument("--dev", default="artifacts/splits/dev_s1.parquet")
ap.add_argument("--n", type=int, default=45_000)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--out", default="artifacts/splits")
args = ap.parse_args()

from src.prep import load  # noqa: E402

s1 = load(args.norm, "train", 1, ["eid", "country"])
dev = pl.read_parquet(args.dev).select("eid", "country")
n_true = truth_pairs(args.truth).group_by("s1_eid").agg(n=pl.len()).rename({"s1_eid": "eid"})
pool = (s1.join(dev, on="eid", how="anti").join(n_true, on="eid", how="left")
          .with_columns(bucket=pl.col("n").fill_null(0).clip(0, 3)))
frac = args.n / pool.height
pool = pool.with_columns(rk=pl.col("eid").map_elements(lambda e: _stable_rank(str(e), args.seed),
                                                        return_dtype=pl.UInt64))
box = (pool.sort("rk").with_columns(pos=pl.int_range(pl.len()).over("country", "bucket"),
                                    size=pl.len().over("country", "bucket"))
           .filter(pl.col("pos") < (pl.col("size") * frac).round())
           .select("eid", "country", "bucket").sort("eid"))
assert box.join(dev, on="eid", how="semi").height == 0, "lockbox overlaps dev"

decode = lambda c: pl.concat_str([pl.lit("S"), (pl.col(c) // ID_BASE).cast(pl.Utf8), pl.lit("-"),
                                  (pl.col(c) % ID_BASE).cast(pl.Utf8)])
os.makedirs(args.out, exist_ok=True)
box.select("eid", "country").write_parquet(os.path.join(args.out, "lockbox_s1.parquet"))
for name, d in (("lockbox_s1.txt", box), ("dev_s1.txt", dev.sort("eid"))):
    d.select(decode("eid")).write_csv(os.path.join(args.out, name), include_header=False)

full = s1.join(n_true, on="eid", how="left").with_columns(bucket=pl.col("n").fill_null(0).clip(0, 3))
mix = lambda d, tag: d.group_by("country", "bucket").len().with_columns(
    (pl.col("len") / pl.col("len").sum()).round(4).alias(tag)).drop("len")
print(f"lockbox {box.height:,} S1 (dev {dev.height:,}, pool {pool.height:,}); disjoint from dev: OK")
with pl.Config(tbl_rows=20, ascii_tables=True):
    print(mix(full, "train_share").join(mix(box, "lockbox_share"), on=["country", "bucket"])
          .sort("country", "bucket"))
