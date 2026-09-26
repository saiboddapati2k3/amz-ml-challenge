"""Deterministic dev sample of train S1 entities (for fast blocking / model iteration).

    python -m scripts.make_dev_split --frac 100   # keep 1 in 100 S1 ids
"""
import argparse
import os

import polars as pl

from src.prep import load

ap = argparse.ArgumentParser()
ap.add_argument("--norm", default="artifacts/norm")
ap.add_argument("--frac", type=int, default=100, help="keep 1 in N S1 entities")
ap.add_argument("--out", default="artifacts/splits/dev_s1.parquet")
args = ap.parse_args()

s1 = load(args.norm, "train", 1, ["eid", "country"])
dev = s1.filter(pl.col("eid").hash(seed=42) % args.frac == 0)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
dev.write_parquet(args.out)
print(f"dev: {dev.height:,} of {s1.height:,} S1")
print(dev["country"].value_counts(sort=True))
