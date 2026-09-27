"""Training sample of train S1: every dev S1 plus random others, never a lockbox S1.

    python -m scripts.make_train_sample --frac 0.10 --out artifacts/splits/train10_s1.parquet
"""
import argparse

import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--norm", default="artifacts/norm")
ap.add_argument("--dev", default="artifacts/splits/dev_s1.parquet")
ap.add_argument("--lockbox", default="artifacts/splits/lockbox_s1.parquet")
ap.add_argument("--frac", type=float, default=0.10, help="share of all train S1")
ap.add_argument("--seed", type=int, default=2027)
ap.add_argument("--out", default="artifacts/splits/train10_s1.parquet")
args = ap.parse_args()

s1 = pl.read_parquet(f"{args.norm}/train_s1/*.parquet", columns=["eid", "country"])
dev = pl.read_parquet(args.dev).select("eid")
box = pl.read_parquet(args.lockbox).select("eid")
assert dev.join(box, on="eid", how="semi").height == 0, "dev and lockbox overlap"
n = round(args.frac * s1.height)
pool = s1.join(dev, on="eid", how="anti").join(box, on="eid", how="anti").sort("eid")
extra = pool.sample(n=n - dev.height, seed=args.seed)
out = pl.concat([s1.join(dev, on="eid", how="semi"), extra]).sort("eid")
assert out.join(box, on="eid", how="semi").height == 0
out.write_parquet(args.out)
print(f"{args.out}: {out.height:,} S1 ({out.height / s1.height:.1%} of train), "
      f"{out.join(dev, on='eid', how='semi').height:,} from dev")
print(out.group_by("country").len().sort("country"))
