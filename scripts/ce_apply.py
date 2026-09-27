"""Blend cross-encoder scores into a LightGBM score dir, part by part:
p = sigmoid(w * logit(p_lgb) + (1 - w) * logit(p_ce)) for pairs that have a cross-encoder score
(the uncertain band); every other pair keeps its LightGBM score.

    python scripts/ce_apply.py artifacts/score/test_n2_v3h1__m10f3 ce_data/ce_scores_test.parquet \
        artifacts/score/test_n2_v3h1__m10f3ce --w 0.55
Then write the submission from the cached blended scores (src.predict with a cand view named like the
output dir; every part is cached, so nothing is re-scored).
"""
import argparse
import glob
import os

import numpy as np
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("ce")
ap.add_argument("out")
ap.add_argument("--w", type=float, default=0.55, help="weight of the LightGBM logit")
args = ap.parse_args()
ce = pl.read_parquet(args.ce).select("s1_eid", "cand_eid", "ce")
os.makedirs(args.out, exist_ok=True)
lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
n_all = n_ce = 0
for path in sorted(glob.glob(os.path.join(args.src, "part-*.parquet"))):
    d = pl.read_parquet(path).join(ce, on=["s1_eid", "cand_eid"], how="left", maintain_order="left")
    z = args.w * lg("p") + (1 - args.w) * lg("ce")
    d = d.with_columns(p=pl.when(pl.col("ce").is_not_null()).then(1 / (1 + (-z).exp())).otherwise(pl.col("p")).cast(pl.Float32))
    n_all += d.height
    n_ce += d["ce"].is_not_null().sum()
    dst = os.path.join(args.out, os.path.basename(path))
    d.drop("ce").write_parquet(dst + ".tmp")
    os.replace(dst + ".tmp", dst)
print(f"{args.out}: {n_all:,} pairs, {n_ce:,} blended with the cross-encoder ({n_ce / max(n_all, 1):.2%}); "
      f"cross-encoder rows {ce.height:,}")
