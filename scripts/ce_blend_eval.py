"""Does the cross-encoder improve the check set? Blend it with the LightGBM score for uncertain pairs
and compare macro F0.5, choosing the blend weight on one half of the S1 and scoring the other half.

    python scripts/ce_blend_eval.py ~/Downloads/ce_scores_check.parquet \
        artifacts/score/train_check_n1_v3h1__m10 artifacts/splits/check_s1.parquet
"""
import os
import sys

import numpy as np
import polars as pl

from src.eval.blocking_recall import truth_pairs
from src.eval.owner_eval import macro_f05

ce_path, score_dir, ids_path = sys.argv[1:4]
LO, HI = 0.001, 0.999
ids = pl.read_parquet(ids_path).select(pl.col("eid").alias("s1_eid"), "country")
tp = truth_pairs("dataset/train/train_ground_truth.tsv").join(ids.select("s1_eid"), on="s1_eid")
n_true = tp.group_by("s1_eid").agg(n_true=pl.len())
sc = (pl.scan_parquet(os.path.join(score_dir, "part-*.parquet")).select("s1_eid", "cand_eid", "p").collect()
        .join(ids.select("s1_eid"), on="s1_eid")
        .join(tp.select("s1_eid", "cand_eid", y=pl.lit(1, pl.Int32)), on=["s1_eid", "cand_eid"], how="left")
        .with_columns(pl.col("y").fill_null(0)))
ce = pl.read_parquet(ce_path).select("s1_eid", "cand_eid", "ce")
sc = sc.join(ce, on=["s1_eid", "cand_eid"], how="left")
band = sc["p"].is_between(LO, HI)
print(f"pairs {sc.height:,}, in band {band.sum():,}, with ce score {sc['ce'].is_not_null().sum():,}")
logit = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
p, c = sc["p"].to_numpy(), sc["ce"].fill_null(0.5).to_numpy()
inb = band.to_numpy() & sc["ce"].is_not_null().to_numpy()


def blended(w: float) -> np.ndarray:
    out = p.copy()
    z = w * logit(p[inb]) + (1 - w) * logit(c[inb])
    out[inb] = 1 / (1 + np.exp(-z))
    return out


rng = np.random.default_rng(0)
s1 = ids["s1_eid"].to_numpy()
half = dict(zip(s1, rng.integers(0, 2, len(s1))))
sc = sc.with_columns(half=pl.col("s1_eid").replace_strict(half, return_dtype=pl.Int8))
taus = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
ws = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0]
res = {}
for w in ws:
    pw = blended(w)
    d = sc.with_columns(q=pl.Series(pw))
    for tau in taus:
        pred = d.filter(pl.col("q") >= tau)
        f = macro_f05(pred.select("s1_eid", "cand_eid", "y"), n_true, ids).join(d.select("s1_eid", "half").unique(), on="s1_eid", how="left")
        f = f.with_columns(pl.col("half").fill_null(0))
        res[(w, tau)] = {h: f.filter(pl.col("half") == h)["f"].mean() for h in (0, 1)} | {"all": f["f"].mean(),
                         **{k: v for k, v in f.group_by("country").agg(pl.col("f").mean()).iter_rows()}}
# cross-fitted: choose (w, tau) on one half, score the other
for fit, ev in ((0, 1), (1, 0)):
    best = max(res, key=lambda k: res[k][fit])
    base = max((k for k in res if k[0] == 1.0), key=lambda k: res[k][fit])
    print(f"fit on half {fit}: best w={best[0]} tau={best[1]} -> half {ev}: {res[best][ev]:.5f}  "
          f"| LightGBM only (tau={base[1]}): {res[base][ev]:.5f}  | gain {res[best][ev] - res[base][ev]:+.5f}")
print("\nfull check set, by blend weight (w=1: LightGBM only) at each tau:")
tab = pl.DataFrame([{"w": w, "tau": t, "ALL": v["all"], "US": v.get("US"), "India": v.get("India")} for (w, t), v in res.items()])
with pl.Config(tbl_rows=60, float_precision=5):
    print(tab.sort("ALL", descending=True).head(12))
    print(tab.filter(pl.col("w") == 1.0).sort("tau"))
