"""Milestone score on the lockbox (train S1 disjoint from dev; see STATUS.md for the protocol).

Only lockbox S1 are scored (scoring against the full truth would count every other S1 as an
empty prediction). Nothing is fit here: the models, tau and blocker were frozen beforehand.
The dev OOF at the same tau is scored by the same code for a like-for-like comparison.

    python -m src.eval.lockbox --pred output/lockbox/matching_results.tsv --cand artifacts/cand/lockbox \
        --oof artifacts/model/oof_dev_frozen.parquet --tau 0.75
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import polars as pl

from src.block import read_cand
from src.eval.blocking_recall import truth_pairs
from src.eval.oracle import oracle_table
from src.eval.score import f_beta_sets
from src.io_utils import encode_id, read_id_list_file
from src.prep import READ_KW


def per_s1(pred: dict, truth: pl.DataFrame, ids: pl.DataFrame) -> pl.DataFrame:
    """(s1_eid, country, n_true, f, tp, n_pred) for every S1 in ids. pred: {s1_eid: set(cand_eid)}."""
    t = truth.group_by("s1_eid").agg(pl.col("cand_eid"))
    tdict = dict(zip(t["s1_eid"].to_list(), t["cand_eid"].to_list()))
    rows = []
    for e in ids["eid"].to_list():
        p, tr = pred.get(e, set()), set(tdict.get(e, ()))
        rows.append((e, len(tr), f_beta_sets(p, tr), len(p & tr), len(p)))
    df = pl.DataFrame(rows, schema=["s1_eid", "n_true", "f", "tp", "n_pred"], orient="row")
    return df.join(ids.rename({"eid": "s1_eid"}), on="s1_eid")


def summary(d: pl.DataFrame, rng: np.random.Generator, boot: int = 2000) -> dict:
    f = d["f"].to_numpy()
    bs = np.array([f[rng.integers(0, len(f), len(f))].mean() for _ in range(boot)])
    sing = d.filter(pl.col("n_true") == 0)["f"]
    multi = d.filter(pl.col("n_true") > 0)["f"]
    return {"n": d.height, "macro_f05": f.mean(), "ci_lo": np.quantile(bs, 0.025), "ci_hi": np.quantile(bs, 0.975),
            "singleton_f05": sing.mean(), "multi_f05": multi.mean(),
            "micro_p": d["tp"].sum() / max(d["n_pred"].sum(), 1), "micro_r": d["tp"].sum() / max(d["n_true"].sum(), 1),
            "empty_share_pred": (d["n_pred"] == 0).mean(), "singleton_share": (d["n_true"] == 0).mean()}


def report(name: str, d: pl.DataFrame, orc: pl.DataFrame | None, mix: dict, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [{"set": name, "country": "ALL", **summary(d, rng)}]
    for c in sorted(d["country"].unique().to_list()):
        rows.append({"set": name, "country": c, **summary(d.filter(pl.col("country") == c), rng)})
    out = pl.DataFrame(rows)
    if orc is not None:
        o = pl.concat([orc.select(pl.lit("ALL").alias("country"), "f"), orc.select("country", "f")])
        out = out.join(o.group_by("country").agg(oracle=pl.col("f").mean()), on="country", how="left")
    # combined score reweighted to the US/India mix of test S1 (bootstrap per country, independent)
    by = {c: d.filter(pl.col("country") == c)["f"].to_numpy() for c in mix}
    w = np.array([mix[c] for c in mix]) / sum(mix.values())
    point = float(sum(wi * by[c].mean() for wi, c in zip(w, mix)))
    bs = np.array([sum(wi * by[c][rng.integers(0, len(by[c]), len(by[c]))].mean() for wi, c in zip(w, mix))
                   for _ in range(2000)])
    rw = pl.DataFrame([{"set": name, "country": "test-mix", "n": d.height, "macro_f05": point,
                        "ci_lo": float(np.quantile(bs, 0.025)), "ci_hi": float(np.quantile(bs, 0.975))}])
    return pl.concat([out, rw], how="diagonal_relaxed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="matching_results.tsv restricted to lockbox S1")
    ap.add_argument("--cand", required=True, help="lockbox candidate dir (oracle ceiling)")
    ap.add_argument("--ids", default="artifacts/splits/lockbox_s1.parquet")
    ap.add_argument("--oof", default=None, help="dev OOF parquet (s1_eid, cand_eid, p) for comparison")
    ap.add_argument("--dev-ids", default="artifacts/splits/dev_s1.parquet")
    ap.add_argument("--dev-cand", default="artifacts/cand/train_dev_v3h1")
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--truth", default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--test-s1", default="dataset/test/test_source1.tsv", help="label-free country counts")
    ap.add_argument("--loco", default="0.9527,0.9098", help="LOCO US, LOCO India-Latin F0.5 (France proxy)")
    ap.add_argument("--out", default="reports/lockbox_milestone.md")
    ap.add_argument("--tag", default="baseline (stage 1, tau 0.75)")
    args = ap.parse_args()

    tp = truth_pairs(args.truth)
    cnt = (pl.read_csv(args.test_s1, **READ_KW, columns=["country"]).with_columns(pl.col("country").str.strip_chars())
             ["country"].value_counts())
    mix = {c: n for c, n in zip(cnt["country"].to_list(), cnt["count"].to_list()) if c in ("US", "India")}

    ids = pl.read_parquet(args.ids).select("eid", "country")
    raw = read_id_list_file(args.pred)
    pred = {encode_id(s): {encode_id(x) for x in v} for s, v in raw.items()}
    assert set(pred) == set(ids["eid"].to_list()), "prediction file must cover exactly the lockbox S1"
    t_box = tp.join(ids.select(pl.col("eid").alias("s1_eid")), on="s1_eid")
    orc = oracle_table(read_cand(args.cand), t_box, ids)
    tables = [report("lockbox", per_s1(pred, t_box, ids), orc, mix, seed=1)]

    if args.oof:
        dids = pl.read_parquet(args.dev_ids).select("eid", "country")
        oof = pl.read_parquet(args.oof).filter(pl.col("p") >= args.tau).group_by("s1_eid").agg(pl.col("cand_eid"))
        dpred = {e: set(c) for e, c in zip(oof["s1_eid"].to_list(), oof["cand_eid"].to_list())}
        t_dev = tp.join(dids.select(pl.col("eid").alias("s1_eid")), on="s1_eid")
        dorc = oracle_table(read_cand(args.dev_cand), t_dev, dids)
        tables.append(report("dev OOF", per_s1(dpred, t_dev, dids), dorc, mix, seed=2))

    res = pl.concat(tables, how="diagonal_relaxed")
    lu, li = (float(x) for x in args.loco.split(","))
    with pl.Config(tbl_rows=20, tbl_cols=20, ascii_tables=True, float_precision=4, tbl_width_chars=250):
        print(res)
    fmt = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.4f}" if isinstance(v, float) else str(v))
    cols = ["set", "country", "n", "macro_f05", "ci_lo", "ci_hi", "singleton_f05", "multi_f05", "micro_p",
            "micro_r", "oracle", "empty_share_pred", "singleton_share"]
    lines = [f"# Lockbox milestone: {args.tag}", "",
             f"tau {args.tau}; models frozen on dev folds; lockbox {ids.height:,} S1. Bootstrap 95% CI over S1 (2000).",
             f"test-mix = US/India weighted by test_source1 counts {mix}.",
             f"France estimate (LOCO Latin view, mean of US-LOCO {lu} and India-Latin-LOCO {li}): {(lu + li) / 2:.4f}", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in res.iter_rows(named=True):
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n\n")
    print(f"appended to {args.out}")


if __name__ == "__main__":
    main()
