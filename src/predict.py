"""End-to-end inference for one split: blocker parts -> features -> fold-averaged stage-1 score
-> decision -> submission files.

    python -m src.block   --split test                        # artifacts/cand/test/part-*.parquet
    python -m src.predict --split test --models 'artifacts/model/model_dev_frozen_*.txt' --tau 0.75
    python -m src.predict --split train --cand artifacts/cand/lockbox --s1-ids artifacts/splits/lockbox_s1.parquet \
        --models 'artifacts/model/model_dev_frozen_*.txt' --tau 0.75 --out output/lockbox

Writes
  artifacts/score/<cand dir name>/part-*.parquet   (s1_eid, cand_eid, src, p) for every scored pair
  output/matching_results.tsv              S1 -> matches with p >= tau
  output/candidate_pairs.tsv               S1 -> EXACTLY the pairs the model scored
Every S1 of <split>_source1.tsv gets one row, in file order. Parameters (models, tau) come from
train OOF only; nothing here is fit on the split being predicted.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from src.features import build_part, token_df
from src.io_utils import ID_BASE
from src.prep import READ_KW, encode_id_expr


def load_models(pattern: str):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    models = [lgb.Booster(model_file=p) for p in paths]
    names = models[0].feature_name()
    assert all(m.feature_name() == names for m in models), "fold models disagree on features"
    return models, names


def score(models, names, f: pl.DataFrame) -> np.ndarray:
    X = f.select(names).to_numpy().astype(np.float32)
    return np.mean([m.predict(X) for m in models], axis=0).astype(np.float32)


def decode(col: str) -> pl.Expr:
    c = pl.col(col)
    return pl.concat_str([pl.lit("S"), (c // ID_BASE).cast(pl.Utf8), pl.lit("-"), (c % ID_BASE).cast(pl.Utf8)])


def _lists(pairs: pl.DataFrame) -> pl.DataFrame:
    """(s1_eid, ids): ids comma-joined in descending score order."""
    return (pairs.sort(["s1_eid", "p"], descending=[False, True])
                 .group_by("s1_eid", maintain_order=True).agg(ids=decode("cand_eid").str.join(",")))


def write_outputs(out: str, s1_order: pl.DataFrame, scores: pl.LazyFrame, tau: float,
                  batch: int = 200_000) -> None:
    """matching_results.tsv (p >= tau) and candidate_pairs.tsv (every scored pair): one row per
    S1 in file order. Streams over S1 batches, so the full split's pairs are never in memory."""
    os.makedirs(out, exist_ok=True)
    files = {"matched_entity_ids": os.path.join(out, "matching_results.tsv"),
             "candidate_entity_ids": os.path.join(out, "candidate_pairs.tsv")}
    handles = {h: open(path, "w", encoding="utf-8", newline="\n") for h, path in files.items()}
    try:
        for h, fh in handles.items():
            fh.write(f"source1_entity_id\t{h}\n")
        for lo in range(0, s1_order.height, batch):
            b = s1_order.slice(lo, batch).select("entity_id", "s1_eid")
            sc = scores.join(b.lazy().select("s1_eid"), on="s1_eid", how="semi").collect()
            for h, pairs in (("matched_entity_ids", sc.filter(pl.col("p") >= tau)),
                             ("candidate_entity_ids", sc)):
                rows = (b.join(_lists(pairs), on="s1_eid", how="left", maintain_order="left")
                         .select("entity_id", pl.col("ids").fill_null("")))
                handles[h].write(rows.write_csv(separator="\t", quote_style="never",
                                                include_header=False, line_terminator="\n"))
            del sc
    finally:
        for fh in handles.values():
            fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--norm", default="artifacts/norm")
    ap.add_argument("--cand", default=None, help="default artifacts/cand/<split>")
    ap.add_argument("--s1-ids", default=None, help="parquet with eid: restrict S1 (train lockbox)")
    ap.add_argument("--models", required=True, help="glob of fold models, averaged")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--rescore", action="store_true", help="recompute scores even if cached")
    ap.add_argument("--max-parts", type=int, default=0,
                    help="score at most N new parts, then exit with code 3 (run in a loop: memory "
                         "is not returned to the OS between parts, so long runs start swapping)")
    args = ap.parse_args()
    lg = lambda m: print(m, flush=True)
    t0 = time.time()

    cand_dir = args.cand or os.path.join("artifacts", "cand", args.split)
    score_dir = os.path.join("artifacts", "score", os.path.basename(os.path.normpath(cand_dir)))
    os.makedirs(score_dir, exist_ok=True)
    models, names = load_models(args.models)
    idf = token_df(args.norm, args.split)
    parts = sorted(glob.glob(os.path.join(cand_dir, "part-*.parquet")))
    lg(f"{args.split}: {len(parts)} candidate parts, {len(models)} fold models, {len(names)} features")
    done_now = 0
    for i, path in enumerate(parts):
        dst = os.path.join(score_dir, os.path.basename(path))
        if os.path.exists(dst) and not args.rescore:
            continue
        if args.max_parts and done_now >= args.max_parts:
            lg(f"  stopping after {done_now} new parts (--max-parts); re-run to continue")
            raise SystemExit(3)
        done_now += 1
        cand = pl.read_parquet(path)
        f = build_part(args.norm, args.split, cand, idf)
        (f.select("s1_eid", "cand_eid", "src", "script_b").with_columns(p=pl.Series(score(models, names, f)))
          .write_parquet(dst + ".tmp"))
        os.replace(dst + ".tmp", dst)  # a crash never leaves a half-written part that looks done
        lg(f"  part {i + 1}/{len(parts)}: {cand.height:,} pairs ({time.time() - t0:.0f}s)")
        del cand, f
    del idf

    scores = pl.scan_parquet(os.path.join(score_dir, "part-*.parquet"))
    s1_order = (pl.read_csv(os.path.join(args.data, args.split, f"{args.split}_source1.tsv"), **READ_KW,
                            columns=["entity_id", "country"])
                  .with_columns(s1_eid=encode_id_expr("entity_id"), country=pl.col("country").fill_null("").str.strip_chars()))
    if args.s1_ids:
        s1_order = s1_order.join(pl.read_parquet(args.s1_ids).select(pl.col("eid").alias("s1_eid")),
                                 on="s1_eid", how="semi", maintain_order="left")
    write_outputs(args.out, s1_order, scores, args.tau)
    lg(f"  outputs written ({time.time() - t0:.0f}s)")

    # sanity statistics per country (compare the unseen country with train OOF: empty ~0.06, ~3.4/S1)
    s1c = s1_order.select("s1_eid", "country").lazy()
    per_s1 = (scores.group_by("s1_eid").agg(n=(pl.col("p") >= args.tau).sum(), n_cand=pl.len(),
                                            n_native=(pl.col("script_b") == 2).sum())
                    .collect(engine="streaming"))
    st = (s1c.join(per_s1.lazy(), on="s1_eid", how="left").fill_null(0)
             .group_by("country").agg(s1=pl.len(), empty_share=(pl.col("n") == 0).mean(),
                                      matches_per_s1=pl.col("n").mean(), p95=pl.col("n").quantile(0.95),
                                      cand_per_s1=pl.col("n_cand").mean(),
                                      pred_rate=pl.col("n").sum() / pl.col("n_cand").sum(),
                                      native_share=pl.col("n_native").sum() / pl.col("n_cand").sum())
             .sort("s1", descending=True).collect())
    with pl.Config(tbl_rows=20, tbl_cols=20, ascii_tables=True):
        lg(st)
    st.write_csv(os.path.join(args.out, "prediction_stats.csv"))
    n_scored, n_matched = int(per_s1["n_cand"].sum()), int(per_s1["n"].sum())
    with open(os.path.join(args.out, "run_meta.json"), "w") as fh:
        json.dump({"split": args.split, "tau": args.tau, "models": sorted(glob.glob(args.models)),
                   "cand": cand_dir, "pairs_scored": n_scored, "pairs_matched": n_matched,
                   "s1": s1_order.height, "seconds": round(time.time() - t0)}, fh, indent=1)
    lg(f"done: {n_scored:,} scored, {n_matched:,} matched, {s1_order.height:,} S1 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
