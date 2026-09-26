"""Error taxonomy of a decision on OOF scores, in macro-F0.5 points that add up.

Gap decomposition over the evaluated S1 set
  1 - current  =  (1 - oracle)            blocking: true matches never retrieved
               +  (oracle - current)      matcher + decision, split by primary cause

Primary cause (exactly one per erroneous pair; FN = retrieved true match not predicted)
  singleton_fp     predicted, not true, S1 has no true match
  multi_fp         predicted, not true, S1 has >= 1 true match
  rejected         FN with p >= 0.5 (the model leaned "match", the decision rule dropped it)
  missed_extra     FN with p < 0.5 while the S1 has at least one correct prediction
  scored_low       FN with p < 0.5 and the S1 has no correct prediction at all

Loss attribution: per S1 the categories present are the players of a cooperative game
whose value is the S1's F0.5 after fixing that subset of categories; the exact Shapley
value of each category is its loss. Per S1 the values sum to oracle F - current F, so the
category totals sum to the matcher/decision gap exactly.

Secondary tags (overlapping, reported per pair, NOT additive), all from normalized views:
  same_name     identical name_core             same_addr   identical addr_n (non-empty)
  phon_collide  same phonetic skeleton, different name_core
  legal_collide identical name_core, different legal-form skeleton
  native        candidate name in a non-Latin script       src (2/3), country

    python -m src.eval.taxonomy --oof artifacts/model/oof_dev.parquet --s1-ids artifacts/splits/dev_s1.parquet --tau 0.75
"""
from __future__ import annotations

import argparse
from itertools import combinations
from math import factorial

import polars as pl

from src.block import phonetic
from src.eval.blocking_recall import truth_pairs
from src.eval.score import f_beta_sets
from src.features import _legal_skel, _tokens
from src.prep import load

CATS = ["singleton_fp", "multi_fp", "rejected", "missed_extra", "scored_low"]
FP_CATS = {"singleton_fp", "multi_fp"}
TAGS = ["same_name", "same_addr", "phon_collide", "legal_collide", "native"]


def label_errors(oof: pl.DataFrame, tp: pl.DataFrame, tau: float) -> pl.DataFrame:
    """One row per error pair (s1_eid, cand_eid, p, cat); threshold rule p >= tau."""
    pr = oof.with_columns(pred=pl.col("p") >= tau)
    tpp = tp.select("s1_eid", "cand_eid", truth=pl.lit(True))
    pr = pr.join(tpp, on=["s1_eid", "cand_eid"], how="left").with_columns(pl.col("truth").fill_null(False))
    ent = (tp.group_by("s1_eid").agg(n_true=pl.len())
             .join(pr.group_by("s1_eid").agg(n_tp=(pl.col("pred") & pl.col("truth")).sum()), on="s1_eid", how="left")
             .with_columns(pl.col("n_tp").fill_null(0)))
    pr = pr.join(ent, on="s1_eid", how="left").with_columns(pl.col("n_true", "n_tp").fill_null(0))
    return (pr.filter(pl.col("pred") != pl.col("truth"))
              .with_columns(cat=pl.when(pl.col("pred") & (pl.col("n_true") == 0)).then(pl.lit("singleton_fp"))
                              .when(pl.col("pred")).then(pl.lit("multi_fp"))
                              .when(pl.col("p") >= 0.5).then(pl.lit("rejected"))
                              .when(pl.col("n_tp") > 0).then(pl.lit("missed_extra"))
                              .otherwise(pl.lit("scored_low")))
              .select("s1_eid", "cand_eid", "p", "cat"))


def shapley_losses(pred: set, truth: set, errs: dict) -> dict:
    """errs: cat -> set of cand ids. Returns cat -> Shapley share of (F(fixed all) - F(pred))."""
    players = list(errs)
    n = len(players)

    def value(fixed):
        p = set(pred)
        for c in fixed:
            p = p - errs[c] if c in FP_CATS else p | errs[c]
        return f_beta_sets(p, truth)

    out = {}
    for c in players:
        others = [o for o in players if o != c]
        tot = 0.0
        for r in range(n):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for sub in combinations(others, r):
                tot += w * (value(sub + (c,)) - value(sub))
        out[c] = tot
    return out


def pair_tags(pairs: pl.DataFrame, norm: str) -> pl.DataFrame:
    cols = ["eid", "script", "name_n", "name_core", "addr_n"]
    a = load(norm, "train", 1, cols).join(pairs.select(pl.col("s1_eid").alias("eid")).unique(), on="eid", how="semi")
    b = pl.concat([load(norm, "train", k, cols) for k in (2, 3)]).join(
        pairs.select(pl.col("cand_eid").alias("eid")).unique(), on="eid", how="semi")
    d = (pairs.join(a.rename({"eid": "s1_eid"}), on="s1_eid")
              .join(b.rename({c: f"{c}_b" for c in cols if c != "eid"}).rename({"eid": "cand_eid"}), on="cand_eid"))
    ph = lambda c: _tokens(c).list.eval(phonetic(pl.element())).list.join(" ")
    return d.with_columns(
        same_name=pl.col("name_core") == pl.col("name_core_b"),
        same_addr=(pl.col("addr_n") == pl.col("addr_n_b")) & (pl.col("addr_n") != ""),
        phon_collide=(ph("name_core") == ph("name_core_b")) & (pl.col("name_core") != pl.col("name_core_b")),
        legal_collide=(pl.col("name_core") == pl.col("name_core_b"))
                      & (_legal_skel("name_n", "script") != _legal_skel("name_n_b", "script_b")),
        native=pl.col("script_b") == 2,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--s1-ids", required=True)
    ap.add_argument("--truth", default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--norm", default="artifacts/norm")
    ap.add_argument("--tau", type=float, default=0.75)
    args = ap.parse_args()

    ids = pl.read_parquet(args.s1_ids).select(pl.col("eid").alias("s1_eid"), "country")
    n = ids.height
    oof = pl.read_parquet(args.oof)
    tp = truth_pairs(args.truth).join(ids, on="s1_eid")
    errs = label_errors(oof, tp, args.tau)

    # per-S1 sets
    truth = {s: set() for s in ids["s1_eid"].to_list()}
    for a, b in zip(tp["s1_eid"].to_list(), tp["cand_eid"].to_list()):
        truth[a].add(b)
    retrieved = {}
    pred = {s: set() for s in truth}
    for a, b, p in zip(oof["s1_eid"].to_list(), oof["cand_eid"].to_list(), oof["p"].to_list()):
        retrieved.setdefault(a, set()).add(b)
        if p >= args.tau:
            pred[a].add(b)
    by = {}
    for a, b, c in zip(errs["s1_eid"].to_list(), errs["cand_eid"].to_list(), errs["cat"].to_list()):
        by.setdefault(a, {}).setdefault(c, set()).add(b)

    cur = sum(f_beta_sets(pred[s], truth[s]) for s in truth) / n
    orc = sum(f_beta_sets(truth[s] & retrieved.get(s, set()), truth[s]) for s in truth) / n
    loss_rows, pair_share = [], []
    for s, e in by.items():
        sh = shapley_losses(pred[s], truth[s], e)  # full truth: fixing every category gives the oracle F
        for c, v in sh.items():
            loss_rows.append((s, c, v))
            for b in e[c]:
                pair_share.append((s, b, v / len(e[c])))
    L = pl.DataFrame(loss_rows, schema=["s1_eid", "cat", "loss"], orient="row").join(ids, on="s1_eid")
    print(f"S1 {n:,}  current {cur:.4f}  oracle {orc:.4f}")
    print(f"gap 1 - current {1 - cur:.4f} = blocking {1 - orc:.4f} + matcher/decision {orc - cur:.4f}"
          f"  (sum of primary losses {L['loss'].sum() / n:.4f})")
    ascii_cfg = pl.Config(tbl_rows=40, tbl_cols=20, ascii_tables=True, tbl_width_chars=200)
    with ascii_cfg:
        cnt = errs.group_by("cat").agg(pairs=pl.len(), s1=pl.col("s1_eid").n_unique())
        print(L.group_by("cat").agg(loss_pts=(pl.col("loss").sum() / n * 100))
               .join(cnt, on="cat").sort("loss_pts", descending=True))
        print(L.group_by("cat", "country").agg(loss_pts=(pl.col("loss").sum() / n * 100))
               .pivot("country", index="cat", values="loss_pts").sort("cat"))
        # per-pair shares -> source and secondary tags (tags overlap: rows are not additive)
        P = pl.DataFrame(pair_share, schema=["s1_eid", "cand_eid", "loss"], orient="row")
        P = P.join(errs.select("s1_eid", "cand_eid", "cat", "p"), on=["s1_eid", "cand_eid"])
        P = P.join(oof.select("s1_eid", "cand_eid", "src").unique(), on=["s1_eid", "cand_eid"], how="left")
        print(P.group_by("src", "cat").agg(loss_pts=(pl.col("loss").sum() / n * 100))
               .pivot("src", index="cat", values="loss_pts").sort("cat"))
        T = pair_tags(P, args.norm)
        rows = []
        for t in TAGS:
            g = T.filter(pl.col(t))
            rows.append({"tag": t, "error_pairs": g.height, "loss_pts": g["loss"].sum() / n * 100,
                         **{c: g.filter(pl.col("cat") == c).height for c in CATS}})
        print(pl.DataFrame(rows))
        # hard-case evaluation over ALL candidates (not just errors): pair precision / recall
        allp = pair_tags(oof.select("s1_eid", "cand_eid", "p", "y"), args.norm).with_columns(pred=pl.col("p") >= args.tau)
        hc = []
        for t in TAGS:
            g = allp.filter(pl.col(t))
            tpn = (g["pred"] & (g["y"] == 1)).sum()
            hc.append({"case": t, "pairs": g.height, "pos_rate": g["y"].mean(),
                       "precision": tpn / max(g["pred"].sum(), 1), "recall": tpn / max(g["y"].sum(), 1),
                       "fp": int((g["pred"] & (g["y"] == 0)).sum()), "fn": int((~g["pred"] & (g["y"] == 1)).sum())})
        print(pl.DataFrame(hc))
        print("high-scoring FP (p >= tau, not true):", int(((allp["pred"]) & (allp["y"] == 0)).sum()))


if __name__ == "__main__":
    main()
