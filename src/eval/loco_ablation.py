"""Feature-set ablation for the unseen country: pooled OOF vs leave-one-country-out.

For every variant we fit
  pooled  5-fold S1-grouped OOF on both countries           (in-distribution: US, India on test)
  LOCO    train on one country, score the other              (proxy for France)
and score each at the frozen threshold tau with exact macro F0.5. India is scored on its
Latin view for LOCO (native-script candidates removed from candidates AND truth), because
that is the part of the shift that can apply to France.

    expected = (1 - f) * pooled + f * LOCO,   LOCO = mean(US-LOCO, India-Latin-LOCO)

f = France share of test S1 (a label-free count of test_source1.tsv, passed in; nothing
else is read from test). Per-S1 F0.5 vectors are cached per variant so a crash resumes and
the paired bootstrap (same resampled S1 for both variants) needs no refit.

    python -m src.eval.loco_ablation --feat artifacts/feat/train_dev_frozen.parquet \
        --s1-ids artifacts/splits/dev_s1.parquet --france-share 0.1498
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import polars as pl

from src.eval.blocking_recall import truth_pairs
from src.eval.decide_oof import truth_dict
from src.eval.score import f_beta_sets
from src.features import SCALED
from src.prep import load
from src.train import EXCLUDE, ID_COLS, fit, s1_folds

IDF_PREFIX = ("name_w", "addr_w", "name_unm", "addr_unm")
STRUCT = ["ad_n_a", "ad_n_b", "num_n_a", "num_n_b", "lnum_n_a", "lnum_n_b", "nm_len_a", "nm_len_b",
          "nt_n_a", "nt_n_b"]
SCRIPT = ["script_a", "script_b", "script_diff"]
RAWISH = SCALED + ["s1_gap"]  # s1_gap is a difference of s_all, so it carries the same scale


def variants(feat: pl.DataFrame, base: list, lnN: pl.DataFrame):
    """name -> (extra columns to add to feat, feature list). Base features are already / ln N."""
    idf = [c for c in base if c.startswith(IDF_PREFIX)]
    postal = [c for c in base if c.startswith("lnum_")]
    raw = {f"{c}__raw": pl.col(c) * pl.col("lnN") for c in RAWISH}
    # within-country percentile is scale-free, so it is the same for raw and / ln N inputs
    pct = {f"{c}__pct": pl.col(c).rank("average").over("country") / pl.len().over("country") for c in RAWISH}
    drop = lambda cols: [c for c in base if c not in cols]
    return {
        "current (/lnN)": ({}, base),
        "raw idf sums": (raw, drop(RAWISH) + list(raw)),
        "within-country pct": (pct, drop(RAWISH) + list(pct)),
        "-idf feats": ({}, drop(idf)),
        "-addr/len struct": ({}, drop(STRUCT)),
        "-postal": ({}, drop(postal)),
        "-script": ({}, drop(SCRIPT)),
        "-idf -struct": ({}, drop(idf + STRUCT)),
        "pct -struct": (pct, drop(RAWISH + STRUCT) + list(pct)),
    }


def per_s1_f(s1: np.ndarray, cand: np.ndarray, p: np.ndarray, ids: list, truth: dict, tau: float) -> np.ndarray:
    """Exact per-S1 F0.5 of the threshold rule (p >= tau), in the order of `ids`."""
    pred = {s: [] for s in ids}
    for a, b in zip(s1[p >= tau], cand[p >= tau]):
        pred[a].append(b)
    return np.array([f_beta_sets(pred[s], truth.get(s, ())) for s in ids], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--s1-ids", required=True)
    ap.add_argument("--truth", default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--idf", default="artifacts/stats/train_idf.parquet")
    ap.add_argument("--france-share", type=float, required=True)
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--cache", default="artifacts/ablation")
    ap.add_argument("--out", default="reports/loco_ablation_full.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    lg = lambda m: print(m, flush=True)
    os.makedirs(args.cache, exist_ok=True)

    ids_df = pl.read_parquet(args.s1_ids).select("eid", "country")
    folds = s1_folds(ids_df, args.truth, 5, 42)
    lnN = pl.read_parquet(args.idf).group_by("country").agg(lnN=pl.col("idf").max())
    feat = pl.read_parquet(args.feat).join(folds, on="s1_eid").join(lnN, on="country")
    base = [c for c in feat.columns if c not in ID_COLS + EXCLUDE + ["fold", "country", "lnN"]]

    ids = ids_df["eid"].to_list()
    t_all = truth_dict(args.truth, ids)
    tp = truth_pairs(args.truth).join(pl.DataFrame({"s1_eid": ids}), on="s1_eid")
    sc = pl.concat([load("artifacts/norm", "train", k, ["eid", "script"]) for k in (2, 3)]).join(
        tp.select(pl.col("cand_eid").alias("eid")).unique(), on="eid", how="semi")
    native = set(sc.filter(pl.col("script") == 2)["eid"].to_list())
    t_lat = {s: [c for c in v if c not in native] for s, v in t_all.items()}
    id_us = ids_df.filter(pl.col("country") == "US")["eid"].to_list()
    id_in = ids_df.filter(pl.col("country") == "India")["eid"].to_list()

    y, g = feat["y"].to_numpy(), feat["s1_eid"].to_numpy()
    s1, cand = g, feat["cand_eid"].to_numpy()
    ctry, fold = feat["country"].to_numpy(), feat["fold"].to_numpy()
    latin = ~feat["cand_eid"].is_in(list(native)).to_numpy()

    V = variants(feat, base, lnN)
    res = {}
    for name, (extra, cols) in V.items():
        path = os.path.join(args.cache, "".join(ch if ch.isalnum() else "_" for ch in name) + ".npz")
        if os.path.exists(path):
            res[name] = dict(np.load(path))
            lg(f"{name}: cached")
            continue
        t0 = time.time()
        X = (feat.with_columns(**extra) if extra else feat).select(cols).to_numpy().astype(np.float32)
        pooled = np.full(len(y), np.nan, dtype=np.float32)
        for k in np.unique(fold):
            te = fold == k
            m = fit(X[~te], y[~te], g[~te], cols)
            pooled[te] = m.predict(X[te], num_iteration=m.best_iteration)
        loco = np.full(len(y), np.nan, dtype=np.float32)
        for held in ("US", "India"):
            te = ctry == held
            m = fit(X[~te], y[~te], g[~te], cols)
            loco[te] = m.predict(X[te], num_iteration=m.best_iteration)
        del X
        il = (ctry == "India") & latin
        r = {"pooled": per_s1_f(s1, cand, pooled, ids, t_all, args.tau),
             "loco_us": per_s1_f(s1[ctry == "US"], cand[ctry == "US"], loco[ctry == "US"], id_us, t_all, args.tau),
             "loco_inlat": per_s1_f(s1[il], cand[il], loco[il], id_in, t_lat, args.tau),
             "loco_inall": per_s1_f(s1[ctry == "India"], cand[ctry == "India"], loco[ctry == "India"], id_in,
                                    t_all, args.tau)}
        np.savez(path, **r)
        res[name] = r
        lg(f"{name}: pooled {r['pooled'].mean():.4f}  LOCO US {r['loco_us'].mean():.4f}  "
           f"IN-Latin {r['loco_inlat'].mean():.4f} ({time.time() - t0:.0f}s)")

    f = args.france_share
    is_us = np.isin(np.array(ids), np.array(id_us))
    rng = np.random.default_rng(0)
    # paired bootstrap: one S1 resample per replicate, shared by every variant
    B = [(rng.integers(0, len(ids), len(ids)), rng.integers(0, len(id_us), len(id_us)),
          rng.integers(0, len(id_in), len(id_in))) for _ in range(args.n_boot)]

    def expected(r, b=None):
        if b is None:
            return (1 - f) * r["pooled"].mean() + f * (r["loco_us"].mean() + r["loco_inlat"].mean()) / 2
        return ((1 - f) * r["pooled"][b[0]].mean()
                + f * (r["loco_us"][b[1]].mean() + r["loco_inlat"][b[2]].mean()) / 2)

    ref = res["current (/lnN)"]
    ref_boot = np.array([expected(ref, b) for b in B])
    rows = []
    for name, r in res.items():
        d = np.array([expected(r, b) for b in B]) - ref_boot
        rows.append(dict(variant=name, pooled=r["pooled"].mean(), pooled_us=r["pooled"][is_us].mean(),
                         pooled_in=r["pooled"][~is_us].mean(), loco_us=r["loco_us"].mean(),
                         loco_inlat=r["loco_inlat"].mean(), loco_inall=r["loco_inall"].mean(),
                         expected=expected(r), d_vs_current=expected(r) - expected(ref),
                         ci_lo=np.quantile(d, 0.025), ci_hi=np.quantile(d, 0.975), p_better=(d > 0).mean()))
    tab = pl.DataFrame(rows)
    with pl.Config(tbl_rows=20, tbl_cols=20, ascii_tables=True, float_precision=4, tbl_width_chars=200):
        lg(tab)
    best = tab.sort("expected", descending=True).row(0, named=True)
    lines = [f"# Feature ablation: pooled OOF vs LOCO (dev, frozen blocker, tau {args.tau})", "",
             f"France share of test S1 f = {f:.4f}. expected = (1-f)*pooled + f*mean(LOCO US, LOCO India-Latin).",
             f"Paired bootstrap over S1 ({args.n_boot} replicates) of expected(variant) - expected(current).", "",
             "| variant | pooled | pooled US | pooled IN | LOCO US | LOCO IN-Latin | LOCO IN-all | expected | d vs current [95% CI] | P(better) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['pooled']:.4f} | {r['pooled_us']:.4f} | {r['pooled_in']:.4f} | "
                     f"{r['loco_us']:.4f} | {r['loco_inlat']:.4f} | {r['loco_inall']:.4f} | {r['expected']:.4f} | "
                     f"{r['d_vs_current']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_better']:.2f} |")
    lines += ["", f"Best by expected: **{best['variant']}** (d {best['d_vs_current']:+.4f}, "
                  f"95% CI [{best['ci_lo']:+.4f}, {best['ci_hi']:+.4f}]). Switch only if the CI excludes 0."]
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    lg(f"wrote {args.out}")


if __name__ == "__main__":
    main()
