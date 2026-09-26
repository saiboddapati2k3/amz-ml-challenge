"""Phase 3a: pair features for blocker candidates (label-free; labels joined only on train).

    python -m src.features --split train --cand artifacts/cand/train_dev_v3.parquet --tag train_dev
    python -m src.features --split test  --cand artifacts/cand/test.parquet --tag test

Feature groups (every one is script/country agnostic -- no country id is ever a feature):
  name   fuzzy scores on core / compact / phonetic views, alias-aware best, idf-weighted
         token overlap, rarest unmatched token, legal-form agreement, numbers in the name
  addr   fuzzy scores, idf-weighted overlap, house-number and long-number (postal-like)
         agreement / conflict, address length (short S2/S3 addresses are common)
  block  blocker family scores and ranks, gap to the S1's best candidate
  comp   competition on the candidate side: how many S1 list it, rank of this S1 among them
         (the one-owner signal; only meaningful when the candidate file covers the whole split)

Token idf comes from token document frequencies over S1+S2+S3 of the same split and
country (label-free, identical procedure on train and test), cached in artifacts/stats.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from src.block import PHON_STOP, phonetic, read_cand
from src.prep import LEGAL_WORDS, load

REC_COLS = ["eid", "country", "script", "name_n", "name_core", "name_alias", "name_web",
            "addr_n", "addr_nums"]
BLOCK_COLS = ["s_name", "s_addr", "s_mix", "s_all", "n_fam", "r_all", "r_name", "r_addr"]
# idf-scale features divided by the country's ln N (s1_gap follows from s_all)
SCALED = ["s_name", "s_addr", "s_mix", "s_all", "name_unm_a", "name_unm_b", "addr_unm_a",
          "addr_unm_b", "name_wcommon", "addr_wcommon"]


# --------------------------------------------------------------------------- #
# corpus statistics
# --------------------------------------------------------------------------- #
def token_df(norm: str, split: str, out_dir: str = "artifacts/stats") -> pl.DataFrame:
    """(country, kind, t, idf) for name_core and addr_n tokens; cached per split."""
    path = os.path.join(out_dir, f"{split}_idf.parquet")
    if os.path.exists(path):
        return pl.read_parquet(path)
    parts, n_docs = [], []
    for kind, col in (("name", "name_core"), ("addr", "addr_n")):
        for src in (1, 2, 3):
            lf = pl.scan_parquet(os.path.join(norm, f"{split}_s{src}", "*.parquet"))
            if kind == "name":
                n_docs.append(lf.group_by("country").agg(n=pl.len()).collect())
            parts.append(lf.select("country", t=pl.col(col).str.split(" ").list.unique())
                           .explode("t").filter(pl.col("t").str.len_chars() > 0)
                           .group_by("country", "t").agg(df=pl.len())
                           .with_columns(kind=pl.lit(kind)).collect(engine="streaming"))
    n = pl.concat(n_docs).group_by("country").agg(pl.col("n").sum())
    idf = (pl.concat(parts).group_by("country", "kind", "t").agg(pl.col("df").sum())
             .join(n, on="country")
             .select("country", "kind", "t",
                     idf=(pl.col("n").cast(pl.Float64) / pl.col("df")).log().cast(pl.Float32)))
    os.makedirs(out_dir, exist_ok=True)
    idf.write_parquet(path)
    return idf


# --------------------------------------------------------------------------- #
# pair features
# --------------------------------------------------------------------------- #
def _cp(a: pl.Series, b: pl.Series, scorer, **kw) -> np.ndarray:
    return cpdist(a.to_list(), b.to_list(), scorer=scorer, workers=-1, dtype=np.float32, **kw)


def _tokens(col: str) -> pl.Expr:
    return pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() > 0))


def _phon_str(col: str) -> pl.Expr:
    return _tokens(col).list.eval(phonetic(pl.element())).list.join(" ")


def idf_overlap(p: pl.DataFrame, idf: pl.DataFrame, kind: str, ca: str, cb: str) -> pl.DataFrame:
    """Per pid: idf-weighted overlap ratios and the rarest unmatched token on each side."""
    w = idf.filter(pl.col("kind") == kind).drop("kind")
    side = lambda c: (p.select("pid", "country", t=_tokens(c).list.unique()).explode("t")
                       .drop_nulls("t").join(w, on=["country", "t"], how="left")
                       .with_columns(pl.col("idf").fill_null(w["idf"].max() or 10.0)))
    a, b = side(ca), side(cb)
    common = a.join(b.select("pid", "t"), on=["pid", "t"], how="semi")
    agg = lambda d, name: d.group_by("pid").agg(pl.col("idf").sum().alias(name))
    um = lambda d, o, name: (d.join(o.select("pid", "t"), on=["pid", "t"], how="anti")
                              .group_by("pid").agg(pl.col("idf").max().alias(name)))
    out = (p.select("pid").join(agg(a, "wa"), on="pid", how="left").join(agg(b, "wb"), on="pid", how="left")
            .join(agg(common, "wc"), on="pid", how="left")
            .join(um(a, b, f"{kind}_unm_a"), on="pid", how="left").join(um(b, a, f"{kind}_unm_b"), on="pid", how="left")
            .with_columns(pl.col("wa", "wb", "wc").fill_null(0.0), pl.col(f"{kind}_unm_a", f"{kind}_unm_b").fill_null(0.0)))
    eps = 1e-6
    return out.select(
        "pid", f"{kind}_unm_a", f"{kind}_unm_b",
        (pl.col("wc") / (pl.min_horizontal("wa", "wb") + eps)).alias(f"{kind}_wov_min"),
        (pl.col("wc") / (pl.max_horizontal("wa", "wb") + eps)).alias(f"{kind}_wov_max"),
        (pl.col("wc") / (pl.col("wa") + pl.col("wb") - pl.col("wc") + eps)).alias(f"{kind}_wjac"),
        pl.col("wc").alias(f"{kind}_wcommon"),
    )


def _set_feats(a: str, b: str, pre: str) -> list:
    ta, tb = _tokens(a), _tokens(b)
    inter = ta.list.set_intersection(tb).list.len()
    la, lb = ta.list.len(), tb.list.len()
    return [
        inter.alias(f"{pre}_common"),
        la.alias(f"{pre}_n_a"), lb.alias(f"{pre}_n_b"),
        ((la > 0) & (lb > 0) & (inter == 0)).cast(pl.Int8).alias(f"{pre}_conflict"),
        (inter / pl.max_horizontal(pl.min_horizontal(la, lb), 1)).alias(f"{pre}_ov_min"),
    ]


legal_list = sorted(LEGAL_WORDS)
PHON_LEGAL = (pl.DataFrame({"w": sorted(LEGAL_WORDS | {"pvt", "ltd", "pte"})})
              .select(phonetic(pl.col("w")))["w"].unique().to_list())


def _script_core(core: str, script: str) -> pl.Expr:
    """Transliterated (non-Latin) names keep phonetic spellings of legal / noise words
    ('praivet limited'); drop tokens whose skeleton is a stop skeleton. Latin names are
    left alone ('prabhat' shares the skeleton of 'private')."""
    kept = _tokens(core).list.eval(pl.element().filter(~phonetic(pl.element()).is_in(PHON_STOP))).list.join(" ")
    return (pl.when((pl.col(script) == 2) & (kept != "")).then(kept).otherwise(pl.col(core)))


def _legal_skel(name_n: str, script: str) -> pl.Expr:
    """Sorted unique phonetic skeletons of legal-form tokens (comparable across scripts)."""
    sk = _tokens(name_n).list.eval(phonetic(pl.element()))
    return (pl.when(pl.col(script) == 2)
              .then(sk.list.eval(pl.element().filter(pl.element().is_in(PHON_LEGAL))))
              .otherwise(_tokens(name_n).list.eval(pl.element().filter(pl.element().is_in(legal_list)))
                         .list.eval(phonetic(pl.element())))
              .list.unique().list.sort().list.join(" "))



def pair_features(p: pl.DataFrame, idf: pl.DataFrame) -> pl.DataFrame:
    """p: candidate rows joined with record columns (suffix _b = S2/S3 side)."""
    p = p.with_row_index("pid").with_columns(
        name_core=_script_core("name_core", "script"), name_core_b=_script_core("name_core_b", "script_b"))
    long_nums = lambda c: _tokens(c).list.eval(pl.element().filter(pl.element().str.len_chars() >= 5)).list.join(" ")
    name_nums = lambda c: pl.col(c).str.extract_all(r"\d+").list.join(" ")
    p = p.with_columns(
        comp_a=pl.col("name_core").str.replace_all(" ", ""), comp_b=pl.col("name_core_b").str.replace_all(" ", ""),
        ph_a=_phon_str("name_core"), ph_b=_phon_str("name_core_b"),
        lg_a=_legal_skel("name_n", "script"), lg_b=_legal_skel("name_n_b", "script_b"),
        ln_a=long_nums("addr_nums"), ln_b=long_nums("addr_nums_b"),
        nn_a=name_nums("name_core"), nn_b=name_nums("name_core_b"),
    )
    A, B = p["name_core"], p["name_core_b"]
    f = {
        "nm_ratio": _cp(A, B, fuzz.ratio),
        "nm_tset": _cp(A, B, fuzz.token_set_ratio),
        "nm_tsort": _cp(A, B, fuzz.token_sort_ratio),
        "nm_partial": _cp(A, B, fuzz.partial_ratio),
        "nm_jw": _cp(A, B, JaroWinkler.normalized_similarity),
        "nm_comp_jw": _cp(p["comp_a"], p["comp_b"], JaroWinkler.normalized_similarity),
        "nm_comp_partial": _cp(p["comp_a"], p["comp_b"], fuzz.partial_ratio),
        "nm_ph_ratio": _cp(p["ph_a"], p["ph_b"], fuzz.ratio),
        "nm_ph_tset": _cp(p["ph_a"], p["ph_b"], fuzz.token_set_ratio),
        "nm_full_tset": _cp(p["name_n"], p["name_n_b"], fuzz.token_set_ratio),
        "ad_tset": _cp(p["addr_n"], p["addr_n_b"], fuzz.token_set_ratio),
        "ad_partial_tset": _cp(p["addr_n"], p["addr_n_b"], fuzz.partial_token_set_ratio),
        "ad_ratio": _cp(p["addr_n"], p["addr_n_b"], fuzz.ratio),
    }
    # alias-aware best name score (a/k/a, dba segments on either side)
    alias_a = p["name_alias"].fill_null("")
    alias_b = p["name_alias_b"].fill_null("")
    s1 = _cp(alias_a, B, fuzz.token_set_ratio) * (alias_a.str.len_chars() > 0).to_numpy()
    s2 = _cp(A, alias_b, fuzz.token_set_ratio) * (alias_b.str.len_chars() > 0).to_numpy()
    f["nm_alias_best"] = np.maximum(f["nm_tset"], np.maximum(s1, s2))

    out = p.select(
        "pid", "s1_eid", "cand_eid", *BLOCK_COLS,
        src=pl.col("src_b"),
        script_a=pl.col("script"), script_b=pl.col("script_b"),
        script_diff=(pl.col("script") != pl.col("script_b")).cast(pl.Int8),
        web_any=pl.max_horizontal("name_web", "name_web_b"),
        lg_both=((pl.col("lg_a") != "") & (pl.col("lg_b") != "")).cast(pl.Int8),
        lg_same=(pl.col("lg_a") == pl.col("lg_b")).cast(pl.Int8),
        lg_conflict=((pl.col("lg_a") != "") & (pl.col("lg_b") != "") & (pl.col("lg_a") != pl.col("lg_b"))).cast(pl.Int8),
        nm_first_eq=(_tokens("name_core").list.first() == _tokens("name_core_b").list.first()).fill_null(False).cast(pl.Int8),
        nm_len_a=pl.col("name_core").str.len_chars(), nm_len_b=pl.col("name_core_b").str.len_chars(),
        *_set_feats("name_core", "name_core_b", "nt"),
        *_set_feats("addr_nums", "addr_nums_b", "num"),
        *_set_feats("ln_a", "ln_b", "lnum"),
        *_set_feats("nn_a", "nn_b", "nnum"),
        ad_n_b=_tokens("addr_n_b").list.len(), ad_n_a=_tokens("addr_n").list.len(),
        num_first_eq=(_tokens("addr_nums").list.first() == _tokens("addr_nums_b").list.first()).fill_null(False).cast(pl.Int8),
    ).with_columns(**{k: pl.Series(v) for k, v in f.items()})
    for kind, a, b in (("name", "name_core", "name_core_b"), ("addr", "addr_n", "addr_n_b")):
        out = out.join(idf_overlap(p.select("pid", "country", a, b), idf, kind, a, b), on="pid", how="left")
    # Raw idf sums grow with ln(corpus size) and differ by country; divide by the country's
    # ln N (max idf = ln N / 1) so they transfer to an unseen country (LOCO: + on both folds).
    lnN = idf.group_by("country").agg(lnN=pl.col("idf").max())
    out = (out.join(p.select("pid", "country"), on="pid").join(lnN, on="country", how="left")
              .with_columns([(pl.col(c) / pl.col("lnN")).cast(pl.Float32) for c in SCALED])
              .drop("country", "lnN"))
    return out.sort("pid").drop("pid")


def context_features(f: pl.DataFrame) -> pl.DataFrame:
    """Within-S1 and within-candidate blocker context (needs the whole candidate set)."""
    return f.with_columns(
        s1_n_cand=pl.len().over("s1_eid").cast(pl.UInt16),
        s1_gap=pl.col("s_all").max().over("s1_eid") - pl.col("s_all"),
        s1_ratio=pl.col("s_all") / pl.col("s_all").max().over("s1_eid"),
        s1_nm_gap=pl.col("nm_tset").max().over("s1_eid") - pl.col("nm_tset"),
        s1_nm_rank=pl.col("nm_tset").rank("min", descending=True).over("s1_eid").cast(pl.UInt16),
        cand_n_s1=pl.len().over("cand_eid").cast(pl.UInt16),
        cand_rank=pl.col("s_all").rank("min", descending=True).over("cand_eid").cast(pl.UInt16),
        cand_ratio=pl.col("s_all") / pl.col("s_all").max().over("cand_eid"),
        cand_nm_gap=pl.col("nm_tset").max().over("cand_eid") - pl.col("nm_tset"),
    )


def _scan(norm: str, split: str, src: int) -> pl.LazyFrame:
    return pl.scan_parquet(os.path.join(norm, f"{split}_s{src}", "*.parquet")).select(REC_COLS)


def build_part(norm: str, split: str, cand: pl.DataFrame, idf: pl.DataFrame,
               chunk: int = 1_500_000) -> pl.DataFrame:
    """Features for one candidate frame that holds COMPLETE S1 candidate lists (a blocker part).

    Only the records this part needs are read (lazy semi-join scans), so memory is bounded by
    the part, not by the split. Candidate-side context (cand_*) is per part here, which is why
    the stage-1 model excludes it (src.train.EXCLUDE).
    """
    s1_need = cand.select(pl.col("s1_eid").alias("eid")).unique()
    c_need = cand.select(pl.col("cand_eid").alias("eid")).unique()
    s1 = _scan(norm, split, 1).join(s1_need.lazy(), on="eid", how="semi").collect()
    s23 = pl.concat([_scan(norm, split, k).join(c_need.lazy(), on="eid", how="semi").collect()
                       .with_columns(src_b=pl.lit(k, pl.Int8)) for k in (2, 3)])
    s23 = s23.drop("country").rename({c: f"{c}_b" for c in REC_COLS if c not in ("eid", "country")})
    parts = []
    for lo in range(0, cand.height, chunk):
        p = (cand.slice(lo, chunk).join(s1.rename({"eid": "s1_eid"}), on="s1_eid")
                 .join(s23.rename({"eid": "cand_eid"}), on="cand_eid"))
        parts.append(pair_features(p, idf))
    return context_features(pl.concat(parts))


def build(norm: str, split: str, cand: pl.DataFrame, chunk: int = 1_500_000, log=print) -> pl.DataFrame:
    t0 = time.time()
    idf = token_df(norm, split)
    log(f"  idf: {idf.height:,} tokens ({time.time() - t0:.0f}s)")
    f = build_part(norm, split, cand, idf, chunk)
    log(f"  pairs {cand.height:,} ({time.time() - t0:.0f}s)")
    return f


def main():
    from src.eval.blocking_recall import truth_pairs

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--norm", default="artifacts/norm")
    ap.add_argument("--out", default="artifacts/feat")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--truth", default="dataset/train/train_ground_truth.tsv")
    args = ap.parse_args()
    lg = lambda m: print(m, flush=True)
    feats = build(args.norm, args.split, read_cand(args.cand), log=lg)
    if args.split == "train":
        tp = truth_pairs(args.truth).select("s1_eid", "cand_eid", y=pl.lit(1, pl.Int8))
        feats = feats.join(tp, on=["s1_eid", "cand_eid"], how="left").with_columns(pl.col("y").fill_null(0))
        lg(f"  positives {feats['y'].sum():,} of {feats.height:,}")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.tag}.parquet")
    feats.write_parquet(path)
    lg(f"wrote {path}: {feats.height:,} rows x {feats.width} cols")


if __name__ == "__main__":
    main()
