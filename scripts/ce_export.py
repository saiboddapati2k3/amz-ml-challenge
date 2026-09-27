"""Export record-pair text for the cross-encoder reranker (Kaggle GPU).

    # training pairs: out-of-fold scores of the training sample (labels included)
    python scripts/ce_export.py train artifacts/model/oof_m10.parquet out/ce_train.parquet
    # pairs to rescore: every pair of a score dir whose model score is uncertain
    python scripts/ce_export.py pairs artifacts/score/train_check_n1_v3h1__m10 out/ce_check.parquet --split train
    python scripts/ce_export.py pairs artifacts/score/test_n2_v3h1__m10f3 out/ce_test.parquet --split test --norm artifacts/norm_n2

Text of a record: "<name> ; <address>", transliterated to Latin (anyascii), so native-script
records work with an English wordpiece model.
"""
import argparse
import os

import polars as pl
from anyascii import anyascii

LO, HI = 0.001, 0.999  # "uncertain" band of the LightGBM score


def text(norm: str, split: str, eids: pl.DataFrame, src: tuple) -> pl.DataFrame:
    parts = [pl.scan_parquet(os.path.join(norm, f"{split}_s{k}", "*.parquet")).select("eid", "name_raw", "addr_raw")
               .join(eids.lazy(), on="eid", how="semi").collect() for k in src]
    d = pl.concat(parts)
    ascii_ = lambda c: pl.col(c).fill_null("").map_elements(lambda x: anyascii(x) if not x.isascii() else x,
                                                             return_dtype=pl.String)
    return d.select("eid", t=pl.concat_str([ascii_("name_raw"), pl.lit(" ; "), ascii_("addr_raw")]).str.replace_all(r"\s+", " "))


def attach(pairs: pl.DataFrame, norm: str, split: str) -> pl.DataFrame:
    a = text(norm, split, pairs.select(pl.col("s1_eid").alias("eid")).unique(), (1,))
    b = text(norm, split, pairs.select(pl.col("cand_eid").alias("eid")).unique(), (2, 3))
    return (pairs.join(a.rename({"eid": "s1_eid", "t": "text_a"}), on="s1_eid", how="left")
                 .join(b.rename({"eid": "cand_eid", "t": "text_b"}), on="cand_eid", how="left"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "pairs"])
    ap.add_argument("src", help="train: OOF parquet; pairs: score dir")
    ap.add_argument("out")
    ap.add_argument("--split", default="train")
    ap.add_argument("--norm", default="artifacts/norm")
    ap.add_argument("--easy", type=int, default=150_000, help="train: confident positives and negatives each")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.mode == "train":
        oof = pl.read_parquet(args.src, columns=["s1_eid", "cand_eid", "y", "p"])
        hard = oof.filter(pl.col("p").is_between(LO, HI))
        pos = oof.filter((pl.col("p") > HI) & (pl.col("y") == 1))
        neg = oof.filter((pl.col("p") < LO) & (pl.col("y") == 0))
        pairs = pl.concat([hard, pos.sample(min(args.easy, pos.height), seed=1),
                           neg.sample(min(args.easy, neg.height), seed=2)]).sample(fraction=1.0, shuffle=True, seed=3)
    else:
        pairs = (pl.scan_parquet(os.path.join(args.src, "part-*.parquet")).select("s1_eid", "cand_eid", "p")
                   .filter(pl.col("p").is_between(LO, HI)).collect())
    out = attach(pairs, args.norm, args.split)
    assert out["text_a"].null_count() == 0 and out["text_b"].null_count() == 0
    out.write_parquet(args.out, compression="zstd")
    print(f"{args.out}: {out.height:,} pairs" + (f", positives {int(out['y'].sum()):,}" if "y" in out.columns else ""),
          f"({os.path.getsize(args.out) / 2**20:.0f} MB)")
    print(out.select("text_a", "text_b").head(3).to_dicts())


if __name__ == "__main__":
    main()
