"""Decision rule: a candidate whose name contains the record's own country word (e.g. 'france',
'india', 'us') while the S1 name does not is a different entity, so its score is set to 0.

Evidence (train, m10 out-of-fold pairs): the generator drops the country word from copies but never
adds it. Candidate-only 'india': 0 positives of 158,579 pairs; US candidate-only 'us'/'usa'/'america'/
'american': 0 of ~18,800. On test, France predictions with a candidate-only 'france' grew from 8,545
(0.9456 upload) to 27,213 (0.942 upload).

    python scripts/country_rule.py artifacts/score/<in> artifacts/score/<out> --norm artifacts/norm_n2 --split test
"""
import argparse
import glob
import os

import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--norm", default="artifacts/norm")
ap.add_argument("--split", default="test")
args = ap.parse_args()


def with_word(srcs):
    """eids whose normalized name contains their own country label as a token."""
    lf = pl.concat([pl.scan_parquet(os.path.join(args.norm, f"{args.split}_s{s}", "*.parquet"))
                      .select("eid", "country", "name_n") for s in srcs])
    return (lf.filter(pl.col("name_n").fill_null("").str.split(" ").list.contains(pl.col("country").str.to_lowercase()))
              .select("eid").collect())


s1_word = with_word((1,)).rename({"eid": "s1_eid"})
c_word = with_word((2, 3)).rename({"eid": "cand_eid"})
os.makedirs(args.out, exist_ok=True)
n_all = n_zero = 0
for path in sorted(glob.glob(os.path.join(args.src, "part-*.parquet"))):
    d = pl.read_parquet(path)
    hit = (d.select("s1_eid", "cand_eid").with_row_index("i")
             .join(c_word, on="cand_eid", how="semi").join(s1_word, on="s1_eid", how="anti")["i"])
    mask = pl.Series("m", [False] * d.height)
    if hit.len():
        mask = mask.scatter(hit, True)
    d = d.with_columns(p=pl.when(mask).then(0.0).otherwise(pl.col("p")).cast(pl.Float32))
    n_all += d.height
    n_zero += int(mask.sum())
    dst = os.path.join(args.out, os.path.basename(path))
    d.write_parquet(dst + ".tmp")
    os.replace(dst + ".tmp", dst)
print(f"{args.out}: {n_all:,} pairs, {n_zero:,} zeroed by the country-word rule "
      f"(S1 with the word {s1_word.height:,}, candidates with it {c_word.height:,})")
