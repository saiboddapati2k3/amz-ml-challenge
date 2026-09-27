"""Split cross-encoder test scores by country: countries present in the reranker's training data
(the train split's country labels) get the blend; unseen countries (France) get veto-only.

    python scripts/ce_split.py ce_data/ce_scores_test.parquet ce_data/ce_scores_test_seen.parquet ce_data/ce_scores_test_unseen.parquet
"""
import sys

import polars as pl

from src.prep import READ_KW, encode_id_expr

src, seen_out, unseen_out = sys.argv[1:4]
seen = pl.read_csv("dataset/train/train_source1.tsv", **READ_KW, columns=["country"])["country"].str.strip_chars().unique()
s1 = (pl.read_csv("dataset/test/test_source1.tsv", **READ_KW, columns=["entity_id", "country"])
        .select(encode_id_expr().alias("s1_eid"), pl.col("country").str.strip_chars()))
ce = pl.read_parquet(src).join(s1, on="s1_eid")
a, b = ce.filter(pl.col("country").is_in(seen.implode())), ce.filter(~pl.col("country").is_in(seen.implode()))
a.drop("country").write_parquet(seen_out)
b.drop("country").write_parquet(unseen_out)
print(f"training countries {sorted(seen.to_list())}: {a.height:,} rows -> {seen_out}; unseen: {b.height:,} rows -> {unseen_out}")
