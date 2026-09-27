"""Fast structural check of a submission (the official validator is pure Python and slow on 8 GB):
one row per test S1 in file order, header names, no duplicate ids in a list, S2/S3 ids only, and
every matched id among that S1's candidates.

    python scripts/quick_check.py output/<run>
"""
import sys

import polars as pl

out = sys.argv[1]
kw = dict(separator="\t", quote_char=None, infer_schema=False)
s1 = pl.read_csv("dataset/test/test_source1.tsv", **kw, columns=["entity_id"])
m = pl.read_csv(f"{out}/matching_results.tsv", **kw)
c = pl.read_csv(f"{out}/candidate_pairs.tsv", **kw)
assert m.columns == ["source1_entity_id", "matched_entity_ids"], m.columns
assert c.columns == ["source1_entity_id", "candidate_entity_ids"], c.columns
assert m.height == c.height == s1.height, (m.height, c.height, s1.height)
assert m["source1_entity_id"].equals(s1["entity_id"]) and c["source1_entity_id"].equals(s1["entity_id"])
explode = lambda d, col: (d.with_columns(pl.col(col).fill_null("").str.split(",")).explode(col)
                           .filter(pl.col(col) != ""))
me, ce = explode(m, "matched_entity_ids"), explode(c, "candidate_entity_ids")
for d, col in ((me, "matched_entity_ids"), (ce, "candidate_entity_ids")):
    assert d[col].str.contains(r"^S[23]-\d+$").all(), f"bad id in {col}"
    assert d.select("source1_entity_id", col).is_unique().all(), f"duplicate id within a {col} list"
missing = me.join(ce, left_on=["source1_entity_id", "matched_entity_ids"],
                  right_on=["source1_entity_id", "candidate_entity_ids"], how="anti")
assert missing.height == 0, f"{missing.height} matches not among candidates"
print(f"OK {out}: {m.height:,} rows, {me.height:,} matches, {ce.height:,} candidates, "
      f"{(m['matched_entity_ids'].fill_null('') == '').sum():,} empty")
