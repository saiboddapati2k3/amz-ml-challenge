"""Average the per-pair scores of several models, part by part (an ensemble without re-scoring).

    python scripts/avg_scores.py artifacts/score/test_n2_v3h1__ens \
        artifacts/score/test_n2_v3h1__m10 artifacts/score/test_n2_v3h1__m10b

Every input dir must hold the same part files (same candidates); rows are matched on
(s1_eid, cand_eid), never on row order. Then write the submission from the cached average:
    python -m src.predict --split test --cand artifacts/cand/test_n2_v3h1__ens ... (all parts cached)
"""
import glob
import os
import sys

import polars as pl

out, ins = sys.argv[1], sys.argv[2:]
assert len(ins) >= 2, "need at least two score dirs"
os.makedirs(out, exist_ok=True)
parts = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ins[0], "part-*.parquet")))
for d in ins[1:]:
    other = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "part-*.parquet")))
    assert other == parts, f"{d} has different parts than {ins[0]}"
for name in parts:
    base = pl.read_parquet(os.path.join(ins[0], name))
    ps = [base["p"]]
    for d in ins[1:]:
        o = base.select("s1_eid", "cand_eid").join(pl.read_parquet(os.path.join(d, name), columns=["s1_eid", "cand_eid", "p"]),
                                                   on=["s1_eid", "cand_eid"], how="left", maintain_order="left")
        assert o["p"].null_count() == 0, f"{d}/{name}: pairs missing"
        ps.append(o["p"])
    avg = base.with_columns(p=(sum(ps) / len(ps)).cast(pl.Float32))
    avg.write_parquet(os.path.join(out, name + ".tmp"))
    os.replace(os.path.join(out, name + ".tmp"), os.path.join(out, name))
print(f"wrote {len(parts)} averaged parts to {out} ({len(ins)} models)")
