"""Phase 2: candidate generation (blocking) with a multi-key weighted inverted index.

For every S1 record we retrieve S2/S3 records that share *rare* keys with it. Each key
family targets one noise pattern observed in the training data:

  N1  name token / compact name        reordering, junk prefixes, web-domain names
  N2  pair of name tokens              common words that are only rare together
  N3  4-char prefix of a name token    typos late in a token, truncation
  A1  house number x rare addr token   DBA / random trade names, name typos
  A2  pair of rare address tokens      addresses without numbers
  M1  name token x rare addr token     common business names at one place
  P1  phonetic skeleton of name token  lossy transliteration (dainamik ~ dynamic)
  P2  pair of phonetic skeletons       same, for common words
  A3  pair of address numbers          short addresses: '205 206, Mumbai'
  A4  house number x any addr token    'C-74, Khordha, Odisha' (city is not rare)
  PN  phonetic name token x house number  native-script names ('best phuds', '806 mumbai')

Blocking runs separately per country label (works for any label, incl. unseen ones).
A key whose S2+S3 posting list is longer than its cap identifies nothing and is
dropped; every other shared key adds idf = ln(N / df) to the family score.

Memory design (8 GB laptop, ~6M S2+S3 rows per country): tokens are hashed to u64
once; key families are processed ONE AT A TIME; per family and S1 chunk we keep only
the top-M candidates per S1; the families are then merged into name / addr / mix
scores and the final candidate list is the union of the top-K by combined score and
the top-k by name-only and addr-only score (one noisy field cannot hide a match).

Corpus statistics (df, idf) are computed on the corpus being resolved: label-free and
identical for train and test.

    python -m src.block --split train --s1-ids artifacts/splits/dev_s1.parquet --tag train_dev
    python -m src.block --split train --tag train_full    # -> artifacts/cand/train_full/part-*.parquet
    python -m src.block --split test
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import shutil
import time
from typing import Dict, Optional

import numpy as np

# Every polars thread builds its own hash tables in group_by/join, so peak memory grows with the
# thread count; 4 threads keep a country's phase A inside an 8 GB laptop. Override via env.
os.environ.setdefault("POLARS_MAX_THREADS", "4")
import polars as pl  # noqa: E402

from src.prep import LEGAL_WORDS, NOISE_WORDS, load

# family -> (score group, df cap)
FAMILIES = {
    "N1": ("name", 300), "N2": ("name", 300), "N3": ("name", 300),
    "A1": ("addr", 200), "A2": ("addr", 200), "M1": ("mix", 200),
    "P1": ("name", 300), "P2": ("name", 300), "A3": ("addr", 200), "A4": ("addr", 200),
    "PN": ("mix", 200),
}
MAX_NAME_TOK = 4   # rarest name tokens per record used for keys
RARE_ADDR = 4      # rarest address tokens per record
MAX_NUMS = 3       # address numbers per record (A1, A3), longest first
A4_NUMS = 2        # house numbers per record crossed with every address token (A4)
A4_TOKS = 16       # address tokens per record for A4 (city sits near the end in S1)
TOP_M = 25         # per family, candidates kept per S1 before merging
SLICE = 1_000_000  # S2/S3 rows tokenised / keyed at a time (phase A memory ~ slice, not country)
BUCKETS = 8        # key counting is split into hash buckets of this many parts


def _hash(e: pl.Expr) -> pl.Expr:
    return e.hash(seed=17)


# Consonant skeleton: collapses transliteration variants of the same word.
_DIGRAPHS = [("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
             ("sh", "s"), ("ch", "c"), ("jh", "j"), ("ck", "k")]
_CLASSES = [("[ckq]", "k"), ("[vwb]", "b"), ("[jzg]", "j"), ("x", "ks"), ("h", "")]


def phonetic(e: pl.Expr) -> pl.Expr:
    """Lowercase ascii word -> skeleton: first letter + consonant classes, no repeats."""
    for a, b in _DIGRAPHS:
        e = e.str.replace_all(a, b, literal=True)
    first, rest = e.str.slice(0, 1), e.str.slice(1)
    for a, b in _CLASSES:
        first, rest = first.str.replace_all(a, b), rest.str.replace_all(a, b)
    e = pl.concat_str([first, rest.str.replace_all(r"[aeiouy]", "")])
    e = e.str.replace_all(r"m([^aeiouym])", "n$1")  # nasal before a consonant: emtr ~ entr
    for c in "bdfjklmnprst":  # regex has no backrefs: collapse runs letter by letter
        e = e.str.replace_all(c + "+", c)
    return e


PHON_STOP = (pl.DataFrame({"w": sorted(LEGAL_WORDS | NOISE_WORDS | {"pvt", "ltd", "pte"})})
             .select(pl.col("w").str.split(" ")).explode("w").select(phonetic(pl.col("w")))["w"]
             .unique().sort().to_list())


def token_tables(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """(row u32, t u64[, p u64]) tables. `df` has row, name_core, name_alias, addr_n, addr_nums."""
    name = (df.select("row", t=pl.concat_str([pl.col("name_core"), pl.col("name_alias")], separator=" ")
                      .str.split(" "))
              .explode("t").filter(pl.col("t").str.len_chars() > 0).unique(["row", "t"])
              .with_columns(p=pl.when(pl.col("t").str.len_chars() >= 5)
                              .then(_hash(pl.col("t").str.slice(0, 4))).otherwise(None),
                            t=_hash(pl.col("t"))))
    comp = df.select("row", t=_hash(pl.col("name_core").str.replace_all(" ", "")))
    addr = (df.select("row", t=pl.col("addr_n").str.split(" "))
              .explode("t").filter(pl.col("t").str.contains(r"^[a-z][a-z0-9]{2,}$"))
              .unique(["row", "t"], maintain_order=True).with_columns(t=_hash(pl.col("t"))))
    num = (df.select("row", t=pl.col("addr_nums").str.split(" "))
             .explode("t").filter(pl.col("t").str.len_chars() > 0)
             .sort(["row", pl.col("t").str.len_chars(), "t"], descending=[False, True, False])
             .group_by("row", maintain_order=True).head(MAX_NUMS)
             .with_columns(t=_hash(pl.col("t"))))
    phon = (df.select("row", t=pl.concat_str([pl.col("name_core"), pl.col("name_alias")], separator=" ")
                      .str.split(" "))
              .explode("t").filter(pl.col("t").str.contains(r"^[a-z]{3,}$"))
              .with_columns(t=phonetic(pl.col("t")))
              .filter((pl.col("t").str.len_chars() >= 3) & ~pl.col("t").is_in(PHON_STOP))
              .unique(["row", "t"]).with_columns(t=_hash(pl.col("t"))))
    cast = lambda d: d.with_columns(pl.col("row").cast(pl.UInt32))
    return {"name": cast(name), "comp": cast(comp), "addr": cast(addr), "num": cast(num),
            "phon": cast(phon)}


def keep_rarest(tt: Dict[str, pl.DataFrame], freq: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
    out = dict(tt)
    # A4 crosses house numbers with address tokens regardless of rarity (city, locality)
    out["addr_all"] = tt["addr"].group_by("row", maintain_order=True).head(A4_TOKS)
    for kind, k in (("name", MAX_NAME_TOK), ("addr", RARE_ADDR), ("phon", MAX_NAME_TOK)):
        out[kind] = (tt[kind].join(freq[kind], on="t", how="left").with_columns(pl.col("f").fill_null(0))
                     .sort(["row", "f", "t"]).group_by("row", maintain_order=True).head(k).drop("f"))
    return out


def family_keys(fam: str, tt: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """(row, key) for one family, built from rarest-token tables."""
    name, addr, num = tt["name"], tt["addr"], tt["num"]
    pair = lambda a, b, salt: _hash(pl.struct(pl.lit(salt), a, b))
    if fam == "N1":
        k = pl.concat([name.select("row", "t"), tt["comp"]]).select("row", key=_hash(pl.struct(pl.lit(1), "t")))
    elif fam == "N2":
        k = (name.join(name, on="row", suffix="_b").filter(pl.col("t") < pl.col("t_b"))
                 .select("row", key=pair(pl.col("t"), pl.col("t_b"), 2)))
    elif fam == "N3":
        k = name.filter(pl.col("p").is_not_null()).select("row", key=_hash(pl.struct(pl.lit(3), "p")))
    elif fam == "A1":
        k = num.join(addr, on="row", suffix="_a").select("row", key=pair(pl.col("t"), pl.col("t_a"), 4))
    elif fam == "A2":
        k = (addr.join(addr, on="row", suffix="_b").filter(pl.col("t") < pl.col("t_b"))
                 .select("row", key=pair(pl.col("t"), pl.col("t_b"), 5)))
    elif fam == "M1":
        a3 = addr.group_by("row", maintain_order=True).head(2)
        n3 = name.group_by("row", maintain_order=True).head(3)
        k = n3.join(a3, on="row", suffix="_a").select("row", key=pair(pl.col("t"), pl.col("t_a"), 6))
    elif fam == "P1":
        k = tt["phon"].select("row", key=_hash(pl.struct(pl.lit(7), "t")))
    elif fam == "P2":
        ph = tt["phon"]
        k = (ph.join(ph, on="row", suffix="_b").filter(pl.col("t") < pl.col("t_b"))
               .select("row", key=pair(pl.col("t"), pl.col("t_b"), 8)))
    elif fam == "A3":
        k = (num.join(num, on="row", suffix="_b").filter(pl.col("t") < pl.col("t_b"))
                .select("row", key=pair(pl.col("t"), pl.col("t_b"), 9)))
    elif fam == "A4":
        n2 = num.group_by("row", maintain_order=True).head(A4_NUMS)
        k = n2.join(tt["addr_all"], on="row", suffix="_a").select("row", key=pair(pl.col("t"), pl.col("t_a"), 10))
    elif fam == "PN":
        n2 = num.group_by("row", maintain_order=True).head(A4_NUMS)
        k = tt["phon"].join(n2, on="row", suffix="_n").select("row", key=pair(pl.col("t"), pl.col("t_n"), 11))
    else:
        raise ValueError(fam)
    return k.unique()


def _det_rank(score: str, salt: int = 0) -> pl.Expr:
    """1-based rank of `score` (descending) within each S1 row i; ties broken by a seeded hash
    of (i, j, salt).

    rank("ordinal") breaks ties by frame order, which multi-threaded group_by does not fix, so
    equal-score hits at a top-M cut made blocking non-reproducible. Breaking ties by j alone is
    reproducible but lets every family keep the SAME tied candidates (recall -0.24 pts on dev);
    a per-family salt keeps ties diverse across families and the result deterministic.
    """
    tie = pl.struct("i", "j").hash(seed=salt)
    key = pl.struct(-pl.col(score).cast(pl.Float64), tie, pl.col("j"))
    return key.rank("ordinal").over("i").cast(pl.UInt16)


def _merge_chunk(hits: list, k_total: int, k_side: int) -> pl.DataFrame:
    """Family hits of one S1 chunk -> per-group scores, ranks and the kept candidates."""
    agg = (pl.concat(hits).group_by("i", "j").agg(
               s_name=pl.col("w").filter(pl.col("g") == "name").sum(),
               s_addr=pl.col("w").filter(pl.col("g") == "addr").sum(),
               s_mix=pl.col("w").filter(pl.col("g") == "mix").sum(),
               n_fam=pl.len().cast(pl.UInt8))
           .with_columns(s_all=pl.col("s_name") + pl.col("s_addr") + pl.col("s_mix")))
    agg = agg.with_columns(r_all=_det_rank("s_all"), r_name=_det_rank("s_name"), r_addr=_det_rank("s_addr"))
    return agg.filter((pl.col("r_all") <= k_total)
                      | ((pl.col("r_name") <= k_side) & (pl.col("s_name") > 0))
                      | ((pl.col("r_addr") <= k_side) & (pl.col("s_addr") > 0)))


def _spill_postings(fam: str, cap: int, tt23: Dict[str, pl.DataFrame], n23: int, spill: str) -> None:
    """Capped, idf-weighted S2/S3 postings (key, j, w) of one family -> <spill>/<fam>_post/.

    Keys are built per row slice (every key is row-local) and counted per hash bucket (a key
    lives in one bucket), so neither the full key table nor its count table is ever in memory.
    """
    tmp = os.path.join(spill, f"{fam}_tmp")
    out = os.path.join(spill, f"{fam}_post")
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    for n, lo in enumerate(range(0, n23, SLICE)):
        part = {k: v.filter((pl.col("row") >= lo) & (pl.col("row") < lo + SLICE)) for k, v in tt23.items()}
        (family_keys(fam, part).with_columns(b=(pl.col("key") % BUCKETS).cast(pl.UInt8))
            .write_parquet(os.path.join(tmp, f"s{n:03d}.parquet")))
        del part
        gc.collect()
    keys = pl.scan_parquet(os.path.join(tmp, "*.parquet"))
    for b in range(BUCKETS):
        kb = keys.filter(pl.col("b") == b).select("row", "key").collect()
        cnt = kb.group_by("key").agg(n=pl.len()).filter(pl.col("n") <= cap)
        (kb.join(cnt, on="key")
           .with_columns(w=(np.log(n23) - pl.col("n").cast(pl.Float32).log()).cast(pl.Float32))
           .select("key", pl.col("row").alias("j"), "w")
           .write_parquet(os.path.join(out, f"b{b}.parquet")))
        del kb, cnt
        gc.collect()
    shutil.rmtree(tmp, ignore_errors=True)


def candidate_pairs(s1: pl.DataFrame, s23: pl.DataFrame, spill: str, k_total: int = 40,
                    k_side: int = 10, chunk: int = 100_000, log=print):
    """Rows of s1/s23 must have a 0-based `row`. Yields one kept-candidate frame per S1 chunk
    with columns (i, j, s_name, s_addr, s_mix, n_fam, s_all, r_all, r_name, r_addr).

    Phase A builds every family's capped S2/S3 postings (in row slices and key buckets) and S1
    keys once and spills them to `spill`; phase B walks S1 in chunks, so peak memory depends on
    SLICE and `chunk`, not on the size of the country.
    """
    t0 = time.time()
    n23 = s23.height
    tt1 = token_tables(s1)
    # token tables are row-local, so slicing S2/S3 by rows gives the same tables in less memory
    tt23 = {}
    for lo in range(0, n23, SLICE):
        for k, v in token_tables(s23.slice(lo, SLICE)).items():
            tt23.setdefault(k, []).append(v)
        gc.collect()
    del s23  # the caller drops its reference too: raw strings are not needed after tokenising
    tt23 = {k: pl.concat(v) for k, v in tt23.items()}
    freq = {k: pl.concat([tt1[k].select("t"), tt23[k].select("t")]).group_by("t").agg(f=pl.len())
            for k in ("name", "addr", "phon")}
    tt1, tt23 = keep_rarest(tt1, freq), keep_rarest(tt23, freq)
    del freq
    gc.collect()
    log(f"  tokens ready {time.time() - t0:.0f}s")

    os.makedirs(spill, exist_ok=True)
    for fam, (grp, cap) in FAMILIES.items():
        _spill_postings(fam, cap, tt23, n23, spill)
        family_keys(fam, tt1).sort("row").write_parquet(os.path.join(spill, f"{fam}_k1.parquet"))
        gc.collect()
    del tt1, tt23
    gc.collect()
    log(f"  keys spilled ({time.time() - t0:.0f}s)")

    for lo in range(0, s1.height, chunk):
        hits = []
        for fam, (grp, cap) in FAMILIES.items():
            part = (pl.scan_parquet(os.path.join(spill, f"{fam}_k1.parquet"))
                      .filter((pl.col("row") >= lo) & (pl.col("row") < lo + chunk)).collect())
            h = (part.join(pl.read_parquet(os.path.join(spill, f"{fam}_post", "*.parquet")), on="key")
                     .group_by(pl.col("row").alias("i"), "j").agg(pl.col("w").sum()))
            hits.append(h.filter(_det_rank("w", salt=list(FAMILIES).index(fam) + 1) <= TOP_M)
                         .with_columns(g=pl.lit(grp)))
            del part, h
        keep = _merge_chunk(hits, k_total, k_side)
        del hits
        gc.collect()
        log(f"  S1 {min(lo + chunk, s1.height):,}/{s1.height:,}: kept {keep.height/1e6:.2f}M ({time.time() - t0:.0f}s)")
        yield keep


COLS = ["eid", "country", "name_core", "name_alias", "addr_n", "addr_nums"]


def read_cand(path: str, columns=None) -> pl.DataFrame:
    """Candidates from a single parquet file or a directory of part files."""
    if os.path.isdir(path):
        path = os.path.join(path, "*.parquet")
    return pl.read_parquet(path, columns=columns)


def run_split(norm: str, split: str, s1_ids: Optional[np.ndarray], out_dir: str, k_total: int,
              k_side: int, chunk: int, log=print) -> int:
    """Write <out_dir>/part-<country>-<n>.parquet; returns the number of pairs."""
    s1 = load(norm, split, 1, COLS)
    if s1_ids is not None:
        s1 = s1.filter(pl.col("eid").is_in(pl.Series(s1_ids).implode()))
    s23_paths = [os.path.join(norm, f"{split}_s{n}", "*.parquet") for n in (2, 3)]
    n23_all = sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in s23_paths)
    log(f"{split}: s1 {s1.height:,}  s23 {n23_all:,}")
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "*.parquet")):
        os.remove(old)
    total = 0
    for country in s1["country"].unique().sort().to_list():
        a = s1.filter(pl.col("country") == country).with_row_index("row")
        # one country's S2/S3 at a time (filter pushed into the scan): peak memory is per country
        b = (pl.concat([pl.scan_parquet(p).select(COLS) for p in s23_paths])
               .filter(pl.col("country") == country).collect().with_row_index("row"))
        log(f"[{country}] s1 {a.height:,}  s23 {b.height:,}")
        ai = a.select(pl.col("row").alias("i"), pl.col("eid").alias("s1_eid"))
        bj = b.select(pl.col("row").alias("j"), pl.col("eid").alias("cand_eid"))
        spill = os.path.join(out_dir, "_spill")
        safe = "".join(ch if ch.isalnum() else "_" for ch in country) or "none"
        pairs = candidate_pairs(a, b, spill, k_total, k_side, chunk, log)
        del b
        for n, c in enumerate(pairs):
            c = c.join(ai, on="i").join(bj, on="j").drop("i", "j")
            c.write_parquet(os.path.join(out_dir, f"part-{safe}-{n:03d}.parquet"))
            total += c.height
        shutil.rmtree(spill, ignore_errors=True)
        del a, ai, bj, pairs
        gc.collect()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--norm", default="artifacts/norm")
    ap.add_argument("--out", default="artifacts/cand")
    ap.add_argument("--s1-ids", default=None, help="parquet with column eid: restrict S1 (train dev)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--k-total", type=int, default=40)
    ap.add_argument("--k-side", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=100_000)
    args = ap.parse_args()
    ids = pl.read_parquet(args.s1_ids)["eid"].to_numpy() if args.s1_ids else None
    lg = lambda m: print(m, flush=True)
    out_dir = os.path.join(args.out, args.tag or args.split)
    t0 = time.time()
    n = run_split(args.norm, args.split, ids, out_dir, args.k_total, args.k_side, args.chunk, lg)
    n1 = read_cand(out_dir, ["s1_eid"])["s1_eid"].n_unique()
    lg(f"wrote {out_dir}: {n:,} pairs over {n1:,} S1 with candidates ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
