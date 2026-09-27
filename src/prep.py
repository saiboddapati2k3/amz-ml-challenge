"""Phase 1: convert raw source TSVs into a normalized, columnar Parquet store.

    python -m src.prep --data dataset --split train      # or --split test

Every later phase reads artifacts/norm/<split>_s<n>/*.parquet instead of the 500 MB TSVs.
Normalization is vectorized in Polars (Rust, multi-threaded); only rows that contain
non-ASCII characters go through Python (anyascii transliteration: Devanagari, Kannada,
Bengali, accented Latin, ...). The rules are deliberately generic -- nothing below is
specific to US or India -- because the test set contains an unseen country.

Output columns per record
  eid        int64   encoded entity id (io_utils.encode_id)
  src        int8    1 / 2 / 3
  country    str     raw country label (open set; never one-hot encoded)
  name_raw, addr_raw   original strings (kept for error analysis)
  script     int8    0 ascii, 1 latin w/ accents, 2 other script (name)
  name_n     str     folded name incl. legal forms (canonical tokens, space separated)
  name_core  str     name_n minus legal forms / honorifics / generator noise words
  name_alias str     second name segment after an alias marker (a/k/a, dba, ...), else ''
  name_web   int8    1 if the name was a web domain (root kept as one compact token)
  addr_n     str     folded address tokens (canonical, original order)
  addr_nums  str     sorted unique numbers in the address, leading zeros stripped
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import polars as pl
from anyascii import anyascii

from src.io_utils import ID_BASE

# ----------------------------------------------------------------------------- #
# Token tables (canonical <- variant). Generic abbreviations only.
# ----------------------------------------------------------------------------- #
LEGAL_CANON = {
    "pvt": "private", "prv": "private", "pvtltd": "private limited", "ltd": "limited",
    "ltda": "limited", "inc": "incorporated", "corp": "corporation", "co": "company",
    "cos": "company", "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "opc": "opc",
    "pllc": "pllc", "pc": "pc", "pa": "pa", "ltee": "limited",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "spa": "spa", "srl": "srl",
    "cie": "company", "compagnie": "company",
}
# other name abbreviations (canonical <- variant); kept out of LEGAL_CANON so LEGAL_WORDS is unchanged
NAME_CANON = {"ets": "etablissements", "st": "saint", "ste": "sainte"}
LEGAL_WORDS = frozenset(list(LEGAL_CANON.values()) + [
    "private", "limited", "incorporated", "corporation", "company"])
# words that carry no identity: honorifics, articles, generator noise ('(India)')
NOISE_WORDS = frozenset([
    "the", "and", "of", "a", "an", "ms", "m", "s", "mr", "mrs", "smt", "shri", "sri", "shree",
    "dr", "india", "le", "la", "les", "de", "du", "des", "et", "l", "d",
])
# '(France)' is generator noise like '(India)'. Kept out of NOISE_WORDS: the blocker also uses those
# words as phonetic stop skeletons, and 'france' shares its skeleton with 'frank'/'franco'.
COUNTRY_NOISE = frozenset(["france"])
ADDR_CANON = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "ln": "lane", "dr": "drive",
    "hwy": "highway", "pkwy": "parkway", "ct": "court", "pl": "place", "cir": "circle",
    "sq": "square", "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor",
    "bldg": "building", "blk": "block", "sec": "sector", "ph": "phase", "nr": "near",
    "opp": "opposite", "mkt": "market", "ngr": "nagar", "clny": "colony", "col": "colony",
    "no": "number", "num": "number", "unit": "unit", "pmb": "pmb",
    "ft": "fort", "mt": "mount", "trl": "trail", "ter": "terrace", "pt": "point",
    # street-type abbreviations whose long form is common in S1 (found label-free: frequent in
    # S2/S3, near absent in S1). Pure renames for US/India, where the long forms don't occur.
    "r": "rue", "all": "allee", "imp": "impasse", "rte": "route", "ch": "chemin", "che": "chemin",
    "chem": "chemin", "crs": "cours", "q": "quai", "qu": "quai", "pas": "passage", "fbg": "faubourg",
    "res": "residence", "ndeg": "number",  # anyascii('N°') = 'Ndeg'
    "saint": "street", "sainte": "suite",  # 'St'/'Ste' already map to street/suite: keep one token
}
ADDR_NULL_RE = r"<null>|\b(?:null|none|nan|n/a)\b"

ALIAS_RE = r"\b(?:a/k/a|f/k/a|d/b/a|t/a|aka|fka|dba|formerly|trading as|also known as)\b:?"
WEB_RE = r"^(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|net|org|in|co|biz|info|fr|us|io|co\.in)$"


def _translit(s: pl.Series) -> pl.Series:
    """anyascii only for strings that need it (the vast majority are pure ASCII)."""
    return s.map_elements(lambda x: anyascii(x) if x and not x.isascii() else x,
                          return_dtype=pl.String, skip_nulls=True)


def _canon_tokens(col: pl.Expr, table: dict) -> pl.Expr:
    return (col.str.split(" ")
               .list.eval(pl.element().replace(table))
               .list.join(" ").str.replace_all(r"\s+", " ").str.strip_chars())


def _drop_tokens(col: pl.Expr, words) -> pl.Expr:
    words = list(words)
    return (col.str.split(" ")
               .list.eval(pl.element().filter(~pl.element().is_in(words) & (pl.element() != "")))
               .list.unique(maintain_order=True)
               .list.join(" "))


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    name = pl.col("business_name").fill_null("")
    addr = pl.col("business_address").fill_null("")
    df = df.with_columns(
        name_raw=name, addr_raw=addr,
        script=pl.when(name.str.contains(r"^[\x00-\x7F]*$")).then(0)
                 .when(name.str.contains(r"^[\x00-\x7FÀ-ɏ]*$")).then(1)
                 .otherwise(2).cast(pl.Int8),
    )
    df = df.with_columns(
        _n=_translit(pl.col("name_raw")).str.to_lowercase(),
        _a=_translit(pl.col("addr_raw")).str.to_lowercase(),
    )
    # ---------------- name ----------------
    n = (pl.col("_n")
         .str.replace_all(r"\(id:[^)]*\)", " ")
         .str.replace_all(r"\bm/s\b", " ")
         .str.replace_all(ALIAS_RE, " | "))
    df = df.with_columns(_n=n)
    df = df.with_columns(
        _web=pl.col("_n").str.strip_chars().str.extract(WEB_RE, 1),
    )
    n = (pl.when(pl.col("_web").is_not_null()).then(pl.col("_web")).otherwise(pl.col("_n"))
         .str.replace_all(r"[&+]", " and ")
         .str.replace_all(r"'", "")
         .str.replace_all(r"\.", "")                      # l.l.p -> llp, pvt. -> pvt
         .str.replace_all(r"[^a-z0-9|]+", " ")
         .str.replace_all(r"\s+", " ").str.strip_chars(" |"))
    df = df.with_columns(_n=n)
    df = df.with_columns(
        _seg0=pl.col("_n").str.split("|").list.get(0, null_on_oob=True).fill_null("").str.strip_chars(),
        _seg1=pl.col("_n").str.split("|").list.get(1, null_on_oob=True).fill_null("").str.strip_chars(),
    )
    df = df.with_columns(
        name_n=_canon_tokens(pl.col("_n").str.replace_all(r"\|", " "), LEGAL_CANON | NAME_CANON),
        _alias=_canon_tokens(pl.col("_seg1"), LEGAL_CANON | NAME_CANON),
        name_web=pl.col("_web").is_not_null().cast(pl.Int8),
    )
    df = df.with_columns(
        name_core=_drop_tokens(pl.col("name_n"), LEGAL_WORDS | NOISE_WORDS | COUNTRY_NOISE),
        name_alias=_drop_tokens(pl.col("_alias"), LEGAL_WORDS | NOISE_WORDS | COUNTRY_NOISE),
    )
    df = df.with_columns(  # never leave the core empty
        name_core=pl.when(pl.col("name_core") == "").then(pl.col("name_n")).otherwise(pl.col("name_core")))
    # ---------------- address ----------------
    a = (pl.col("_a").str.replace_all(ADDR_NULL_RE, " ")
          .str.replace_all(r"(\d+)(?:st|nd|rd|th)\b", "$1")   # ordinals
          .str.replace_all(r"(\d)([a-z])", "$1 $2")            # 16west -> 16 west
          .str.replace_all(r"([a-z])(\d)", "$1 $2")
          .str.replace_all(r"&", " and ")
          .str.replace_all(r"[^a-z0-9]+", " ")
          .str.replace_all(r"\s+", " ").str.strip_chars())
    df = df.with_columns(_a=a)
    df = df.with_columns(
        addr_n=_canon_tokens(pl.col("_a"), ADDR_CANON),
        addr_nums=(pl.col("_a").str.extract_all(r"\d+")
                   .list.eval(pl.element().str.strip_chars_start("0").replace("", "0"))
                   .list.unique().list.sort().list.join(" ")),
    )
    return df.select(
        "eid", "src", "country", "name_raw", "addr_raw", "script",
        "name_n", "name_core", "name_alias", "name_web", "addr_n", "addr_nums",
    )


def encode_id_expr(col: str = "entity_id") -> pl.Expr:
    c = pl.col(col)
    return (c.str.slice(1, 1).cast(pl.Int64) * ID_BASE + c.str.slice(3).cast(pl.Int64)).alias("eid")


READ_KW = dict(separator="\t", quote_char=None, infer_schema=False, encoding="utf8")


def prep_file(path: str, out_dir: str, batch_rows: int = 500_000, resume: bool = False) -> int:
    """Resume keeps finished parts; batch_rows must match the run that wrote them."""
    os.makedirs(out_dir, exist_ok=True)
    done = set()
    for old in sorted(glob.glob(os.path.join(out_dir, "*.parquet"))):
        try:
            ok = resume and pl.read_parquet_schema(old) is not None
        except Exception:
            ok = False
        if ok:
            done.add(os.path.basename(old))
        else:
            os.remove(old)
    lf = pl.scan_csv(path, **READ_KW)
    total = lf.select(pl.len()).collect().item()
    for i, off in enumerate(range(0, total, batch_rows)):
        if f"part-{i:03d}.parquet" in done:
            continue
        chunk = lf.slice(off, batch_rows).collect()
        chunk = chunk.with_columns(
            encode_id_expr(), src=pl.col("entity_id").str.slice(1, 1).cast(pl.Int8),
            country=pl.col("country").fill_null("").str.strip_chars())
        normalize(chunk).write_parquet(os.path.join(out_dir, f"part-{i:03d}.parquet"),
                                       compression="zstd")
    return total


def load(norm_dir: str, split: str, src: int, columns=None) -> pl.DataFrame:
    return pl.read_parquet(os.path.join(norm_dir, f"{split}_s{src}", "*.parquet"), columns=columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--out", default="artifacts/norm")
    ap.add_argument("--sources", default="1,2,3", help="comma list, e.g. 3 to redo one source")
    ap.add_argument("--batch-rows", type=int, default=500_000)
    ap.add_argument("--resume", action="store_true", help="keep valid parts from a crashed run")
    args = ap.parse_args()
    for n in (int(x) for x in args.sources.split(",")):
        t = time.time()
        path = os.path.join(args.data, args.split, f"{args.split}_source{n}.tsv")
        rows = prep_file(path, os.path.join(args.out, f"{args.split}_s{n}"), args.batch_rows, args.resume)
        print(f"{args.split} s{n}: {rows:,} rows in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
