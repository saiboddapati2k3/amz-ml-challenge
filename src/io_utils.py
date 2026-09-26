"""TSV reading/writing for the challenge.

All files are tab-separated. Addresses can contain commas and quote characters,
so we read with QUOTE_NONE and keep every field as a plain string (no NaN).
"""
from __future__ import annotations

import csv
import os
from array import array
from typing import Dict, Iterable, Iterator, List, Mapping, Tuple

import numpy as np
import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


_READ_KW = dict(sep="\t", dtype=str, keep_default_na=False, na_values=[],
                quoting=csv.QUOTE_NONE, encoding="utf-8")


def _clean_source(df: pd.DataFrame, path: str) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; got {list(df.columns)}")
    for c in SOURCE_COLS:
        df[c] = df[c].astype(str).str.strip()
    df["source"] = df["entity_id"].str.slice(0, 2)  # 'S1' / 'S2' / 'S3'
    return df


def read_source(path: str) -> pd.DataFrame:
    """Read one WHOLE source TSV. Every column is a string; missing values become ''.

    Only for small files / tests. Full-size sources (5M rows) go through iter_source.
    """
    return _clean_source(pd.read_csv(path, **_READ_KW), path)


def iter_source(path: str, chunksize: int = 50_000) -> Iterator[pd.DataFrame]:
    """Stream one source TSV as cleaned chunks of <= chunksize rows.

    Same result as read_source (QUOTE_NONE + all-str + no NA means a field is just
    the text between tabs; blank lines skipped), but parsed line by line so every
    allocation is small and bounded. pandas' C tokenizer grows its buffers
    geometrically and failed with 'out of memory' on a machine with little free
    commit even though the process itself used ~400 MB.
    """
    with open(path, encoding="utf-8", newline="") as f:
        header = [c.strip() for c in f.readline().rstrip("\r\n").split("\t")]
        ncol = len(header)
        cols: List[list] = [[] for _ in range(ncol)]
        lineno = 1
        for line in f:
            lineno += 1
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != ncol:
                raise ValueError(f"{path}:{lineno}: expected {ncol} fields, got {len(parts)}")
            for c, v in zip(cols, parts):
                c.append(v)
            if len(cols[0]) >= chunksize:
                yield _clean_source(pd.DataFrame(dict(zip(header, cols)), dtype=str), path)
                cols = [[] for _ in range(ncol)]
        if cols[0]:
            yield _clean_source(pd.DataFrame(dict(zip(header, cols)), dtype=str), path)


def source_path(data_dir: str, split: str, n: int) -> str:
    return os.path.join(data_dir, split, f"{split}_source{n}.tsv")


# --------------------------------------------------------------------------- #
# Compact entity ids: 'S2-681193310' -> 2 * 10**12 + 681193310 (int64).
# 8 bytes per id instead of ~60 for a Python str; sortable and searchable in numpy.
# --------------------------------------------------------------------------- #
ID_BASE = 10 ** 12


def encode_id(s: str) -> int:
    if len(s) < 4 or s[0] != "S" or s[2] != "-" or s[1] not in "123":
        raise ValueError(f"unexpected entity id {s!r}")
    n = int(s[3:])
    if not 0 <= n < ID_BASE:
        raise ValueError(f"entity id out of range {s!r}")
    return int(s[1]) * ID_BASE + n


def encode_ids(ids: pd.Series) -> np.ndarray:
    """Vectorised encode_id for one chunk of ids."""
    ids = ids.astype(str)
    ok = ids.str.fullmatch(r"S[123]-\d{1,12}")
    if not bool(ok.all()):
        raise ValueError(f"unexpected entity ids, e.g. {ids[~ok].head(3).tolist()}")
    return (ids.str.slice(1, 2).astype(np.int64) * ID_BASE
            + ids.str.slice(3).astype(np.int64)).to_numpy(np.int64)


def decode_id(x: int) -> str:
    return f"S{int(x) // ID_BASE}-{int(x) % ID_BASE}"


def read_id_list_pairs(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream an id-list file into compact arrays without building a dict of lists.

    Returns (row_s1, pair_s1, pair_cand): every S1 row in file order, and one
    (s1, cand) entry per listed id, all int64-encoded. ~16 bytes per pair.
    """
    row_s1, pair_s1, pair_cand = array("q"), array("q"), array("q")
    with open(path, encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            if not line.strip():
                continue
            s1, _, rest = line.rstrip("\n").partition("\t")
            e1 = encode_id(s1.strip())
            row_s1.append(e1)
            for m in parse_id_list(rest):
                pair_s1.append(e1)
                pair_cand.append(encode_id(m))
    return (np.frombuffer(row_s1, np.int64).copy(), np.frombuffer(pair_s1, np.int64).copy(),
            np.frombuffer(pair_cand, np.int64).copy())


def read_split(data_dir: str, split: str) -> Dict[str, pd.DataFrame]:
    """Load s1/s2/s3 for 'train' or 'test' from data_dir/<split>/<split>_sourceN.tsv.

    Loads EVERYTHING into memory (several GB for the real data). Use only on
    small/synthetic data; full-size analyses must stream with iter_source.
    """
    out = {}
    for n in (1, 2, 3):
        out[f"s{n}"] = read_source(os.path.join(data_dir, split, f"{split}_source{n}.tsv"))
    return out


def parse_id_list(cell: str) -> List[str]:
    cell = (cell or "").strip()
    if not cell:
        return []
    return [x.strip() for x in cell.split(",") if x.strip()]


def read_id_list_file(path: str) -> Dict[str, List[str]]:
    """Read ground truth / matching_results / candidate_pairs into {s1: [ids]}."""
    out: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            if not line.strip():
                continue
            s1, _, rest = line.rstrip("\n").partition("\t")
            out[s1.strip()] = parse_id_list(rest)
    return out


def read_ground_truth(data_dir: str) -> Dict[str, List[str]]:
    return read_id_list_file(os.path.join(data_dir, "train", "train_ground_truth.tsv"))


def _clean_ids(ids: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for i in ids:
        i = str(i).strip()
        if i and i not in seen and i.startswith(("S2-", "S3-")):
            seen.add(i)
            out.append(i)
    return out


def write_id_list_file(
    path: str,
    mapping: Mapping[str, Iterable[str]],
    all_s1_ids: Iterable[str],
    header_col: str,
) -> None:
    """Write a submission-style TSV with exactly one row per S1 id, in input order.

    Enforces the scorer's rules: dedup within lists, S2/S3 ids only, no quoting.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_col}\n")
        written = set()
        for s1 in all_s1_ids:
            if s1 in written:
                continue
            written.add(s1)
            f.write(f"{s1}\t{','.join(_clean_ids(mapping.get(s1, [])))}\n")


def write_matching(path, mapping, all_s1_ids):
    write_id_list_file(path, mapping, all_s1_ids, "matched_entity_ids")


def write_candidates(path, mapping, all_s1_ids):
    write_id_list_file(path, mapping, all_s1_ids, "candidate_entity_ids")
