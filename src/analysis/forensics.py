"""Day-1 forensic analysis of the TRAIN split. Writes reports/forensics.md.

python -m src.analysis.forensics --data dataset --out reports/forensics.md
    [--chunksize 100000] [--sample-size 100000] [--max-memory-safe]

Answers the questions that decide the architecture: singleton share, match-count
distribution, S2/S3 mix, one-owner validity, how much exact/fuzzy evidence true
pairs carry, how common look-alike non-matches are, whether S2<->S3 corroborate,
and which token substitutions the noise process actually uses.

Memory architecture (the full train split is ~12.5M source rows + 7.6M GT pairs):
  * Nothing is ever loaded whole. Sources are streamed with iter_source; each
    chunk is normalized, reduced to compact numpy state, then dropped.
  * Entity ids are int64 (io_utils.encode_id); normalized views are kept only as
    64-bit hashes (pd.util.hash_array), so "equal view" / "value frequency" are
    exact up to a ~1e-19 per-pair hash collision chance.
  * EXACT over the full data: sizes, missingness, match counts, source mix,
    one-owner, within-source uniqueness, exact-equality evidence on every true
    pair, and look-alike (same country + name_core) counts.
  * SAMPLED (stratified, weighted back to the population): fuzzy similarity,
    hardest positives, substitution mining, cross-source corroboration, and
    look-alike negative examples. A final pass re-reads the sources and keeps
    only the few hundred thousand sampled rows.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.io_utils import ID_BASE, decode_id, encode_ids, iter_source, read_id_list_pairs, source_path
from src.normalize import normalize_frame

try:
    import psutil
except ImportError:  # memory reporting is optional
    psutil = None

# bit flags per record
F_NAME_EMPTY, F_ADDR_EMPTY, F_NO_DIGITS, F_NO_POSTAL = 1, 2, 4, 8
# bit flags per true pair (exact evidence)
E_NAME_CORE, E_NAME_SORTED, E_NAME_COMPACT, E_ADDR, E_POSTAL_EQ, E_POSTAL_CONFLICT, E_COUNTRY = (
    1, 2, 4, 8, 16, 32, 64)
EVIDENCE_BITS = [
    ("name_core equal", E_NAME_CORE), ("name_sorted equal", E_NAME_SORTED),
    ("name_compact equal", E_NAME_COMPACT), ("addr_norm equal", E_ADDR),
    ("postal equal (both present)", E_POSTAL_EQ),
    ("postal CONFLICT (both present, differ)", E_POSTAL_CONFLICT), ("country equal", E_COUNTRY),
]
LOOKALIKE_MAX_GROUP = 50  # ignore very common names, as before


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def h64(values) -> np.ndarray:
    """64-bit hash of a chunk's string column (one value per row)."""
    return pd.util.hash_array(np.asarray(values, dtype=object))


class Monitor:
    """Tracks runtime, rows, chunks and peak memory of the process."""

    def __init__(self):
        self.t0 = time.time()
        self.proc = psutil.Process() if psutil else None
        self.peak_rss = 0
        self.rows: Dict[str, int] = collections.Counter()
        self.chunks: Dict[str, int] = collections.Counter()
        self.phases: List[tuple] = []
        self._pt = self.t0

    def rss_mb(self) -> float:
        if not self.proc:
            return float("nan")
        mi = self.proc.memory_info()
        self.peak_rss = max(self.peak_rss, mi.rss, getattr(mi, "peak_wset", 0))
        return mi.rss / 2 ** 20

    def peak_mb(self) -> float:
        self.rss_mb()
        return self.peak_rss / 2 ** 20 if self.proc else float("nan")

    def chunk(self, name: str, n: int):
        self.rows[name] += n
        self.chunks[name] += 1
        self.rss_mb()

    def phase(self, name: str):
        now = time.time()
        self.phases.append((name, now - self._pt, self.rss_mb()))
        print(f"[{now - self.t0:7.1f}s] {name}: {now - self._pt:.1f}s, rss {self.rss_mb():.0f} MB, "
              f"peak {self.peak_mb():.0f} MB", flush=True)
        self._pt = now


def with_retry(fn, tries: int = 4, wait_s: float = 2.0):
    """Run fn(); on a transient allocation failure free memory, wait, and retry.

    Other processes on a small machine can briefly exhaust the system commit
    limit; a per-chunk retry keeps a 10-minute run from dying on that. fn must be
    idempotent (all per-chunk work here is).
    """
    for attempt in range(tries):
        try:
            return fn()
        except MemoryError:
            if attempt == tries - 1:
                raise
            gc.collect()
            print(f"  MemoryError, retry {attempt + 1}/{tries - 1} after {wait_s * (attempt + 1):.0f}s", flush=True)
            time.sleep(wait_s * (attempt + 1))


def mine_substitutions(a_vals, b_vals, top: int = 40):
    """Count 1-for-1 token swaps between matched fields (the learned noise dictionary)."""
    cnt = collections.Counter()
    for x, y in zip(a_vals, b_vals):
        tx, ty = x.split(), y.split()
        sx, sy = set(tx), set(ty)
        dx, dy = [t for t in tx if t not in sy], [t for t in ty if t not in sx]
        if len(dx) == 1 and len(dy) == 1:
            cnt[tuple(sorted((dx[0], dy[0])))] += 1
    return cnt.most_common(top)


def weighted_quantiles(x: np.ndarray, w: np.ndarray, qs) -> List[float]:
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return [float(x[o][min(np.searchsorted(cw, q / 100.0), len(x) - 1)]) for q in qs]


def value_freq(h: np.ndarray) -> np.ndarray:
    """For each element, how many elements of h share its value (exact, via sort)."""
    _, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
    return cnt[inv].astype(np.int32)


def expand_matches(sorted_keys: np.ndarray, query: np.ndarray):
    """For each query value, all positions in sorted_keys equal to it.

    Returns (query_idx, key_pos) arrays, one entry per match (handles duplicates).
    """
    lo = np.searchsorted(sorted_keys, query, "left")
    hi = np.searchsorted(sorted_keys, query, "right")
    cnt = hi - lo
    q_idx = np.repeat(np.arange(len(query)), cnt)
    starts = np.repeat(lo - np.cumsum(cnt) + cnt, cnt)
    return q_idx, starts + np.arange(cnt.sum())


# --------------------------------------------------------------------------- #
# per-source streaming pass
# --------------------------------------------------------------------------- #
class Countries:
    def __init__(self):
        self.code: Dict[str, int] = {}

    def encode(self, s: pd.Series) -> np.ndarray:
        for c in s.unique():
            self.code.setdefault(c, len(self.code))
        return s.map(self.code).to_numpy(np.int16)

    def names(self) -> List[str]:
        return sorted(self.code, key=self.code.get)


def chunk_state(n: pd.DataFrame, countries: Countries) -> Dict[str, np.ndarray]:
    """Reduce one normalized chunk to compact per-record arrays."""
    name_core = n["name_core"].to_numpy(object)
    addr_norm = n["addr_norm"].to_numpy(object)
    postal = n["addr_postal"].to_numpy(object)
    flags = ((n["business_name"] == "").to_numpy() * F_NAME_EMPTY
             | (n["business_address"] == "").to_numpy() * F_ADDR_EMPTY
             | (n["addr_digits"] == "").to_numpy() * F_NO_DIGITS
             | (postal == "") * F_NO_POSTAL).astype(np.int8)
    key = (n["country_norm"] + "\x1f" + n["name_core"]).to_numpy(object)
    name_addr = (n["name_core"] + "\x1f" + n["addr_norm"]).to_numpy(object)
    return {
        "id": encode_ids(n["entity_id"]),
        "country": countries.encode(n["country"]),
        "flags": flags,
        "h_name_core": h64(name_core),
        "h_name_sorted": h64(n["name_sorted"]),
        "h_name_compact": h64(n["name_compact"]),
        "h_addr": h64(addr_norm),
        "h_postal": h64(postal),
        "h_name_addr": h64(name_addr),
        "h_key": h64(key),  # country_norm + name_core: the look-alike blocking key
    }


def source_pass(path: str, label: str, args, countries: Countries, mon: Monitor,
                keep: List[str], on_chunk=None) -> Dict[str, object]:
    """Stream one source; return per-record arrays listed in `keep` + exact stats."""
    parts: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
    uniq_parts: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
    for chunk in iter_source(path, args.chunksize):
        st = with_retry(lambda: chunk_state(
            normalize_frame(chunk[["entity_id", "business_name", "business_address", "country"]]), countries))
        del chunk
        if on_chunk is not None:
            with_retry(lambda: on_chunk(st))
        for k in keep:
            if k in st:  # derived arrays (name_freq) are added after the pass
                parts[k].append(st[k])
        for k in ("h_name_core", "h_addr", "h_name_addr"):
            uniq_parts[k].append(st[k])
        if "country" not in keep:
            parts["_country"].append(st["country"])
        if "flags" not in keep:
            parts["_flags"].append(st["flags"])
        del st
        mon.chunk(label, len(parts["_country" if "country" not in keep else "country"][-1]))
        if mon.chunks[label] % 10 == 0:
            print(f"  {label}: {mon.rows[label]:,} rows, rss {mon.rss_mb():.0f} MB", flush=True)

    out = {k: np.concatenate(v) for k, v in parts.items()}
    parts.clear()
    country = out.pop("_country", None)
    country = out["country"] if country is None else country
    flags = out.pop("_flags", None)
    flags = out["flags"] if flags is None else flags

    stats = {"records": len(country),
             "country_counts": {c: int(x) for c, x in zip(countries.names(), np.bincount(country, minlength=len(countries.code)))
                                if x},
             "empty_name": float(np.mean(flags & F_NAME_EMPTY > 0)),
             "empty_addr": float(np.mean(flags & F_ADDR_EMPTY > 0)),
             "no_digits": float(np.mean(flags & F_NO_DIGITS > 0)),
             "no_postal": float(np.mean(flags & F_NO_POSTAL > 0))}
    uniq = {}
    for k, label_v in (("h_name_core", "name_core"), ("h_addr", "addr_norm"), ("h_name_addr", "name+addr")):
        h = np.concatenate(uniq_parts.pop(k))
        f = value_freq(h)
        del h
        uniq[label_v] = (float(np.mean(f == 1)), float(np.mean(f >= 10)))
        if k == "h_name_core" and "name_freq" in keep:
            out["name_freq"] = f
        del f
    stats["uniqueness"] = uniq
    gc.collect()
    return {"arrays": out, "stats": stats}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="reports/forensics.md")
    ap.add_argument("--chunksize", type=int, default=50_000, help="rows per streamed chunk")
    ap.add_argument("--sample-size", type=int, default=100_000,
                    help="true pairs sampled (stratified) for fuzzy / text-level stats")
    ap.add_argument("--corroboration-sample", type=int, default=20_000,
                    help="S1 entities (matched in S2 AND S3) sampled for cross-source stats")
    ap.add_argument("--lookalike-sample", type=int, default=2_000,
                    help="S1 entities with same-name non-matches sampled for negative stats")
    ap.add_argument("--max-memory-safe", action="store_true",
                    help="conservative settings for small machines (chunksize<=25k, samples<=30k)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.max_memory_safe:
        args.chunksize = min(args.chunksize, 20_000)
        args.sample_size = min(args.sample_size, 30_000)
        args.corroboration_sample = min(args.corroboration_sample, 5_000)
        args.lookalike_sample = min(args.lookalike_sample, 500)
    rng = np.random.default_rng(args.seed)
    mon = Monitor()
    countries = Countries()
    rep = ["# Forensic analysis (train)\n"]
    if psutil:
        vm = psutil.virtual_memory()
        print(f"system RAM {vm.total / 2**30:.1f} GB, available {vm.available / 2**30:.1f} GB; "
              f"chunksize {args.chunksize:,}, sample {args.sample_size:,}", flush=True)

    # ---- ground truth -> compact pair arrays ------------------------------- #
    gt_rows, pair_s1, pair_cand = read_id_list_pairs(os.path.join(args.data, "train", "train_ground_truth.tsv"))
    n_raw_pairs = len(pair_s1)
    comb = np.unique(np.stack([pair_s1, pair_cand], 1), axis=0)  # dedupe (s1, cand), like the old set()
    pair_s1, pair_cand = np.ascontiguousarray(comb[:, 0]), np.ascontiguousarray(comb[:, 1])
    del comb
    n_pairs = len(pair_s1)
    pair_src = (pair_cand // ID_BASE).astype(np.int8)  # 2 / 3
    mon.chunk("gt", n_raw_pairs)
    mon.phase(f"ground truth: {len(gt_rows):,} rows, {n_pairs:,} pairs")

    # ---- S1 pass ----------------------------------------------------------- #
    s1_keep = ["id", "country", "flags", "h_name_core", "h_name_sorted", "h_name_compact",
               "h_addr", "h_postal", "h_key", "name_freq"]
    r1 = source_pass(source_path(args.data, "train", 1), "s1", args, countries, mon, s1_keep)
    S1 = r1["arrays"]
    stats = {"s1": r1["stats"]}
    n_s1 = len(S1["id"])
    s1_order = np.argsort(S1["id"], kind="stable").astype(np.int32)
    s1_sorted = S1["id"][s1_order]
    pos = np.searchsorted(s1_sorted, pair_s1).clip(max=n_s1 - 1)
    ok = s1_sorted[pos] == pair_s1
    pair_s1_pos = np.where(ok, s1_order[pos], -1).astype(np.int32)
    gt_s1_missing = int(np.unique(pair_s1[~ok]).size)
    del pos, ok
    mon.phase(f"S1 pass: {n_s1:,} rows")

    # ---- S2/S3 passes: exact evidence on every true pair ------------------- #
    cand_perm = np.argsort(pair_cand, kind="stable").astype(np.int32)
    cand_sorted = pair_cand[cand_perm]
    pair_bits = np.zeros(n_pairs, np.uint8)
    pair_found = np.zeros(n_pairs, bool)
    pair_same_key = np.zeros(n_pairs, bool)
    pair_cflags = np.zeros(n_pairs, np.int8)
    S1_has_postal = (S1["flags"] & F_NO_POSTAL) == 0

    def evidence(st):
        qi, kp = expand_matches(cand_sorted, st["id"])
        pi = cand_perm[kp]
        sp = pair_s1_pos[pi]
        good = sp >= 0
        qi, pi, sp = qi[good], pi[good], sp[good]
        pair_found[pi] = True
        b_has_postal = (st["flags"][qi] & F_NO_POSTAL) == 0
        a_has_postal = S1_has_postal[sp]
        postal_eq = S1["h_postal"][sp] == st["h_postal"][qi]
        bits = ((S1["h_name_core"][sp] == st["h_name_core"][qi]) * E_NAME_CORE
                | (S1["h_name_sorted"][sp] == st["h_name_sorted"][qi]) * E_NAME_SORTED
                | (S1["h_name_compact"][sp] == st["h_name_compact"][qi]) * E_NAME_COMPACT
                | (S1["h_addr"][sp] == st["h_addr"][qi]) * E_ADDR
                | (postal_eq & a_has_postal) * E_POSTAL_EQ
                | (~postal_eq & a_has_postal & b_has_postal) * E_POSTAL_CONFLICT
                | (S1["country"][sp] == st["country"][qi]) * E_COUNTRY)
        pair_bits[pi] = bits.astype(np.uint8)
        pair_same_key[pi] = S1["h_key"][sp] == st["h_key"][qi]
        pair_cflags[pi] = st["flags"][qi]

    other_parts = collections.defaultdict(list)
    for n in (2, 3):
        r = source_pass(source_path(args.data, "train", n), f"s{n}", args, countries, mon,
                        ["id", "h_key", "flags"], on_chunk=evidence)
        stats[f"s{n}"] = r["stats"]
        for k, v in r["arrays"].items():
            other_parts[k].append(v)
        del r
        gc.collect()
        mon.phase(f"S{n} pass: {stats[f's{n}']['records']:,} rows")
    O = {k: np.concatenate(v) for k, v in other_parts.items()}
    del other_parts, cand_sorted, cand_perm
    n_others = len(O["id"])

    # ---- 1. sizes and country mix ------------------------------------------ #
    rows = [{"source": k, "records": v["records"],
             **{f"country={c}": v["country_counts"].get(c, 0) for c in countries.names()}}
            for k, v in stats.items()]
    rep += ["## 1. Sizes and countries\n", md_table(pd.DataFrame(rows)), ""]

    # ---- 2-4. match counts, singletons, source mix ------------------------- #
    valid = pair_s1_pos >= 0
    n_match = np.bincount(pair_s1_pos[valid], minlength=n_s1)
    n2 = np.bincount(pair_s1_pos[valid & (pair_src == 2)], minlength=n_s1)
    n3 = np.bincount(pair_s1_pos[valid & (pair_src == 3)], minlength=n_s1)
    cnames = countries.names()
    bucket = np.minimum(n_match, 5)
    tab = {}
    for ci, c in enumerate(cnames):
        m = S1["country"] == ci
        if m.any():
            tab[c] = [pct(np.mean(bucket[m] == b)) for b in range(6)]
    tab_df = pd.DataFrame(tab, index=["0", "1", "2", "3", "4", "5+"]).rename_axis("bucket").reset_index()
    rep += ["## 2-3. Matches per S1 (share of S1 entities, by country)\n", md_table(tab_df), "",
            f"Overall singleton share: **{pct(np.mean(n_match == 0))}**, "
            f"mean matches/S1: {n_match.mean():.3f}, max: {n_match.max()}\n"]
    pattern = np.where((n2 > 0) & (n3 > 0), "S2+S3", np.where(n2 > 0, "S2", np.where(n3 > 0, "S3", "none")))
    pv, pc = np.unique(pattern, return_counts=True)
    rep += ["## 4. Which sources an S1 matches\n",
            md_table(pd.DataFrame([{"pattern": k, "share": pct(v / n_s1)} for k, v in zip(pv, pc)])), ""]
    del pattern

    # ---- one-owner and distractors ----------------------------------------- #
    uc, owners = np.unique(pair_cand, return_counts=True)  # pairs are deduped -> distinct S1 per cand
    o_sorted = np.sort(O["id"])
    in_src = np.isin(uc, o_sorted, assume_unique=True)
    orphan_share = 1 - in_src.sum() / max(n_others, 1)
    rep += ["## One-owner and distractors\n",
            f"- S2/S3 records matched to >1 S1: **{int((owners > 1).sum())}** (expect 0 if S1 is deduplicated)",
            f"- S2/S3 records matched to no S1 (pure distractors): **{pct(orphan_share)}**",
            f"- GT ids missing from source files: {int((~in_src).sum())}; GT S1 ids missing from S1 file: "
            f"{gt_s1_missing}; duplicate (s1, cand) rows in GT: {n_raw_pairs - n_pairs}\n"]
    del uc, owners, in_src, o_sorted

    # ---- 8. missingness ---------------------------------------------------- #
    miss_rows = []
    for k, v in stats.items():
        miss_rows.append({"source": k, "field": "business_name", "empty": pct(v["empty_name"]),
                          "no_digits_in_addr": "", "no_postal": ""})
        miss_rows.append({"source": k, "field": "business_address", "empty": pct(v["empty_addr"]),
                          "no_digits_in_addr": pct(v["no_digits"]), "no_postal": pct(v["no_postal"])})
    rep += ["## 8. Missingness\n", md_table(pd.DataFrame(miss_rows)), ""]

    # ---- 5-7. uniqueness within source ------------------------------------- #
    u_rows = []
    for k, v in stats.items():
        for view, (u, g10) in v["uniqueness"].items():
            u_rows.append({"source": k, "view": view, "unique_share": pct(u),
                           "records_in_groups>=10": pct(g10) if view != "name+addr" else ""})
    rep += ["## 5-7. Uniqueness within source\n", md_table(pd.DataFrame(u_rows)), ""]

    # ---- 12. exact evidence on ALL true pairs ------------------------------ #
    fp = pair_found
    ev_rows = [{"signal": name, "share_of_true_pairs": pct(np.mean(pair_bits[fp] & bit > 0))}
               for name, bit in EVIDENCE_BITS]
    rep += [f"## 12. Exact evidence on TRUE pairs (all {int(fp.sum()):,} pairs, exact)\n",
            md_table(pd.DataFrame(ev_rows)), ""]
    pair_country = np.where(pair_s1_pos >= 0, S1["country"][pair_s1_pos.clip(min=0)], -1)
    brk = []
    for ci, c in enumerate(cnames):
        for s in (2, 3):
            m = fp & (pair_country == ci) & (pair_src == s)
            if m.any():
                brk.append({"s1_country": c, "cand": f"S{s}", "pairs": int(m.sum()),
                            **{name.split(" (")[0]: pct(np.mean(pair_bits[m] & bit > 0)) for name, bit in EVIDENCE_BITS}})
    rep += ["By S1 country x candidate source:\n", md_table(pd.DataFrame(brk)), ""]

    # ---- 14 / 17. look-alike non-matches (exact, via key hashes) ---------- #
    uk, kcnt = np.unique(O["h_key"], return_counts=True)
    p = np.searchsorted(uk, S1["h_key"]).clip(max=len(uk) - 1)
    grp = np.where(uk[p] == S1["h_key"], kcnt[p], 0)
    grp_small = np.where(grp <= LOOKALIKE_MAX_GROUP, grp, 0)
    del p
    true_same = np.bincount(pair_s1_pos[fp & pair_same_key], minlength=n_s1)
    true_in_M = np.where(grp_small > 0, true_same, 0)
    M_total = int(grp_small.sum())
    neg_per_s1 = grp_small - true_in_M
    s1_lookalike = neg_per_s1 > 0
    rep += ["## 14 / 17. Look-alike non-matches (same country + identical name_core; exact)\n",
            f"- Pairs sharing identical name_core (groups <= {LOOKALIKE_MAX_GROUP}): {M_total:,}; of those truly "
            f"matched: **{pct(true_in_M.sum() / max(M_total, 1))}**",
            f"- S1 entities with at least one same-name NON-match: {pct(s1_lookalike.mean())}",
            f"- True pairs whose name_core key differs (a name blocker alone misses them): "
            f"{pct(1 - np.mean(pair_same_key[fp]))}",
            "- => precision of a naive 'same normalized name' rule; the gap is what address/rarity features must fix.\n"]
    mon.phase("exact statistics")

    # ---- stratified pair sample -------------------------------------------- #
    # strata: S1 country x cand source x (either side missing address) x S1 name
    # frequency (unique / 2-9 / 10+) x S1 has look-alike non-match. Floors keep
    # rare strata visible; weights (stratum size / stratum sample) undo the floors.
    sp_ = pair_s1_pos.clip(min=0)
    miss_any = ((S1["flags"][sp_] | pair_cflags) & F_ADDR_EMPTY) > 0
    freq_b = np.digitize(S1["name_freq"][sp_], [2, 10])
    stratum = (((pair_country.astype(np.int64) * 4 + pair_src) * 2 + miss_any) * 3 + freq_b) * 2 + s1_lookalike[sp_]
    stratum = np.where(fp, stratum, -1)
    del miss_any, freq_b, sp_
    s_ids, s_sizes = np.unique(stratum[fp], return_counts=True)
    floor = max(50, args.sample_size // (4 * max(len(s_ids), 1)))
    alloc = np.minimum(s_sizes, np.maximum(np.round(args.sample_size * s_sizes / s_sizes.sum()).astype(int), floor))
    order = np.lexsort((rng.random(n_pairs), stratum))
    st_sorted = stratum[order]
    starts = np.searchsorted(st_sorted, s_ids)
    samp = np.concatenate([order[a:a + k] for a, k in zip(starts, alloc)])
    weight_by_stratum = dict(zip(s_ids.tolist(), (s_sizes / alloc).tolist()))
    samp_w = np.array([weight_by_stratum[s] for s in stratum[samp]])
    del order, st_sorted, stratum
    # strata summary (collapsed to country x source)
    samp_country, samp_src = pair_country[samp], pair_src[samp]

    # corroboration sample: S1 matched in both S2 and S3, one of each
    both = np.flatnonzero((n2 > 0) & (n3 > 0))
    both = rng.choice(both, min(args.corroboration_sample, len(both)), replace=False) if len(both) else both
    corr = {}
    if len(both):
        sel = np.isin(pair_s1_pos, both) & fp
        for s in (2, 3):
            idx = np.flatnonzero(sel & (pair_src == s))
            first = np.unique(pair_s1_pos[idx], return_index=True)
            corr[s] = dict(zip(first[0].tolist(), pair_cand[idx[first[1]]].tolist()))
        both = np.array([b for b in both if b in corr[2] and b in corr[3]], dtype=np.int64)

    # look-alike sample: S1 with same-key non-matches
    la_s1 = np.flatnonzero(s1_lookalike)
    la_s1 = rng.choice(la_s1, min(args.lookalike_sample, len(la_s1)), replace=False) if len(la_s1) else la_s1
    la_keys = np.unique(S1["h_key"][la_s1])

    need_ids = np.unique(np.concatenate([
        pair_s1[samp], pair_cand[samp], S1["id"][both], S1["id"][la_s1],
        np.fromiter(corr.get(2, {}).values(), np.int64), np.fromiter(corr.get(3, {}).values(), np.int64)]))
    need_other_by_key = np.isin(O["h_key"], la_keys)
    need_ids = np.unique(np.concatenate([need_ids, O["id"][need_other_by_key]]))
    la_pairs = np.isin(pair_s1_pos, la_s1) & fp  # true pairs of the SAMPLED look-alike S1s only
    true_set = set(zip(pair_s1[la_pairs].tolist(), pair_cand[la_pairs].tolist()))
    del la_pairs
    la_s1_ids = S1["id"][la_s1]
    s1_key_of = dict(zip(la_s1_ids.tolist(), S1["h_key"][la_s1].tolist()))
    del need_other_by_key
    mon.phase(f"sampling: {len(samp):,} pairs, {len(both):,} corroboration S1s, {len(la_s1):,} look-alike S1s; "
              f"{len(need_ids):,} records to fetch")

    # ---- fetch pass: re-stream sources, keep only sampled rows ------------- #
    rec_parts = []
    for n in (1, 2, 3):
        for chunk in iter_source(source_path(args.data, "train", n), args.chunksize):
            ids = encode_ids(chunk["entity_id"])
            m = np.isin(ids, need_ids)
            if m.any():
                sub = chunk.loc[m, ["entity_id", "business_name", "business_address", "country"]].copy()
                sub["id"] = ids[m]
                rec_parts.append(with_retry(lambda: normalize_frame(sub)))
            mon.chunk(f"fetch_s{n}", len(ids))
            del chunk, ids, m
    R = pd.concat(rec_parts, ignore_index=True).drop_duplicates("id").set_index("id")
    del rec_parts
    gc.collect()
    mon.phase(f"fetch pass: {len(R):,} records kept")

    # ---- 13. fuzzy evidence on sampled TRUE pairs (weighted) --------------- #
    a = R.loc[pair_s1[samp]]
    b = R.loc[pair_cand[samp]]
    name_tsr = np.fromiter((fuzz.token_set_ratio(x, y) for x, y in zip(a["name_core"], b["name_core"])),
                           float, len(samp))
    addr_tsr = np.fromiter((fuzz.token_set_ratio(x, y) for x, y in zip(a["addr_norm"], b["addr_norm"])),
                           float, len(samp))
    W = samp_w / samp_w.sum()
    fz = {
        "name token_set >= 90": float(W @ (name_tsr >= 90)),
        "addr token_set >= 90": float(W @ (addr_tsr >= 90)),
        "name < 60 AND addr < 60 (very hard)": float(W @ ((name_tsr < 60) & (addr_tsr < 60))),
    }
    rep += [f"## 13. Fuzzy evidence on TRUE pairs (stratified sample n={len(samp):,}, population-weighted)\n",
            md_table(pd.DataFrame([{"signal": k, "share_of_true_pairs": pct(v)} for k, v in fz.items()])), "",
            f"Name token_set percentiles (p5/p25/p50): {weighted_quantiles(name_tsr, samp_w, [5, 25, 50])}; "
            f"address: {weighted_quantiles(addr_tsr, samp_w, [5, 25, 50])}\n"]
    for ci, c in enumerate(cnames):
        for s in (2, 3):
            m = (samp_country == ci) & (samp_src == s)
            if m.any():
                rep.append(f"- {c} / S{s}: median name sim {weighted_quantiles(name_tsr[m], samp_w[m], [50])[0]:.1f}, "
                           f"median addr sim {weighted_quantiles(addr_tsr[m], samp_w[m], [50])[0]:.1f}, "
                           f"sampled n={int(m.sum())} of {int(((pair_country == ci) & (pair_src == s) & fp).sum()):,}")
    rep.append("")

    # ---- 16. hardest positives --------------------------------------------- #
    hard = np.lexsort((addr_tsr, name_tsr))[:15]
    hard_df = pd.DataFrame({"a_business_name": a["business_name"].to_numpy()[hard],
                            "b_business_name": b["business_name"].to_numpy()[hard],
                            "a_business_address": a["business_address"].to_numpy()[hard],
                            "b_business_address": b["business_address"].to_numpy()[hard],
                            "name_tsr": name_tsr[hard], "addr_tsr": addr_tsr[hard]})
    rep += ["## 16. Hardest true pairs in the sample (lowest name similarity)\n", md_table(hard_df), ""]

    # ---- 14b. look-alike negatives: examples + how address separates them -- #
    Rk = R[R.index.to_numpy() // ID_BASE != 1]  # fetched S2/S3 records
    Rk = Rk.assign(h_key=h64(Rk["country_norm"] + "\x1f" + Rk["name_core"]))
    by_key = {k: g for k, g in Rk.groupby("h_key", sort=False)}
    neg_rows, neg_addr, pos_addr = [], [], []
    for s1_id in la_s1_ids.tolist():
        g = by_key.get(s1_key_of[s1_id])
        if g is None or len(g) > LOOKALIKE_MAX_GROUP:
            continue
        ra = R.loc[s1_id]
        for oid, rb in g.iterrows():
            sim = fuzz.token_set_ratio(ra["addr_norm"], rb["addr_norm"])
            if (s1_id, oid) in true_set:
                pos_addr.append(sim)
            else:
                neg_addr.append(sim)
                if len(neg_rows) < 15:
                    neg_rows.append({"business_name_a": ra["business_name"], "business_name_b": rb["business_name"],
                                     "business_address_a": ra["business_address"],
                                     "business_address_b": rb["business_address"], "addr_tsr": sim})
    if neg_addr:
        rep += [f"Sampled {len(la_s1):,} S1s with look-alikes: {len(neg_addr):,} same-name NON-matches, "
                f"{len(pos_addr):,} same-name true matches.",
                f"- addr token_set median: non-match {np.median(neg_addr):.1f} vs true {np.median(pos_addr) if pos_addr else float('nan'):.1f}; "
                f"non-matches with addr >= 90: {pct(np.mean(np.array(neg_addr) >= 90))}\n",
                md_table(pd.DataFrame(neg_rows)), ""]

    # ---- 19. cross-source corroboration ------------------------------------ #
    if len(both):
        r1_ = R.loc[S1["id"][both], "name_core"].to_numpy()
        r2_ = R.loc[[corr[2][x] for x in both.tolist()], "name_core"].to_numpy()
        r3_ = R.loc[[corr[3][x] for x in both.tolist()], "name_core"].to_numpy()
        sims = np.array([(fuzz.token_set_ratio(x, y), fuzz.token_set_ratio(x, z), fuzz.token_set_ratio(y, z))
                         for x, y, z in zip(r1_, r2_, r3_)])
        rep += ["## 19. Cross-source corroboration (S1 matched in both S2 and S3; random sample)\n",
                f"- n={len(sims):,} of {int(((n2 > 0) & (n3 > 0)).sum()):,}; median name sim S1-S2 {np.median(sims[:, 0]):.1f}, "
                f"S1-S3 {np.median(sims[:, 1]):.1f}, S2-S3 {np.median(sims[:, 2]):.1f}",
                f"- Cases where S2-S3 is MORE similar than the weaker S1 link: "
                f"{pct(np.mean(sims[:, 2] > sims[:, :2].min(axis=1)))} (=> corroboration feature has signal)\n"]

    # ---- mined noise dictionary (sample) ----------------------------------- #
    rep += [f"## Mined 1-for-1 token substitutions (true pairs, sample n={len(samp):,})\n", "Names:\n"]
    rep += [f"- `{x}` <-> `{y}`: {c}" for (x, y), c in mine_substitutions(a["name_norm"], b["name_norm"])]
    rep += ["", "Addresses:\n"]
    rep += [f"- `{x}` <-> `{y}`: {c}" for (x, y), c in mine_substitutions(a["addr_norm"], b["addr_norm"])]
    mon.phase("sampled statistics")

    # ---- run report -------------------------------------------------------- #
    run = {
        "runtime_s": round(time.time() - mon.t0, 1),
        "peak_memory_mb": round(mon.peak_mb(), 0),
        "rows_processed": dict(mon.rows),
        "chunks_processed": dict(mon.chunks),
        "chunksize": args.chunksize,
        "max_memory_safe": args.max_memory_safe,
        "samples": {"true_pairs": int(len(samp)), "strata": int(len(s_ids)), "stratum_floor": int(floor),
                    "corroboration_s1": int(len(both)), "lookalike_s1": int(len(la_s1)),
                    "fetched_records": int(len(R))},
        "phases": [{"phase": p_, "seconds": round(s, 1), "rss_mb": round(m, 0)} for p_, s, m in mon.phases],
    }
    rep += ["", "## Run statistics\n",
            f"- Runtime: {run['runtime_s']}s; peak memory (process): ~{run['peak_memory_mb']:.0f} MB",
            f"- Rows processed: {run['rows_processed']}",
            f"- Chunks processed: {run['chunks_processed']} (chunksize {args.chunksize:,})",
            f"- Samples: {run['samples']}",
            "- Exact (full data): sections 1-8, 12, 14/17 counts. Sampled: 13, 16, 14b, 19, substitutions.\n"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    with open(os.path.splitext(args.out)[0] + "_run.json", "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    print("\n".join(rep))


if __name__ == "__main__":
    main()
