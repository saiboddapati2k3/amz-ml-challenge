"""Constant-memory check of candidate_pairs.tsv (+ matched subset of candidates).

utils/validate_submission.py keeps every listed ID in Python sets, which for ~72M candidate
IDs needs several GB. This applies the same per-file rules to candidate_pairs.tsv while
streaming it in lockstep with matching_results.tsv (both are written in test_source1 order):

  header, one row per required S1 (no duplicates, none missing, none extra), IDs S2-/S3- only,
  no repeated ID within a list, and every matched ID present in that S1's candidate list.

Run the official validator on the scored file (with --check-ids and -c none) separately.

    python scripts/check_submission_stream.py -m output/matching_results.tsv -c output/candidate_pairs.tsv -t dataset/test
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--matching", default="output/matching_results.tsv")
ap.add_argument("-c", "--candidate", default="output/candidate_pairs.tsv")
ap.add_argument("-t", "--test-dir", default="dataset/test")
args = ap.parse_args()

with open(os.path.join(args.test_dir, "test_source1.tsv"), encoding="utf-8") as f:
    next(f)
    required = {line.split("\t", 1)[0].strip() for line in f if line.strip()}

errors = []
bad = {"dup_row": [], "intra_dup": [], "prefix": [], "not_subset": [], "order": [], "malformed": []}
seen = set()
n_ids = n_rows = empties = 0


def ids_of(rest):
    rest = rest.rstrip("\n")
    return rest.split(",") if rest.strip() else []


with open(args.matching, encoding="utf-8") as fm, open(args.candidate, encoding="utf-8") as fc:
    hm, hc = fm.readline().rstrip("\n").split("\t"), fc.readline().rstrip("\n").split("\t")
    if hc != ["source1_entity_id", "candidate_entity_ids"]:
        errors.append(f"candidate header {hc}")
    if hm != ["source1_entity_id", "matched_entity_ids"]:
        errors.append(f"matching header {hm}")
    for n, (lm, lc) in enumerate(zip(fm, fc), start=2):
        s1c, tab, rest = lc.partition("\t")
        s1m, tabm, restm = lm.partition("\t")
        if not tab or not tabm:
            bad["malformed"].append(n)
            continue
        n_rows += 1
        if s1c != s1m:
            bad["order"].append(n)
        if s1c in seen:
            bad["dup_row"].append(s1c)
        seen.add(s1c)
        cand = ids_of(rest)
        n_ids += len(cand)
        if not cand:
            empties += 1
        cset = set(cand)
        if len(cset) != len(cand):
            bad["intra_dup"].append(s1c)
        if any(not x.startswith(("S2-", "S3-")) for x in cset):
            bad["prefix"].append(s1c)
        if not set(ids_of(restm)) <= cset:
            bad["not_subset"].append(s1c)
    if fm.readline() or fc.readline():
        errors.append("files have different row counts")

missing, extra = required - seen, seen - required
print(f"candidate_pairs.tsv: {n_rows:,} rows ({empties:,} empty), {n_ids:,} candidate IDs "
      f"({n_ids / max(n_rows, 1):.1f}/S1); required S1 {len(required):,}")
for k, v in bad.items():
    if v:
        errors.append(f"{k}: {len(v)} e.g. {v[:5]}")
if missing:
    errors.append(f"missing S1: {len(missing)} e.g. {sorted(missing)[:5]}")
if extra:
    errors.append(f"unknown S1: {len(extra)} e.g. {sorted(extra)[:5]}")
print("ERRORS:\n  " + "\n  ".join(errors) if errors else "OK: all candidate rules pass; matches are a subset of candidates")
sys.exit(1 if errors else 0)
