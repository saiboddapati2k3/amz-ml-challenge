"""Validation splits. Always split by Source-1 entity, never by pair row.

* assign_folds: deterministic K-fold over S1 ids, stratified by country and by
  whether the entity is a singleton, so every fold has the same singleton share.
* loco_splits: leave-one-country-out, the stand-in for the unseen test country.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np


def _stable_rank(s: str, seed: int) -> int:
    return int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest()[:12], 16)


def assign_folds(
    s1_ids: Iterable[str],
    country: Mapping[str, str],
    truth: Mapping[str, List[str]],
    k: int = 5,
    seed: int = 42,
) -> Dict[str, int]:
    """Return {s1_id: fold} stratified by (country, match-count bucket)."""
    strata: Dict[Tuple[str, int], List[str]] = {}
    for s in s1_ids:
        n = len(truth.get(s, []))
        bucket = min(n, 3)  # 0, 1, 2, 3+
        strata.setdefault((country.get(s, ""), bucket), []).append(s)
    folds = {}
    for members in strata.values():
        members.sort(key=lambda s: _stable_rank(s, seed))
        for i, s in enumerate(members):
            folds[s] = i % k
    return folds


def loco_splits(s1_ids: Iterable[str], country: Mapping[str, str]):
    """Yield (held_out_country, train_ids, test_ids) for each country present."""
    ids = list(s1_ids)
    countries = sorted({country.get(s, "") for s in ids})
    for c in countries:
        test = [s for s in ids if country.get(s, "") == c]
        train = [s for s in ids if country.get(s, "") != c]
        yield c, train, test


def fold_array(pair_s1_ids: np.ndarray, folds: Mapping[str, int]) -> np.ndarray:
    """Map a column of pair-level S1 ids to fold numbers (for GroupKFold-style CV)."""
    return np.array([folds[s] for s in pair_s1_ids], dtype=np.int16)
