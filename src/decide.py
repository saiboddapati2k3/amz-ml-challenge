"""Decision layer: turn calibrated pair probabilities into final match sets.

Core idea (expected-F0.5 set selection)
---------------------------------------
For one S1 entity with candidate match probabilities p_1 >= p_2 >= ... we treat
each candidate as an independent Bernoulli(p_i) "is a true match". For every
choice "predict the top-k" (k = 0, 1, 2, ...) we compute the EXACT expected
per-entity F0.5 and output the k with the highest expectation:

  k = 0 : E[F] = P(no candidate is a match) = prod(1 - p_i)    (singleton credit)
  k > 0 : TP ~ PoissonBinomial(top-k),  FN ~ PoissonBinomial(the rest)
          F  = 1.25*TP / (1.25*TP + 0.25*FN + (k - TP)),  F = 0 when TP = 0

This single rule handles singletons, 1-vs-many and ambiguity together, and it
optimises the leaderboard metric directly. It only works well if the inputs are
CALIBRATED probabilities (fit isotonic/Platt on out-of-fold scores first).

`miss_probs` lets you add Bernoulli "true matches the blocker never retrieved"
(estimate from blocking recall on train); they can only ever be FN.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BETA2 = 0.25  # beta^2 for F0.5


def poisson_binomial_pmf(p: np.ndarray) -> np.ndarray:
    pmf = np.ones(1)
    for pi in p:
        pmf = np.convolve(pmf, [1.0 - pi, pi])
    return pmf


def _f_table(k: int, max_fn: int) -> np.ndarray:
    """F0.5 value for every (tp in 0..k, fn in 0..max_fn) given k predictions."""
    a = np.arange(k + 1)[:, None].astype(float)
    b = np.arange(max_fn + 1)[None, :].astype(float)
    num = (1 + BETA2) * a
    den = (1 + BETA2) * a + BETA2 * b + (k - a)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(a > 0, num / den, 0.0)
    return f


def expected_f_curve(
    probs: Sequence[float],
    max_k: int = 12,
    miss_probs: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Expected F0.5 for k = 0..K (K = min(max_k, n)), probs sorted descending."""
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    p = np.sort(p)[::-1]
    miss = np.clip(np.asarray(miss_probs if miss_probs is not None else [], float), 0, 1)
    n = len(p)
    K = min(max_k, n)

    all_p = np.concatenate([p, miss])
    curve = np.zeros(K + 1)
    curve[0] = float(np.prod(1.0 - all_p))
    if K == 0:
        return curve

    # suffix[j] = pmf of #matches among p[j:] plus miss (the would-be FNs)
    tail = poisson_binomial_pmf(np.concatenate([p[K:], miss]))
    suffix = [None] * (K + 1)
    suffix[K] = tail
    for j in range(K - 1, -1, -1):
        suffix[j] = np.convolve(suffix[j + 1], [1.0 - p[j], p[j]])

    tp_pmf = np.ones(1)
    for k in range(1, K + 1):
        tp_pmf = np.convolve(tp_pmf, [1.0 - p[k - 1], p[k - 1]])
        fn_pmf = suffix[k]
        f = _f_table(k, len(fn_pmf) - 1)
        curve[k] = float(tp_pmf @ f @ fn_pmf)
    return curve


def select_expected_f(
    cand_ids: Sequence[str],
    probs: Sequence[float],
    max_k: int = 12,
    miss_probs: Optional[Sequence[float]] = None,
    min_prob: float = 0.0,
) -> Tuple[List[str], float]:
    """Return (chosen ids, expected F0.5) for one S1 entity."""
    if len(cand_ids) == 0:
        return [], 1.0
    order = np.argsort(-np.asarray(probs, float), kind="stable")
    ids = [cand_ids[i] for i in order]
    p = np.asarray(probs, float)[order]
    curve = expected_f_curve(p, max_k=max_k, miss_probs=miss_probs)
    k = int(np.argmax(curve))
    chosen = [i for i, pi in zip(ids[:k], p[:k]) if pi >= min_prob]
    return chosen, float(curve[k])


# --------------------------------------------------------------------------- #
# Global constraint: S1 is deduplicated, so each S2/S3 record has <= 1 owner.
# --------------------------------------------------------------------------- #
def apply_one_owner(df: pd.DataFrame, mode: str = "soft", col: str = "p") -> pd.DataFrame:
    """Adjust probabilities so each candidate is owned by at most one S1.

    df columns: s1_id, cand_id, <col>.
    * 'hard': zero out every (s1, cand) that is not the cand's best S1.
    * 'soft': p' = p / max(1, sum_s p(s, cand)) -- keeps probabilities coherent
      (they sum to <= 1 per candidate) without discarding runner-ups entirely.
    Tune the mode on out-of-fold data; do not assume either is better.
    """
    out = df.copy()
    if mode == "none":
        return out
    g = out.groupby("cand_id")[col]
    if mode == "hard":
        best = g.transform("max")
        out[col] = np.where(out[col] >= best, out[col], 0.0)
    elif mode == "soft":
        tot = g.transform("sum").clip(lower=1.0)
        out[col] = out[col] / tot
    else:
        raise ValueError(mode)
    return out


# --------------------------------------------------------------------------- #
# Whole-dataset drivers
# --------------------------------------------------------------------------- #
def _groups(df: pd.DataFrame, col: str):
    d = df[["s1_id", "cand_id", col]].sort_values("s1_id", kind="stable")
    s1 = d["s1_id"].to_numpy()
    cand = d["cand_id"].to_numpy()
    p = d[col].to_numpy(dtype=float)
    if len(s1) == 0:
        return
    cuts = np.flatnonzero(s1[1:] != s1[:-1]) + 1
    for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(s1)]):
        yield s1[lo], cand[lo:hi], p[lo:hi]


def decide_expected_f(
    df: pd.DataFrame,
    all_s1_ids: Iterable[str],
    col: str = "p",
    max_k: int = 12,
    miss_prob: float = 0.0,
    fast_skip: float = 1e-3,
) -> Dict[str, List[str]]:
    """Apply expected-F0.5 selection to every S1. Entities with no rows -> []."""
    result: Dict[str, List[str]] = {s: [] for s in all_s1_ids}
    miss = [miss_prob] if miss_prob > 0 else None
    for s1, cands, p in _groups(df, col):
        if p.max(initial=0.0) < fast_skip:  # nothing plausible: predicting empty is optimal
            continue
        chosen, _ = select_expected_f(list(cands), p, max_k=max_k, miss_probs=miss)
        result[s1] = chosen
    return result


def decide_threshold(
    df: pd.DataFrame,
    all_s1_ids: Iterable[str],
    col: str = "p",
    tau_gate: float = 0.5,
    tau_pair: float = 0.5,
    alpha: float = 0.0,
) -> Dict[str, List[str]]:
    """Baseline: gate on best score, absolute threshold, relative cut vs best."""
    result: Dict[str, List[str]] = {s: [] for s in all_s1_ids}
    for s1, cands, p in _groups(df, col):
        best = p.max(initial=0.0)
        if best < tau_gate:
            continue
        keep = (p >= tau_pair) & (p >= alpha * best)
        order = np.argsort(-p[keep], kind="stable")
        result[s1] = list(cands[keep][order])
    return result
