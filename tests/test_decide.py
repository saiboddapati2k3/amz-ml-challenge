import itertools
import numpy as np
import pandas as pd
import pytest
from src.decide import expected_f_curve, select_expected_f, apply_one_owner, decide_expected_f
from src.eval.score import f_beta_sets


def brute_force(p, k):
    """Enumerate every truth world; predict the top-k (p already sorted desc)."""
    ids = list(range(len(p)))
    total = 0.0
    for world in itertools.product([0, 1], repeat=len(p)):
        prob = np.prod([pi if w else 1 - pi for pi, w in zip(p, world)])
        truth = [i for i, w in zip(ids, world) if w]
        total += prob * f_beta_sets(ids[:k], truth)
    return total


@pytest.mark.parametrize("p", [[0.9, 0.6, 0.3, 0.1], [0.5], [0.2, 0.2, 0.2], [0.99, 0.95, 0.9, 0.4, 0.05]])
def test_expected_f_exact(p):
    curve = expected_f_curve(p)
    for k in range(len(p) + 1):
        assert curve[k] == pytest.approx(brute_force(p, k), abs=1e-9)


def test_singleton_preferred_when_weak():
    ids, _ = select_expected_f(["S2-1", "S2-2"], [0.3, 0.2])
    assert ids == []


def test_multi_match_selected():
    ids, _ = select_expected_f(["S2-1", "S3-9", "S2-5"], [0.95, 0.9, 0.05])
    assert ids == ["S2-1", "S3-9"]


def test_one_owner_hard_and_soft():
    df = pd.DataFrame({"s1_id": ["a", "b"], "cand_id": ["S2-1", "S2-1"], "p": [0.9, 0.6]})
    hard = apply_one_owner(df, "hard")
    assert hard.p.tolist() == [0.9, 0.0]
    soft = apply_one_owner(df, "soft")
    assert soft.p.sum() == pytest.approx(1.0)


def test_every_s1_present():
    df = pd.DataFrame({"s1_id": ["a"], "cand_id": ["S2-1"], "p": [0.99]})
    out = decide_expected_f(df, ["a", "b"])
    assert out == {"a": ["S2-1"], "b": []}
