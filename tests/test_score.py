import pytest
from src.eval.score import f_beta_sets, macro_f_beta


def test_readme_example():
    assert f_beta_sets(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"]) == pytest.approx(0.7142857, abs=1e-6)


def test_singleton_rules():
    assert f_beta_sets([], []) == 1.0
    assert f_beta_sets(["S2-1"], []) == 0.0
    assert f_beta_sets([], ["S2-1"]) == 0.0
    assert f_beta_sets(["S2-2"], ["S2-1"]) == 0.0


def test_macro_includes_missing_predictions():
    truth = {"a": [], "b": ["S2-1"]}
    assert macro_f_beta({}, truth) == 0.5  # empty pred: a scores 1, b scores 0
