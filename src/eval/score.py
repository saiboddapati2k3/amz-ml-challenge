"""Competition-exact macro F0.5.

Per Source-1 entity:
  * truth empty, pred empty      -> 1.0
  * truth empty, pred non-empty  -> 0.0
  * truth non-empty, pred empty  -> 0.0
  * otherwise F_beta from set precision/recall
Then the plain mean over ALL S1 entities in the evaluation set.

Usage:
  python -m src.eval.score --pred output/matching_results.tsv \
      --truth dataset/train/train_ground_truth.tsv [--ids-from truth]
"""
from __future__ import annotations

import argparse
from typing import Dict, Iterable, Mapping, Optional

BETA = 0.5


def f_beta_sets(pred: Iterable[str], truth: Iterable[str], beta: float = BETA) -> float:
    pred, truth = set(pred), set(truth)
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f_beta(
    pred: Mapping[str, Iterable[str]],
    truth: Mapping[str, Iterable[str]],
    s1_ids: Optional[Iterable[str]] = None,
    beta: float = BETA,
) -> float:
    """Mean per-entity F_beta over s1_ids (default: every S1 in truth)."""
    ids = list(s1_ids) if s1_ids is not None else list(truth.keys())
    if not ids:
        return 0.0
    return sum(f_beta_sets(pred.get(s, ()), truth.get(s, ()), beta) for s in ids) / len(ids)


def breakdown(pred, truth, s1_ids=None, beta: float = BETA) -> Dict[str, float]:
    """Macro score split into singleton / matched entities, plus micro P/R for diagnosis."""
    ids = list(s1_ids) if s1_ids is not None else list(truth.keys())
    single = [s for s in ids if not truth.get(s)]
    multi = [s for s in ids if truth.get(s)]
    tp = fp = fn = 0
    for s in ids:
        p, t = set(pred.get(s, ())), set(truth.get(s, ()))
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
    return {
        "macro_f05": macro_f_beta(pred, truth, ids, beta),
        "macro_f05_singletons": macro_f_beta(pred, truth, single, beta) if single else float("nan"),
        "macro_f05_matched": macro_f_beta(pred, truth, multi, beta) if multi else float("nan"),
        "singleton_share": len(single) / max(len(ids), 1),
        "micro_precision": tp / max(tp + fp, 1),
        "micro_recall": tp / max(tp + fn, 1),
        "n_entities": len(ids),
    }


def main():
    from src.io_utils import read_id_list_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()
    pred = read_id_list_file(args.pred)
    truth = read_id_list_file(args.truth)
    for k, v in breakdown(pred, truth).items():
        print(f"{k:>24}: {v:.5f}" if isinstance(v, float) else f"{k:>24}: {v}")


if __name__ == "__main__":
    main()
