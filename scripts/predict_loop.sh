#!/usr/bin/env bash
# Run src.predict one blocker part per process until every part is scored (exit code 3 = parts left).
# Memory is not returned to the OS between parts, so a single long process starts swapping on 8 GB.
# Works from any shell (fish, zsh, bash). Arguments go straight to src.predict, e.g.
#   scripts/predict_loop.sh --split test --models "artifacts/model/model_dev_frozen_*.txt" --tau 0.75
# Parts per process: PARTS=4 scripts/predict_loop.sh ...  (default 1; use more on a big machine)
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
[ -x .venv/bin/python ] && [ -z "${PYTHON:-}" ] && PY=.venv/bin/python
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-2}"
while :; do
  "$PY" scripts/peakmem.py src.predict --max-parts "${PARTS:-1}" "$@"
  rc=$?
  [ $rc -eq 3 ] || exit $rc
done
