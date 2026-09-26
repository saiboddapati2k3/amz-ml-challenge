# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline (target): normalize → multi-pass blocking → pair features → LightGBM (+ cross-encoder)
→ calibration → one-owner resolution → **expected-F0.5 set selection** → outputs.

## Status
| Stage | Module | Status |
|---|---|---|
| IO + submission writer | `src/io_utils.py` | done, passes official validator |
| Exact macro F0.5 | `src/eval/score.py` | done, README example = 0.714 |
| Entity-level folds + leave-one-country-out | `src/eval/splits.py` | done |
| Multi-view normalization | `src/normalize.py` | done (seed dictionaries) |
| Decision engine (expected F0.5, one-owner, threshold baseline) | `src/decide.py` | done, verified vs brute force |
| Day-1 forensic report | `src/analysis/forensics.py` | done |
| Blocking v3 (11 key families, per country) | `src/block.py` | dev: pair recall 0.958 (US 0.979, India 0.926), 42 cand/S1; spill + S1 chunks, deterministic (re-run identical), peak 3.7 GB on dev |
| Oracle ceiling | `src/eval/oracle.py` | dev: 0.985 (US 0.994, India 0.971) |
| Pair features (~75, script-aware) | `src/features.py` | done |
| LightGBM OOF (S1-grouped folds) | `src/train.py` | dev: AUC 0.9998, macro F0.5 0.960 |
| Decision rules on OOF | `src/eval/decide_oof.py` | threshold 0.75 = 0.960 >= expected-F 0.959 |

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate.fish      # fish; zsh/bash: source .venv/bin/activate
brew install libomp                 # macOS: LightGBM needs OpenMP
pip install -r requirements.txt
# later, on the GPU machine: install torch for your CUDA, then
pip install -r requirements-gpu.txt
```
Copy the challenge data to `dataset/train/*.tsv` and `dataset/test/*.tsv` (official file names).

## Run
```bash
pytest                                                  # 16 tests, < 1 s
python -m src.analysis.forensics --data dataset         # -> reports/forensics.md
python -m src.eval.score --pred <pred.tsv> --truth dataset/train/train_ground_truth.tsv
python utils/validate_submission.py -m output/matching_results.tsv -c output/candidate_pairs.tsv -t dataset/test
```
Smoke test without real data: `python scripts/make_synthetic.py --out dataset_synth` then point `--data` at it.

Full runs (prep → blocking → training → lockbox → test submission) go on Kaggle: see [KAGGLE.md](KAGGLE.md)
and `notebooks/kaggle_pipeline.ipynb`.

## Rules we enforce
No external data/APIs/geocoding. Splits by S1 entity only. Models MIT/Apache-2.0, ≤ 8B params.
`candidate_pairs.tsv` = exactly the pairs the final model scores.
