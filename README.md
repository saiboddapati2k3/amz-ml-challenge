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

## Reproduce the final submission

Run from this folder with the venv active. Times are for an 8 GB M1 laptop.

```bash
# 1. Normalize. Train uses the rules the final model's training features were built with (n1);
#    test uses the current rules (n2 = n1 + abbreviations found label-free, see STATUS.md).
python -m src.prep --split train --rules n1                                  # artifacts/norm/train_s*
python -m src.prep --split test  --rules n2 --out artifacts/norm_n2          # artifacts/norm_n2/test_s*

# 2. Blocking (about 42 candidates per S1)
python -m src.block --split train --tag train_full_n1_v3h1 --chunk 25000     # 92.5M pairs, ~9 min
python -m src.block --split test --norm artifacts/norm_n2 --tag test_n2_v3h1 --chunk 25000   # ~10 min

# 3. Model: 10% sample of train S1 (dev included, lockbox excluded), features, 5-fold LightGBM
python -m scripts.make_train_sample --frac 0.10                              # artifacts/splits/train10_s1.parquet
python -m src.features --split train --cand artifacts/cand/train_full_n1_v3h1 \
    --s1-ids artifacts/splits/train10_s1.parquet --tag train10_n1_v3h1       # artifacts/feat/train10_n1_v3h1/
python -m src.train --feat artifacts/feat/train10_n1_v3h1 --s1-ids artifacts/splits/train10_s1.parquet --tag m10

# 4. Test: LightGBM scores for every candidate (folds 0-2 of m10; cached per part)
ln -s test_n2_v3h1 artifacts/cand/test_n2_v3h1__m10f3
scripts/predict_loop.sh --split test --norm artifacts/norm_n2 --cand artifacts/cand/test_n2_v3h1__m10f3 \
    --models "artifacts/model/model_m10_[012].txt" --tau 0.75 --no-outputs
```

5. Cross-encoder on the uncertain band (GPU; we used a Kaggle T4 with `notebooks/kaggle_ce.ipynb`):
```bash
python scripts/ce_export.py train artifacts/model/oof_m10.parquet ce_data/ce_train.parquet       # 700k training pairs
python scripts/ce_export.py pairs artifacts/score/train_check_n1_v3h1__m10 ce_data/ce_check.parquet   # validation
python scripts/ce_export.py pairs artifacts/score/test_n2_v3h1__m10f3 ce_data/ce_test.parquet \
    --split test --norm artifacts/norm_n2 --lo 0.0003 --hi 0.9997                               # 7.0M test pairs
# notebooks/kaggle_ce.ipynb: fine-tune on ce_train, score ce_check + ce_test -> ce_scores_{check,test}.parquet
python scripts/ce_blend_eval.py ce_data/ce_scores_check.parquet artifacts/score/train_check_n1_v3h1__m10 \
    artifacts/splits/check_s1.parquet                                                            # picks w=0.5, tau=0.7
```

6. Decision layer and submission files:
```bash
# split the cross-encoder scores: countries in its training data (US, India) vs unseen countries (France)
python scripts/ce_split.py ce_data/ce_scores_test.parquet ce_data/ce_scores_test_seen.parquet ce_data/ce_scores_test_unseen.parquet
python scripts/ce_apply.py artifacts/score/test_n2_v3h1__m10f3 ce_data/ce_scores_test_seen.parquet \
    artifacts/score/test_n2_v3h1__m10f3ce --w 0.5                      # blend (US, India)
python scripts/ce_apply.py artifacts/score/test_n2_v3h1__m10f3ce ce_data/ce_scores_test_unseen.parquet \
    artifacts/score/test_n2_v3h1__m10f3cev --w 0.5 --veto              # veto only (unseen countries)
python scripts/country_rule.py artifacts/score/test_n2_v3h1__m10f3cev artifacts/score/test_n2_v3h1__final \
    --norm artifacts/norm_n2 --split test                              # sister entities
ln -s test_n2_v3h1 artifacts/cand/test_n2_v3h1__final                  # every part cached: nothing re-scored
python -m src.predict --split test --norm artifacts/norm_n2 --cand artifacts/cand/test_n2_v3h1__final \
    --models "artifacts/model/model_m10_[012].txt" --tau 0.7 --one-owner hard --out output
python utils/validate_submission.py -m output/matching_results.tsv -c output/candidate_pairs.tsv -t dataset/test
```

Honest accuracy checks (all on train S1 the model never saw):
```bash
# one-owner mode x threshold at full density (needs scores for the whole train split)
scripts/predict_loop.sh --split train --cand artifacts/cand/train_full_n1_v3h1 --models "..." --tau 0.75 --no-outputs
python -m src.eval.owner_eval --scores artifacts/score/train_full_n1_v3h1 --ids artifacts/splits/tune_s1.parquet
# lockbox milestone (45,000 S1, never used for decisions)
python -m src.eval.lockbox --pred output/lockbox/matching_results.tsv --cand artifacts/cand/train_lockbox
```

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
