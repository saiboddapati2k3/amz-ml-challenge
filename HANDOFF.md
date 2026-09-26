# Handoff: Business Entity Resolution (Amazon ML Challenge 2026)

State as of 2026-09-26, 21:50 IST. About 1 day left on the round timer. Details and history are in
`STATUS.md`; this file covers what is done and what to do next.

> **Mac copy (2026-09-26).** Work now continues from this Mac repo only. It was copied before the
> m10 work, so these items mentioned below are **not here**: `model_m10_*`, `oof_m10.parquet`,
> `artifacts/feat/train10`, `train10_s1.parquet`, the one-owner option in `src/predict.py`
> (`--one-owner`), `src/eval/oof_f05.py`, `src/stage2.py`, `scripts/build_train_feats.py`,
> `output/m10/`, `output/one_owner/`. The best model here is `model_dev_frozen_*` (lockbox 0.9603,
> leaderboard 0.942), and the Mac reproduces that lockbox number exactly.

## 1. Ground rules (please keep these)

- **Never train, tune or look at the test set** (`dataset/test/*`) to design anything. Test is only for
  final inference. The one exception is label-free corpus statistics (token rarity/IDF), which are
  computed the same way on train and on test.
- **The lockbox** (`artifacts/splits/lockbox_s1.parquet`, 45,000 train S1) is the honest accuracy check.
  Score on it only at milestones and **never tune anything on it**. Dev (`dev_s1.parquet`) and the
  10% sample (`train10_s1.parquet`) are for iteration.
- **Write your own code.** Public repos are fine for ideas, but top teams' code gets reviewed.
- Work from the repo folder with the venv active:
  `cd ~/Documents/amz-ml-challenge/business_entity_resolution; source .venv/bin/activate.fish`.
  Commands fail with "No module named src" when run from another folder.

## 2. Pipeline (all phases done)

| Phase | Command / module | Output |
|---|---|---|
| 1. Normalize (transliteration, legal forms, address canon) | `python -m src.prep --split train` / `test` | `artifacts/norm/` |
| 2. Blocking (11 key families, ~42 candidates per S1) | `python -m src.block --split test --chunk 25000` | `artifacts/cand/<tag>/part-*.parquet` |
| 3. Pair features (73) | `src/features.py`; for large train samples `scripts/build_train_feats.py` | `artifacts/feat/` |
| 4. LightGBM, 5 folds grouped by S1 | `python -m src.train --feat ... --s1-ids ... --tag ...` | `artifacts/model/model_<tag>_*.txt` |
| 5. Predict + submission files | `src/predict.py` (see section 5) | `output/<name>/matching_results.tsv` |
| 6. Evaluate | `src/eval/lockbox.py`, `src/eval/oof_f05.py`, `utils/validate_submission.py` | `reports/` |

## 3. Results so far

Metric: macro F0.5 per S1 record. True matches missed by blocking count as misses.

| Version | Dev OOF | **Lockbox** (honest) | Leaderboard |
|---|---|---|---|
| Stage 1, trained on 22k S1 (`model_dev_frozen_*`), tau 0.75 | 0.9605 | 0.9603 (US 0.9716, India 0.9434) | **0.942** |
| + one-owner rule (`output/one_owner/`) | not measurable on samples | not measurable | not submitted yet? |
| **m10: trained on 216k S1** (`model_m10_*`), tau 0.75 | **0.9665** | **0.9651** (US 0.9751, India 0.9501) | running now |

- **Ceiling with the current candidates (oracle): 0.985.** That's below the leaders' 0.99, so blocking has
  to improve too.
- Measured the way some public repos report it (pairwise F0.5 on candidates), m10 scores 0.9855.
  Those "98%" numbers are not the leaderboard metric.
- **France** (unseen in training, ~15% of test) is the weakest part. The leaderboard implies about 0.86
  for France, against ~0.95–0.97 for US and India.

## 4. Running right now (on the old Windows machine; not available in this Mac copy)

Test re-score with m10 + one-owner, looped one part per process to avoid a memory leak:
- log `reports/predict_test_m10.log`, scores `artifacts/score/test_m10/`, output `output/m10/`
- ~3.4 min per part x 71 parts, so it finishes around **01:30–02:00**. Then upload
  **`output/m10/matching_results.tsv`**. Expected leaderboard around 0.95.
- If it dies, just re-run the same loop. Finished parts are kept (section 5).

## 5. How to run (tested commands)

These work from fish, zsh or bash (tested on the Mac copy with `model_dev_frozen_*`).

```bash
# Test predict/re-score: one part per process (memory is not freed between parts on 8 GB).
# scripts/predict_loop.sh sets POLARS_MAX_THREADS=2 and repeats until every part is scored.
# Scores are cached in artifacts/score/<cand dir name>/; delete that folder (or pass --rescore) for a new model.
scripts/predict_loop.sh --split test --models "artifacts/model/model_dev_frozen_*.txt" --tau 0.75 --out output

# Lockbox check for a model tag (copy the cand dir so its scores go to a separate folder)
cp -r artifacts/cand/train_lockbox artifacts/cand/lockbox_<tag>
scripts/predict_loop.sh --split train --cand artifacts/cand/lockbox_<tag> \
  --s1-ids artifacts/splits/lockbox_s1.parquet --models "artifacts/model/model_<tag>_*.txt" --tau 0.75 --out output/lockbox_<tag>
python -m src.eval.lockbox --pred output/lockbox_<tag>/matching_results.tsv --cand artifacts/cand/lockbox_<tag> \
  --oof artifacts/model/oof_dev_frozen.parquet --tag "<tag>"

# Validate a submission before upload (--check-ids is slow, ~20 min)
python utils/validate_submission.py -m output/matching_results.tsv -c output/candidate_pairs.tsv -t dataset/test
```

## 6. What we tried that did not help (don't repeat)

- **Stage 2 re-scoring with list context** (`src/stage2.py`: best score of the S1, gap, rank, counts):
  dev 0.9604 vs 0.9605. No gain.
- **Linking via accepted siblings** (accept a candidate that looks like an already-accepted match of the same
  S1): only 24–31% precision on train. Non-matches resemble siblings more than true matches do.
- Transliteration and French stopwords/legal forms: **already in** `src/prep.py`.

## 7. What to do next (in priority order)

1. **Submit `output/m10/`** when the run finishes and write the score into `STATUS.md`. If one-owner was never
   submitted alone, `output/one_owner/` isolates its effect.
2. **Model ensemble** (easy, +0.1–0.3 expected). Train a second model type (XGBoost or CatBoost, or a
   LightGBM with different params or seed) on `artifacts/feat/train10`, then average with m10. Check dev OOF
   first, then the lockbox. Needs a test re-score (~4 h).
3. **Blocking recall (raises the 0.985 ceiling).** Error analysis of misses is in `artifacts/cand/dev_misses.parquet`.
   About half are very common names ("sai projects", "raj") whose other record has no address, so the name
   key exceeds its cap (300) and gets dropped. Ideas: raise `k_total` (40 → 60) and see recall@K in
   `reports/blocking_recall_*.txt`; a name-plus-city key; a compact-name key for concatenated names
   (`exportsforgezenith`). Measure with `src/eval/blocking_recall.py` and `src/eval/oracle.py` on dev.
   Every blocking change means re-blocking train10, the lockbox and test, plus retraining and a test re-score.
4. **France**: no labels, so use the leave-one-country-out check (`src/eval/loco_ablation.py`, Latin view) for any
   generic fix (address order, "cedex", postal handling).
5. More training data (e.g. 20–30% of train) gave +0.5 for 10x. Diminishing returns, and memory is the limit.

## 8. Machine notes (Mac: M1, 8 GB RAM, ~22 GB free disk)

- **Close Chrome, Safari and other heavy apps during long runs.** On the old 7.7 GB laptop, Chrome (2.5 GB)
  pushed training into swap: one fold took 3.25 h instead of 8 min. Check Activity Monitor → Memory
  (memory pressure should stay green), or run `sysctl vm.swapusage` in a terminal. When a job is far slower
  than usual, it's swapping.
- Measured on the Mac: scoring runs at about 1M pairs/min with `model_dev_frozen_*` (a full test re-score,
  72.6M pairs, is roughly 1.5 h; bigger models such as m10 are slower). Re-writing outputs from cached scores
  takes seconds.
- Disk is tight (~22 GB free). Full-train features (~27 GB) will not fit here.
- A Kaggle notebook (~30 GB RAM) would make everything 3–5x faster. Check the challenge rules first and keep
  the dataset private.
