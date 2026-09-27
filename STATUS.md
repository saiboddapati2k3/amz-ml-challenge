# Status: Business Entity Resolution (Amazon ML Challenge 2026)

Last updated 2026-09-27 (Mac). Newest findings first; the 2026-09-26 Windows notes follow.

## 2026-09-27 findings
- **Leaderboard decomposition:** 0.942 full; **0.815 with France emptied** → France ≈ 0.87-0.90 (depends
  on France's unknown singleton share), US+India on test ≈ 0.949-0.955.
- **One owner is a hard constraint:** 0 of 7,638,365 train S2/S3 records belong to two S1. In test
  predictions (dev_frozen, τ 0.75), 3.6% of France pairs are claimed by several S1, against 0.4-0.7% for
  US/India. `src/predict.py --one-owner hard|soft` enforces it, with statistics over every scored pair of
  the split. It can't be measured on dev/lockbox (1-2% density), so `src/eval/owner_eval.py` evaluates it
  on full-train scores (`artifacts/score/train_full_n1_v3h1`).
- **France normalization (n2, `artifacts/norm_n2`):** abbreviations found label-free (frequent in
  S2/S3, absent in S1): R/ALL/IMP/RTE/CH/CRS/Q/PAS → rue/allée/impasse/route/chemin/cours/quai/passage,
  N° → number, Saint → street (as St already was), `(France)` wrapper dropped like `(India)`,
  Cie/Compagnie → company, Ets → etablissements. Changes 11-40% of France records and 0-2% of US/India.
  On a fresh 30k-S1 check set (dev_frozen): **n1 0.9618 vs n2 0.9619**, so it's neutral for US/India.
- IDF cache moved into the norm directory (`<norm>/<split>_idf.parquet`), so a new normalization can't
  reuse stale statistics.
- **One-owner at full density** (dev_frozen, 764,627 India S1 never trained on; all 883k India S1
  compete): none τ0.75 **0.94359** → hard τ0.75 **0.94615** (+0.26) → **hard τ0.65 0.94673 (+0.31)**;
  soft is at best 0.94657 (τ0.5). Chosen: hard, and one-owner lets the threshold drop ~0.1.
  (`reports/owner_eval_india_dev_frozen.csv`)
- **Leaderboard: one-owner hard τ0.75 = 0.945** (vs 0.942): +0.3 on test, as predicted.
- **Test has more orphan S2/S3 records** (label-free, India): 5.82 S2/S3 per S1 vs 4.68 in train, same
  claims per S1 (3.10 vs 3.08), uncertain band (best p 0.2-0.75) 6.0% vs 3.7% of records. So there's
  more false-positive risk on test than train shows: probe τ on the leaderboard, don't just lower it.
- Test blocking with n2: 72,613,074 pairs. 7.2% of France candidate pairs change, 0.4-0.8% of US/India.
- **m10 (10% sample, 220,682 S1, 9,247,141 pairs, 5 folds, ~1,800 trees each, 50 min, peak 1.7 GB):**
  OOF AUC 0.99987 / AP 0.99858 (dev_frozen 0.99977 / 0.99757). Dev OOF at τ0.75 **0.96641** (US 0.97598,
  India 0.95178) vs dev_frozen 0.96055: +0.59. Fresh check set (30k S1) **0.9670** vs 0.9618 (+0.52,
  non-overlapping CIs). OOF-best τ without one-owner is 0.70-0.75.
- Inference cost: m10 5 folds ≈ 195 s per 1M pairs (prediction dominates), so test takes ~3.9 h. The
  submission uses folds 0-2 (~135 s/M, ~2.7 h) plus a second model m10b trained on a disjoint 10% sample
  (`train10b_s1`); their scores are averaged per pair (`scripts/avg_scores.py`).
- **Leaderboard: m10 (3 folds) + n2 + one-owner τ0.75 = 0.942** (worse than 0.9456). Diagnosis:
  France gained 45.7k predictions, 27,213 of them with the country word only on the candidate side
  ("Triangle Gipsy Jeunes **France** SARL" at another house number). n2 dropped 'france' from name_core,
  which made these sister entities look identical.
- **Country-word rule** (`scripts/country_rule.py`): a candidate whose name has the record's own country
  word while the S1 name doesn't is a different entity (train: candidate-only 'india' **0 positives of
  158,579**; US 'us'/'usa'/'america'/'american' 0 of ~18,800). Zeroing them removes 27k France
  predictions from the 0.942 run (3.26 → 3.15 per S1) and leaves US/India unchanged.
- The cross-encoder is applied only to countries in its training data (US, India); unseen countries keep
  the LightGBM score.
- **Cross-encoder reranker** (`notebooks/kaggle_ce.ipynb`, MiniLM-L6 ms-marco, Apache-2.0, fine-tuned on
  Kaggle T4 on 700k pairs: every m10 out-of-fold pair with 0.001 < p < 0.999, plus 150k confident
  positives and 150k confident negatives; holdout AUC 0.9941 on that hard mix). It rescores only pairs
  with an uncertain LightGBM score; blend = sigmoid(w·logit(p_lgb) + (1-w)·logit(p_ce)). **Check set:
  LightGBM 0.9670 → blend 0.9734 (w0.6, τ0.65); cross-fitted over S1 halves +0.57 / +0.68.** US
  0.9775→0.9825, India 0.9517→0.9600. Test has 4.1% of pairs in the band (train 2%).
- Blocking misses (dev, 3,965 pairs): native-script India 1,228; S2/S3 without address 1,130, of which
  only 110 have a unique S1 name (the rest are shared by 6-50+ S1: not recoverable without false
  positives); fully renamed or relocated records. No blocking rework before the deadline.
- Per-source (S2/S3) thresholds: no gain on dev OOF (single τ 0.75 is best).
- New splits: `train10_s1` (10% of train S1, incl. dev, excl. lockbox), `check_s1` (30k fresh
  holdout), `tune_s1` (the remaining 1.91M S1, for decision tuning at full density).

## Current best (dev, 22,133 train S1, 5-fold out-of-fold)
| Item | Value |
|---|---|
| Blocker | v3h1: 11 key families, deterministic tie-breaks, pair recall 0.9581 (US 0.979, India 0.926), 41.9 cand/S1 |
| Oracle ceiling | 0.985 (US 0.994, India 0.971) |
| Stage-1 model | LightGBM, 73 features (`model_dev_frozen_*`), OOF AUC 0.99977, AP 0.99757 |
| Decision | p >= 0.75, macro F0.5 0.9605 (US 0.9716, India 0.9436) |
| France estimate (LOCO, Latin view) | 0.931 = mean(US-LOCO 0.9527, India-Latin-LOCO 0.9098) |

**Lockbox baseline (honest, 45,000 held-out train S1, 2026-09-26): macro F0.5 0.9603
[0.9591, 0.9614]; US 0.9716, India 0.9434; test-mix 0.9561; oracle 0.9845.** Matches dev, so the
dev number was not optimistic. Expected test (US/India/France LOCO 0.931 mix) is about 0.95.
Note: the first lockbox run was invalid (0.075) because the candidates had been blocked for an
earlier 45,518-S1 draft of the lockbox file. Re-blocked; stale outputs kept under `*_stale`.

## Decisions
1. **Feature set: current, with idf sums divided by the country's ln N.** Full LOCO ablation
   (`reports/loco_ablation_full.md`): best expected test score, 0.9562. Every alternative is
   lower, with a paired-bootstrap CI that excludes 0.
2. **Rarity statistics are computed on the corpus being resolved** (label-free, same procedure on
   train and test). A documented transductive statistic; nothing is fit on test.
3. **One global threshold (0.75), no country rules.**
4. **Blocking is memory-bounded and deterministic.** Phase A tokenises S2/S3 in 1M-row slices,
   builds keys per slice and counts them per hash bucket (8), all spilled to disk. Phase B walks
   S1 in chunks. `POLARS_MAX_THREADS` defaults to 4. Output is identical to v3h1 on dev
   (927,282 pairs), and dev peak RSS dropped from 3.7 GB to 2.3 GB.
5. **Inference is chunked end to end.** Features are built and scored per blocker part (complete
   S1 lists, about 1M pairs per part at `--chunk 25000`), and scores are written per part. The
   submission TSVs are streamed in 200k-S1 batches, so test pairs are never all in memory.

## Lockbox protocol (honest accuracy)
- `artifacts/splits/lockbox_s1.{txt,parquet}`: 45,000 train S1, disjoint from dev (asserted;
  dev list in `dev_s1.txt`). Stratified by country x match-count bucket (0/1/2/3+), seed 2026.
  Its strata shares match train to 4 decimals.
- Blocked against the full train S2/S3 pool with the frozen blocker. Scored with the 5 dev-fold
  models (averaged) and the frozen rule, p >= 0.75.
- Scored only on lockbox S1 (`python -m src.eval.lockbox`). Reports bootstrap 95% CIs, US and
  India separately, a test-mix reweighted score, and the LOCO France estimate.
- **Scored only at milestones (baseline, after stage 2, before the final submission). Never used
  to choose or tune anything.** Differences under about ±0.2 points are noise.

## Scale estimates
- Test: 1.73M S1 (US 663k, India 810k, France 259k) against about 10M S2/S3, so about 72M
  candidate pairs at 42 per S1.
- Full train: 2.21M S1 x 41.9, about 92M pairs. That's about 27 GB of float32 features, too big
  to hold in memory. Stage 2 will train on sampled entities (all positives, non-trivial
  negatives, a weighted sample of easy negatives). Owner (candidate-side) features come from
  scoring all pairs in chunks.

## Run log
| Run | Result |
|---|---|
| Sliced blocker, dev | identical to v3h1, peak 2.32 GB, 743 s |
| Test blocking | done: 72.6M pairs, 1.73M S1, peak 3.2 GB, 675 s |
| Lockbox baseline | 0.9603 (see top) |
| Test submission (stage 1, tau 0.75) | `output/matching_results.tsv`, 1,732,544 S1 (108,201 empty), 5.49M matches; validator `--check-ids` PASS. France 3.22 matches/S1, empty 5.4% (train-like). Predict now runs as a loop with `--max-parts 1` and `POLARS_MAX_THREADS=2`: memory is not released between parts and the 7.7 GB laptop swapped (one part took 43 min). |
| (old) Test blocking | France done (11 parts, about 42 cand/S1, 106 s). India was killed during the key spill because the **machine** ran out of RAM: other apps hold about 6 of 7.7 GB. |

## Next steps (in order)
1. Free RAM (close browsers, WhatsApp, extra editors; about 3+ GB free is needed), or move to a
   bigger machine (Kaggle notebook, about 30 GB; check the rules first and keep the dataset private).
2. Blocking on test: `python scripts/peakmem.py src.block --split test --chunk 25000`.
3. Safety submission: `python scripts/peakmem.py src.predict --split test --models "artifacts/model/model_dev_frozen_*.txt" --tau 0.75`,
   then `python utils/validate_submission.py -m output/matching_results.tsv -c output/candidate_pairs.tsv -t dataset/test --check-ids`.
   Report per-country prediction rate, empty share and matches per S1 (train: empty about 0.056, about 3.4/S1).
4. Lockbox baseline: block `--s1-ids artifacts/splits/lockbox_s1.parquet --tag lockbox`, run
   predict with `--split train --cand artifacts/cand/lockbox --s1-ids ... --out output/lockbox`,
   then `python -m src.eval.lockbox --pred output/lockbox/matching_results.tsv --cand artifacts/cand/lockbox --oof artifacts/model/oof_dev_frozen.parquet`.
5. Full-train blocking (`--tag train_full --chunk 25000`), then stage 2 with sampled training.
