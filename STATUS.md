# Status: Business Entity Resolution (Amazon ML Challenge 2026)

Last updated 2026-09-26.

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
