# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** leakyReLU
**Team Members:** Mayank Kumar (IIIT Tiruchirappalli), Priya Singh (IIIT Tiruchirappalli), Sai Boddapati (VIT Chennai), Harshita Singh (IIIT Tiruchirappalli)
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We use a blocking + pairwise classifier + constrained decision pipeline, tuned for the competition
metric (macro F0.5 per Source-1 entity):

- A multi-key weighted inverted index generates about 42 candidates per Source-1 (S1) record.
- A LightGBM ensemble scores every candidate pair on 73 label-free string, number and blocking
  features.
- A fine-tuned **cross-encoder** (MiniLM, 22M parameters) rereads the 4% of pairs LightGBM is unsure
  about, and the two scores are blended.
- A decision layer enforces a fact we verified on all 7.6M training pairs: **every S2/S3 record belongs
  to at most one S1** (S1 is deduplicated). Only the best-scoring S1 keeps each record.

The unseen country (France) gets no labels and no hand-written country logic. Its rarity statistics
come from its own records, and its abbreviations were found by a label-free comparison of S2/S3 against
S1.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on train (US, India) and label-free statistics on test:

- **Name noise:** legal-form swaps and drops (Pvt/Private, Ltd/Limited, Inc, LLC, SARL/SAS), appended
  generic descriptors (US: services, holdings, downtown, riverside...; France: groupe, holding,
  participations, distribution...), `(India)`/`(France)` wrappers, web-domain names
  (`nantesinstitut.com`), bracketed legal forms (`[EURL]`), typos, random accents, and native-script
  transliterations (India: "praivet limited").
- **Address noise:** component reordering, missing components (about 3% of S2/S3 addresses are empty),
  street-type abbreviations (Rd/Road, St/Street; France R/Rue, BD, ALL, IMP, RTE, CH, N°), region vs
  sub-region (Hauts-de-France vs Nord; Illinois vs IL), house-number noise (5% of true pairs share no
  address number), and landmark references.
- **Common names:** 40-55% of S1 names are shared with at least one other S1 of the same country (mean
  group size about 20), so the address (house number + street) usually decides.
- **One owner:** in the ground truth, 0 of 7,638,365 S2/S3 records are matched to two S1 records.
- **Test vs train:** test has 5.8 S2/S3 records per S1 against 4.7 in train, with the same number of
  claims per S1. So test contains more orphan records, which act as distractors, and 1.6x more
  candidates fall in the uncertain score band.
- **France (test only):** names are often "<city> <generic word>" ("Bordeaux Amicale"), and the S2/S3
  addresses abbreviate heavily (25% use "R" for "Rue"). Many near-duplicate S1 businesses sit on the
  same street.

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted classifier + transformer cross-encoder on uncertain
pairs + constrained (one-owner) decision layer.
**Core Innovations:**
1. **One-owner resolution, evaluated at full density.** The rule only works when every competing S1 is
   present, so it can't be measured on a small validation sample. We scored the entire training split
   and tuned it on 764k held-out S1 records.
2. **Label-free abbreviation discovery for an unseen country:** tokens frequent in S2/S3 but absent in
   S1 of the same country are abbreviations or generator noise. This found R→rue, ALL→allée,
   IMP→impasse, N°→number and the `(France)` wrapper without any French labels.
3. **Cascade reranking:** a small cross-encoder, fine-tuned on the pairs where LightGBM is uncertain,
   rescores only that band (2% of train pairs, 4% of test pairs). Their logits are averaged. On a
   fresh 30k-S1 check set: 0.9670 → 0.9733.
4. **Transductive, label-free rarity:** token idf and blocking-key weights are computed per country on
   the corpus being resolved, and idf-scale features are divided by the country's ln N. They transfer
   to a new country without retuning.

Pipeline: `src/prep.py` (normalization) → `src/block.py` (candidates) → `src/features.py` (pair
features) → `src/train.py` (LightGBM, S1-grouped folds) → `src/predict.py` (scoring, one-owner,
threshold, output files).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used (11 families, per country):**
  - name token / compact name
  - pair of name tokens
  - 4-character name-token prefix (typos, truncation)
  - house number × rare address token
  - pair of rare address tokens
  - name token × rare address token
  - phonetic skeleton of a name token, single and in pairs (transliteration)
  - pair of address numbers
  - house number × any address token
  - phonetic name × house number (native-script names)

  Each shared key adds idf = ln(N/df) to its family score. Keys whose S2+S3 posting list exceeds a cap
  identify nothing and are dropped.
- **Selection:** per S1, the union of the top 40 by combined score and the top 10 by name-only and by
  address-only score, so a single noisy field can't hide a match.
- **Candidate pairs generated:** 72,613,074 on test (41.9 per S1); 92,458,548 on the full train split.
- **Ensuring true matches aren't lost:** recall and an oracle ceiling (a perfect matcher on our
  candidates) are measured on a 1% dev sample after every change. Pair recall is 0.958 (US 0.979,
  India 0.926), and the oracle ceiling is 0.985 macro F0.5 (US 0.994, India 0.971).
- **What the blocker still misses** (dev, 3,965 pairs):
  - 31%: native-script India records
  - 29%: S2/S3 records without an address, and 70% of those have a name shared by 6+ S1, so they can't
    be resolved without false merges
  - 25%: fully renamed or relocated records

  Recovering these would mostly add false positives.
- **Memory:** keys are hashed to u64, families are processed one at a time and spilled to disk, and S1
  is walked in chunks. Blocking all 2.2M train S1 takes 8.5 min at a 1.7 GB peak on an 8 GB laptop.

---

## 4. Matching Model

**Features used (73, all label-free):**
- **Name:** fuzzy ratios (Levenshtein ratio, token set/sort, partial, Jaro-Winkler) on core,
  compact and phonetic views; alias-aware best score (a/k/a, dba); idf-weighted token overlap (min, max,
  Jaccard); rarest unmatched token on each side; legal-form agreement/conflict; numbers inside the name;
  first-token equality; lengths.
- **Address:** fuzzy ratios; idf-weighted overlap; house-number and long-number (postal-like)
  agreement, conflict and overlap; token counts.
- **Blocking:** family scores (name, address, mix, total), number of families that retrieved the pair,
  ranks within the S1's list, gap/ratio to the S1's best candidate.
- **Other:** source (S2/S3), script of each side (Latin / accented / native).

Idf-scale features are divided by the country's ln N (leave-one-country-out tests: better transfer in
both directions).

**Model type:** LightGBM binary classifier (127 leaves, learning rate 0.05, feature fraction 0.8,
bagging 0.8, L2 1.0, early stopping on an S1-grouped inner holdout, about 1,800 trees). Two models are
each trained with 5 S1-grouped folds on a disjoint random 10% of train S1 (220,682 S1, 9.2M pairs each).
The submission averages folds 0-2 of both models (6 models).

**Cross-encoder (stage 2):** `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0, 22M parameters),
fine-tuned for 2 epochs on a Kaggle T4 (AdamW, lr 3e-5, batch 256, fp16, max length 128). Training
data: 700k pairs, namely every out-of-fold pair of the LightGBM training sample with 0.001 < p < 0.999,
plus 150k confident positives and 150k confident negatives. Input: "<name> ; <address>" of both
records, transliterated to Latin (anyascii), so native-script names work with an English wordpiece
vocabulary. Holdout AUC on this hard mix: 0.9941. At inference only pairs with 0.0003 < p_lgb < 0.9997
are rescored, and p = σ(w·logit(p_lgb) + (1−w)·logit(p_ce)) with w = 0.55, chosen on half of the check
set and confirmed on the other half.

**Threshold selection method:**
- τ comes from out-of-fold macro F0.5 on the training sample, and is adjusted for the one-owner rule on
  full-density held-out train scores.
- Test has more orphan distractors than train, so τ was then checked on the leaderboard.
- Final: one-owner hard, τ = [TAU].

---

## 5. Results & Error Analysis

| Step | Validation (macro F0.5) | Public leaderboard |
|---|---|---|
| Baseline (1% sample model, τ 0.75) | dev OOF 0.9605, lockbox 0.9603 | 0.942 |
| + one-owner (hard) | India full-density +0.26 (τ0.75), +0.31 (τ0.65) | 0.945644 |
| + 10% training sample (m10), France normalization | dev OOF 0.9664, fresh check set 0.9670 (+0.52) | [U3] |
| + cross-encoder blend on uncertain pairs | check set 0.9733 (+0.64; cross-fitted +0.57/+0.68) | [U4] |
| + second model on a disjoint 10% (ensemble) | check set [ENS] | [U5] |

- France: an upload with the France rows emptied (0.815) gives France ≈ 0.87-0.90, against about 0.95 for US+India.
- **Common false positives (wrong merges):**
  - A different business with the same or similar generic name at the same house number on another
    street ("Bordeaux Amicale SARL, 102 Rue Joseph Fauré" vs "Bordeaux Amicale, 102 Rue Falquet").
    One-owner removes these when the true owner is also a candidate.
  - Records without an address whose name is shared by many S1.
  - Same street and nearby house number, with a name differing in one generic word ("Solidarites Club"
    vs "Solidarites Union").
- **Common false negatives (missed matches):**
  - Pairs the blocker never retrieved (about 1.5 points of the gap to 1.0).
  - Heavily transliterated native-script names.
  - Extra matches of multi-match S1 whose second record is heavily abbreviated or has only a city as
    address.

---

## 6. Conclusion

Most of the gain came from the data's structure rather than from model complexity:
- One-owner resolution, measured at full density.
- 10× more training data with memory-lean training on an 8 GB laptop.
- Label-free normalization that transfers to an unseen country.

The main lessons:
1. Validate decision rules at the density they'll run at (samples hide competition between records).
2. Check the test distribution label-free (orphan records, abbreviations) before trusting validation
   thresholds.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/`: `prep.py`, `block.py`, `features.py`, `train.py`, `predict.py`, `decide.py`, `io_utils.py`
- `src/eval/`: scoring, lockbox, oracle, recall, one-owner and OOF evaluators
- `scripts/`: sampling, loop runner, score averaging, packaging
- `README.md`: exact commands ("Reproduce the final submission")
- `requirements.txt`: pinned versions

Entry points: `src.prep` → `src.block` → `src.features` → `src.train` → `scripts/predict_loop.sh`
(`src.predict`), which writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Additional Results

- Leave-one-country-out (train only): a US-only model scores India's Latin-script records at 0.904, and
  an India-only model scores US at 0.951. Most transfer loss comes from ranking, not calibration.
- One-owner full-density table: `reports/owner_eval_india_dev_frozen.csv`.
