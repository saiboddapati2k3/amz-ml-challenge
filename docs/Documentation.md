# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** leakyReLU
**Team Members:** Mayank Kumar (IIIT Tiruchirappalli), Priya Singh (IIIT Tiruchirappalli), Sai Boddapati (VIT Chennai), Harshita Singh (IIIT Tiruchirappalli)
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

A classic blocking + pairwise-classifier pipeline, built to be precise on the metric (macro F0.5 per
Source-1 entity). The main ingredients are:

- A multi-key weighted inverted index generates about 42 candidates per S1 record.
- A LightGBM classifier on 73 label-free string, number and blocking features scores each pair.
- Two dataset facts drive the decision layer. S1 is deduplicated, so **every S2/S3 record belongs to
  at most one S1** (true for all 7.6M training pairs). The one-owner rule enforces this at test time.
- Normalization is data-driven. Abbreviations are discovered label-free: tokens frequent in S2/S3
  but absent from S1.

The unseen country (France) gets no special rules and no labels. Its rarity statistics come from its
own records, and its extra abbreviations were found by the same label-free comparison.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on train (US, India) and from label-free statistics on test:

- **Name noise:** legal-form swaps and drops (Pvt/Private, Ltd/Limited, Inc, LLC, SARL/SAS),
  generator-appended descriptors (US: services, holdings, downtown, riverside...; France: groupe,
  holding, participations, distribution...), `(India)`/`(France)` wrappers, web-domain names
  (`nantesinstitut.com`), typos, native-script transliterations (India), and random accents.
- **Address noise:** component reordering, missing components (about 3% of S2/S3 addresses are
  empty), street-type abbreviations (Rd/Road, St/Street; France R/Rue, BD, ALL, IMP, RTE, N°),
  region vs département (Hauts-de-France vs Nord), and landmark references.
- **Common names:** 40-55% of S1 names are shared with another S1 of the same country, so a name alone
  rarely identifies an entity. The address (house number + street) must agree.
- **One owner:** no S2/S3 record is matched to two S1 records in the ground truth (0 of 7,638,365).
- **Unseen country:** France appears only in test. Its business names are often "<city> <generic
  word>" ("Bordeaux Amicale", "Lille Club"), and its S2/S3 addresses abbreviate heavily (25% use "R"
  for "Rue").

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier + constrained decision layer
**Core Innovation:** Full-density evaluation of a one-owner decision rule, and label-free abbreviation
discovery that transfers to the unseen country without hand-tuned country logic.

Pipeline: `src/prep.py` (normalization) → `src/block.py` (candidates) → `src/features.py` (pair
features) → `src/train.py` (LightGBM, S1-grouped folds) → `src/predict.py` (scoring, one-owner rule,
threshold, submission files).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** 11 key families per country. They cover name tokens and compact names, name
  token pairs, 4-character prefixes, house number × rare address token, rare address token pairs,
  name token × address token, phonetic skeletons (single and pairs), address number pairs, and phonetic
  name × house number for transliterated names. Each shared key adds idf = ln(N/df) to its family
  score. Keys whose posting list exceeds a cap are dropped.
- **Selection:** per S1, the union of the top 40 by combined score and the top 10 by name-only and
  address-only score, so a single noisy field cannot hide a match.
- **Candidate pairs generated:** [TEST_PAIRS] on test (about 42 per S1); 92.5M on the full train set.
- **How we ensured true matches were not lost:** recall was measured on a 1% dev sample after every
  change. Pair recall is 0.958 (US 0.979, India 0.926), and the oracle ceiling (a perfect matcher on
  these candidates) is 0.985 macro F0.5.

---

## 4. Matching Model

**Features used (73, all label-free):**
- Name: fuzzy ratios (ratio, token set/sort, partial, Jaro-Winkler) on core, compact and phonetic
  views; alias-aware best score; idf-weighted token overlap; rarest unmatched token; legal-form
  agreement/conflict; numbers in the name.
- Address: fuzzy ratios, idf-weighted overlap, house-number and postal-like number agreement/conflict,
  token counts.
- Blocking: family scores and ranks, number of families that retrieved the pair, gap to the S1's best
  candidate. Idf-scale features are divided by the country's ln N so they transfer to a new country.

**Model type:** LightGBM (binary, 127 leaves, learning rate 0.05, early stopping on an S1-grouped inner
holdout), 5 folds grouped by S1, fold models averaged at inference. Final model trained on a 10% sample
of train S1 ([TRAIN_ROWS] pairs).

**Decision:** one-owner rule ([OWNER_MODE]) + threshold τ = [TAU], chosen on full-density out-of-sample
train scores (1.9M S1 never used in training).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** lockbox (45,000 held-out train S1): [LOCKBOX]; public leaderboard: [LB].
- **Common false positives:** candidates owned by another S1 with the same generic name, often at the
  same house number (fixed mainly by the one-owner rule). Also common names whose other record has no
  address.
- **Common false negatives:** true matches the blocker never retrieved (about 1.5 points of the gap to
  1.0), heavily abbreviated or transliterated names, and addresses reduced to a city.

---

## 6. Conclusion

[FILL]

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: `src/` holds all source code; the `README.md` has the exact commands.

### B. Additional Results

[FILL]
