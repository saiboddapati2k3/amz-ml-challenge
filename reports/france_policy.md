# Country shift (LOCO) and the policy for the unseen country

Dev sample (22,133 train S1), stage-1 LightGBM, threshold rule p >= 0.75, macro F0.5.

## 1. Ranking or calibration?

| Held out | Pooled 5-fold | LOCO, transferred tau 0.75 | LOCO, tau re-tuned on held-out (upper bound) | AP pooled -> LOCO |
|---|---|---|---|---|
| US | 0.9712 | 0.9455 | 0.9500 (tau 0.9) | 0.9978 -> 0.9912 |
| India | 0.9436 | 0.8400 | 0.8400 (tau 0.75) | 0.9968 -> 0.9552 |

Re-tuning the threshold on the held-out country recovers nothing for India and 0.45 of 2.6 points for US:
**the failure is ranking (AP), not calibration.** Under LOCO, India also gets too few predictions
(empty share 0.099 vs 0.055 in truth, 2.75 vs 3.46 matches/S1).

## 2. Where the India ranking failure comes from

| Candidate script | Pooled recall @0.75 | LOCO recall @0.75 | LOCO AP |
|---|---|---|---|
| Latin (0) | 0.952 | 0.925 | 0.982 |
| Latin w/ accents (1) | 0.961 | 0.946 | 0.993 |
| Native script (2) | 0.949 | **0.194** | 0.904 |

Most of the drop comes from native-script candidates, a record type that exists only in India, so a US-only model has never
seen it. That does not transfer to a Latin-script country. The France-relevant part is the Latin view
(native-script candidates removed from candidates and truth): in-country 0.943 -> LOCO 0.904.

## 3. Ablation of scale-dependent features (LOCO; India on the Latin view)

| Variant | US @.75 | US @.9 | India-Latin @.75 | India-Latin @.9 |
|---|---|---|---|---|
| all features | 0.9455 | 0.9500 | 0.9038 | 0.9044 |
| **raw idf sums / ln N (per country)** | **0.9535** | **0.9567** | **0.9069** | 0.9036 |
| raw -> within-country percentile | 0.9466 | 0.9493 | 0.9022 | 0.8983 |
| drop idf features | 0.9523 | 0.9500 | 0.8990 | 0.8946 |
| drop address/length structure | 0.9400 | 0.9432 | 0.8945 | 0.8973 |
| drop postal-like numbers | 0.9475 | 0.9508 | 0.9028 | 0.9021 |
| drop script features | 0.9467 | 0.9506 | 0.9035 | 0.9029 |

Only ln N scaling improves both directions; it is adopted in `src/features.py` (`SCALED`). After
the change: pooled 0.9605 (was 0.9603, no in-distribution cost); LOCO US 0.9511, India 0.8407.
Single fits: differences below ~0.3 points are within noise.

## 4. Policy for the unseen country

1. Rarity statistics are computed on the corpus being resolved (label-free, same procedure as train)
   and idf-scale features are divided by the country's ln N.
2. The final model is trained on both countries, which is more diverse than the single-country LOCO proxy.
3. One global threshold, no country-specific rules. A conservative threshold (0.9) helps US-LOCO
   (+0.3) and hurts India-Latin (-0.3), so there's no evidence for it. Kept as a decision option to re-test after stage 2.
4. Safety run: report prediction rate, empty share and matches/S1 for France next to US and India;
   a large deviation from the train distribution (empty ~0.056, ~3.4 matches/S1) is the warning signal.
5. Residual risk: about 2-4 points of Latin-script ranking shift remain under LOCO. The candidate
   lever is a character-level cross-encoder score (stage 2 / final), which can only be validated by LOCO.
