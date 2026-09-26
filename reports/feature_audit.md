# Feature and rarity audit (stage 1, `src/features.py` + `src/block.py`)

## Where every corpus statistic comes from

| Statistic | Used by | Computed on | Labels? | Train vs test |
|---|---|---|---|---|
| Blocking key weight `w = ln(N23 / df)` and df caps | `s_name`, `s_addr`, `s_mix`, `s_all`, `n_fam`, ranks | S2+S3 of the **split being resolved**, per country | no | same procedure; test uses test S2+S3 |
| Token frequency for "rarest k tokens" selection | blocking keys | S1+S2+S3 of the split, per country | no | same |
| Token idf `ln(N / df)` | `name_w*`, `addr_w*`, `*_unm_*` | S1+S2+S3 of the split, per country, cached `artifacts/stats/<split>_idf.parquet` | no | same; test uses test corpus |
| Phonetic stop skeletons, legal words, address abbreviations | normalization, `lg_*`, native-script core | fixed generic word lists (not mined from test) | no | identical |

All rarity is **transductive but label-free** (user-approved): computed per country on the corpus being resolved, so an unseen country (France) gets rarity from its own records instead of looking maximally rare under train statistics. Nothing is fit or tuned on test. The idf cache is keyed by split name only; delete `artifacts/stats/` if normalization changes.

## Feature inventory

| Group | Features | Exists | OOF-safe | Needs change | Notes |
|---|---|---|---|---|---|
| Name fuzzy | `nm_ratio`, `nm_tset`, `nm_tsort`, `nm_partial`, `nm_jw`, `nm_comp_*`, `nm_full_tset`, `nm_alias_best` | yes | yes (label-free, per pair) | no | |
| Name phonetic | `nm_ph_ratio`, `nm_ph_tset` | yes | yes | no | cross-script |
| Name tokens / idf | `nt_*`, `name_w*`, `name_unm_*` | yes | yes | no | |
| Legal form | `lg_both`, `lg_same`, `lg_conflict` | yes | yes | no | skeleton-based, script-aware |
| Address fuzzy / idf | `ad_*`, `addr_w*`, `addr_unm_*` | yes | yes | no | |
| Numbers | `num_*`, `lnum_*` (postal-like), `nnum_*` (in name), `num_first_eq` | yes | yes | no | |
| Blocker raw scores | `s_name`, `s_addr`, `s_mix`, `s_all`, `n_fam` | yes | yes | **LOCO check** | idf sums; scale depends on country address structure |
| Blocker ranks | `r_all`, `r_name`, `r_addr` | yes | yes | no | |
| S1 context (blocker / name) | `s1_n_cand`, `s1_gap`, `s1_ratio`, `s1_nm_gap`, `s1_nm_rank` | yes | yes | no | already covers "gap to best candidate" |
| Candidate-side context | `cand_n_s1`, `cand_rank`, `cand_ratio`, `cand_nm_gap` | yes | yes | **yes** | only meaningful on full-split candidates; on the 1% dev file they are near-constant (train/test mismatch) -> recompute after full-split blocking |
| Stage-1 score context | rank / best / second / margins / counts above 0.5/0.7/0.9 of the **model score** | no | needs OOF stage-1 scores | new | stage 2 |
| S2<->S3 corroboration | cross-source name/addr similarity, shared house number, corroboration count | no | label-free | new | stage 2, feature only, no transitive matching |
| Best-owner (model score) | best-owner score, owner rank, margin to best owner | no | needs out-of-sample scores for all S1 | new | after full-split blocking |

Not added on purpose: "shared country" (blocking is already within country), synthetic outside-blocker negatives.
