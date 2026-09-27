# Running the pipeline on Kaggle

> **Note (2026-09-27):** the final submission was produced locally with the commands in
> `README.md` ("Reproduce the final submission"): normalization rules n1 for train and n2 for test,
> the m10 model and `--one-owner hard`. This notebook reproduces the earlier baseline (dev model,
> no one-owner). It still works for heavy runs, but pass the new options yourself.

The Mac (8 GB) is for editing code and quick checks. Heavy runs go to a Kaggle notebook (about 30 GB of RAM,
4 CPUs, free). The notebook is [`notebooks/kaggle_pipeline.ipynb`](notebooks/kaggle_pipeline.ipynb).

How it fits together:

| What | Where it lives |
|---|---|
| Code | GitHub repo; the notebook clones it at the start of every run |
| Challenge data | A **private** Kaggle dataset (uploaded once) |
| Everything the pipeline builds (`artifacts/`, `output/`) | `state.tar` in the notebook's output; attach it to the next run to resume |
| Submission files | `submission_<tags>/` in the notebook's output |

---

## Part A. One-time setup

### A1. Kaggle account
1. Sign in at kaggle.com.
2. Verify your phone number (Settings → Phone verification). Notebooks can't use the internet without it,
   and the notebook needs internet to clone GitHub and install packages.

### A2. Upload the data as a private dataset
1. kaggle.com → **Datasets** → **+ New Dataset**.
2. Drag in `6ab10eb3b23ba_student_resource.zip` (1.1 GB, in `~/Documents/amz-ml-challenge/`). Kaggle unzips it.
3. Title: for example `amz-er-data`. Visibility: **Private**. Click **Create** and wait until processing
   finishes (a few minutes).

The notebook finds the data by itself: it looks for `train/train_source1.tsv` anywhere under `/kaggle/input`.

### A3. GitHub access (only if the repo is private)
1. GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate.
   Repository access: only `amz-ml-challenge`. Permissions: **Contents: Read-only**. Expiry: after the challenge.
2. Copy the token. You'll add it to the notebook in step A4.

The notebook passes the token only in a request header. It is never saved in the notebook output.

### A4. Create the notebook
1. kaggle.com → **Code** → **+ New Notebook**.
2. **File → Import Notebook** → upload `notebooks/kaggle_pipeline.ipynb` from this repo.
3. Right-hand panel → **Session options**:
   - Accelerator: **None** (CPU; the pipeline doesn't use a GPU)
   - Internet: **On**
4. Right-hand panel → **Add Input** → **Your Work → Datasets** → `amz-er-data`.
5. Private repo only: **Add-ons → Secrets** → **Add secret**: label `GITHUB_TOKEN`, value = the token.
   Tick the box so this notebook can use it.
6. Rename the notebook (top left), for example `amz-er-pipeline`.

---

## Part B. First full run

1. Check the settings cell (section 1). The defaults reproduce the current best local model on Kaggle:
   `NORM_TAG="n1"`, `BLOCK_TAG="v3h1"`, `MODEL_TAG="dev_frozen"`, `TRAIN_MODEL=False`, all `RUN_*=True`.
2. Click **Save Version** (top right) → **Save & Run All (Commit)** → **Save**.
   This runs in the background for up to 12 hours. You can close the browser.
3. Watch progress: open the notebook page → **Versions** (or the running version) → **Logs**.

Expected time for the first run: roughly **3–4.5 hours**.

| Step | Time |
|---|---|
| Setup (clone, packages) | ~2 min |
| Prep (normalize train + test) | ~15–25 min |
| Dev blocking + recall + oracle | ~10–15 min |
| Lockbox blocking + scoring + report | ~25 min |
| Test blocking | ~15 min |
| Test scoring (72.6M pairs) | ~1.5–3 h |
| Saving `state.tar` (after each section) | ~1–2 min each |

Later runs reuse every finished step, so they're much faster (see Part D).

---

## Part C. Checking accuracy

Open the finished version → **Logs**, or the notebook with outputs. The test set has no labels, so accuracy
is measured on train records the model never saw.

| Where | What it tells you | Current baseline |
|---|---|---|
| `oracle_dev_*` in section 4 | **Ceiling**: the best score any matcher could get with these candidates | 0.985 |
| `recall_dev_*` in section 4 | Share of true pairs the blocker finds | 0.958 |
| `decide_oof_*` in section 4 | Dev out-of-fold macro F0.5 per decision rule (dev is also used for tuning) | 0.960 at p ≥ 0.7 |
| **Lockbox table** in section 5 | **The honest number.** Row `lockbox ALL` = macro F0.5 on 45,000 held-out S1, with a 95% CI, US, India and `test-mix` | **0.9603** (US 0.9716, India 0.9434) |
| Leaderboard | The real test score, including France | 0.942 |

How to read the lockbox table:
- `macro_f05`: the competition metric. `ci_lo`/`ci_hi`: differences smaller than this interval (about
  ±0.2 points) are noise.
- `oracle`: the ceiling for these candidates. A big gap between `macro_f05` and `oracle` means the matcher is
  losing points. A low `oracle` means the blocker is.
- Use the lockbox at milestones only, and never tune on it. For experiments, compare `decide_oof` and
  `oracle_dev` numbers instead.

The report is also saved as `output/lockbox_<tags>/lockbox_report.md` inside `state.tar`.

### Submitting
1. Finished version → **Output** tab → `submission_<tags>/`.
2. Download `matching_results.tsv` and `candidate_pairs.tsv` (the second is about 1 GB).
3. Upload both to the challenge portal. For the final submission, run once with `CHECK_IDS = True` first
   (adds ~20 min for the full id check).

---

## Part D. Resuming and the experiment loop

Every Kaggle session starts from an empty machine. The notebook saves everything it built into
`/kaggle/working/state.tar`, which becomes part of that version's output. To continue from it:

1. Open the notebook → **Edit**.
2. **Add Input** → **Your Work → Notebooks** → `amz-er-pipeline` (this same notebook) → choose the **latest
   finished version**. Its `state.tar` is restored automatically.
   If Kaggle won't let a notebook use its own output: open the finished version → **Output** tab →
   **New Dataset**, then add that dataset as input instead.
3. **Each time you start a new run, point that input at the newest version** (input's ⋮ menu → change or update
   the version). Otherwise it resumes from an older state. Remove any older state inputs, so there is only
   one `state.tar`.
4. **Save Version → Save & Run All**. Finished steps print `skip ... (done)`.

### Changing code (the normal loop)
1. On the Mac: edit code, run `pytest`, try small things locally if they're quick.
2. Push: `git add -A && git commit -m "..." && git push`.
3. On Kaggle, set the tags for what you changed (below), then **Save & Run All**.
   The notebook always pulls the latest code from `main`.

| You changed | Set in section 1 | What reruns |
|---|---|---|
| `src/normalize.py` or `src/prep.py` | new `NORM_TAG` (e.g. `"n2"`) + new `MODEL_TAG` + `TRAIN_MODEL=True` | everything |
| `src/block.py` | new `BLOCK_TAG` (e.g. `"v4"`) + new `MODEL_TAG` + `TRAIN_MODEL=True` | blocking, features, training, scoring |
| `src/features.py` or `src/train.py` | new `MODEL_TAG` (e.g. `"m2"`) + `TRAIN_MODEL=True` | features, training, scoring |
| Only the threshold | `TAU` | only the output files (seconds) |

**Always change the tag when you change the code behind it.** Finished steps are only reused under the
same tags, so reusing a tag after a code change silently keeps the old results.

Speed-ups while experimenting:
- `RUN_TEST = False` to skip the ~2 h test scoring until you have a model worth submitting.
- `RUN_LOCKBOX = False` for quick dev-only iterations (blocking recall, oracle, OOF F0.5).

### Limits to keep in mind
- 12 hours per run. Weekly CPU quota is generous, but check your remaining hours on your Kaggle profile.
- Saved output up to about 20 GB. `state.tar` is about 7 GB after a full run. If you try many tags, old
  candidate and score folders pile up; delete old ones in the notebook
  (`shutil.rmtree(f"{WORK}/artifacts/score/<old>")`) before `save_state()`.
- Keep the data, notebook and GitHub repo **private**, and check the challenge rules on using Kaggle.

---

## Part E. Troubleshooting

| Symptom | Fix |
|---|---|
| `Challenge data not found` | Add the `amz-er-data` dataset as input (A4 step 4) |
| `git clone` fails with 401/403/404 | Private repo: add the `GITHUB_TOKEN` secret (A3, A4 step 5) and check it hasn't expired |
| `pip install` hangs or fails | Session options → Internet **On** (needs phone verification) |
| `no models ...: set TRAIN_MODEL = True` | New `MODEL_TAG` without training: set `TRAIN_MODEL = True` |
| Run stopped at 12 h or crashed | Add the failed version's output as input and run again; it resumes from the last saved section |
| Results look identical after a code change | You kept the same tag: change it (Part D table) |
