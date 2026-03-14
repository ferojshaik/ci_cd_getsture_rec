# CI/CD for the ML Pipeline (Lesson 7)

**Why CI/CD?**  
- **CI (continuous integration):** Every push/PR is checked automatically. Broken data or code is caught before you run the full pipeline.  
- **CD (continuous delivery):** The full pipeline can run and produce a model artifact you can deploy.

---

## I don’t have cloud – can we do CI/CD with just Git?

**Short answer:** Git alone does **not** run CI/CD. Git only stores history. Something has to **run** the checks and the pipeline. You have two ways to do that without using “your” cloud:

| Option | What runs where | Need GitHub? |
|--------|------------------|---------------|
| **A. Local CI script** | You run a script on your PC before (or instead of) push. Same checks as CI, no server. | No |
| **B. Self-hosted runner** | You use GitHub; when you push, the workflow runs on **your PC** (runner), not on GitHub’s servers. | Yes (repo on GitHub) |

### Option A: Run CI locally (no cloud, no GitHub)

Run the **same checks** as the GitHub Actions CI on your machine before you push:

- **Windows (PowerShell):**
  ```powershell
  cd gesture-recognition
  .\run_ci_local.ps1
  ```
- **Linux / macOS / Git Bash:**
  ```bash
  cd gesture-recognition
  bash run_ci_local.sh
  ```

If the script passes, your data and code are in the same state the cloud CI would check. You can push. No cloud and no GitHub required.

**Full pipeline (train + export)** without cloud: run locally as usual:
```bash
cd gesture-recognition
python run_pipeline.py
```
The model is in `runs/run_<id>/`. No CD server needed.

### Option B: GitHub with a self-hosted runner (CI/CD “in Git”, runs on your PC)

If your repo is on **GitHub** but you don’t want jobs to run on GitHub’s cloud:

1. Add a **self-hosted runner** on your Windows PC:  
   Repo → **Settings** → **Actions** → **Runners** → **New self-hosted runner** → follow the steps (install runner on your machine).
2. In the workflow, use your runner instead of GitHub’s:
   - Open `.github/workflows/ml-validate.yml` and change:
     ```yaml
     runs-on: ubuntu-latest
     ```
     to:
     ```yaml
     runs-on: self-hosted
     ```
   - Do the same in `.github/workflows/ml-pipeline.yml` if you want the full pipeline to run on your PC when you trigger it.

Then when you push (or run the pipeline workflow), GitHub sends the job to **your PC**; nothing runs on GitHub’s servers. So you get “CI/CD in Git” (triggered by Git/GitHub) but execution is on your machine.

---

## Workflows (GitHub Actions)

Workflows live in the **repo root**: `.github/workflows/` (one level above `gesture-recognition/`).

### 1. `ml-validate.yml` (CI)

| What | When | How long |
|------|------|----------|
| Run `validate_data.py` | Every **push** or **pull request** that changes `gesture-recognition/**` | ~1–2 min |
| Check config loads, trainer/export import | Same run | |

- **Purpose:** Fast feedback. If someone adds bad data or breaks the schema, the check fails and the PR doesn’t merge without a fix.
- **No training:** Only validation and import checks, so it stays quick.

### 2. `ml-pipeline.yml` (CD)

| What | When | How long |
|------|------|----------|
| Run full pipeline: validate → train → export → register | **Manual** (“Run workflow” in the Actions tab) and optionally **weekly** (Sunday 02:00 UTC) | ~10–30+ min |
| Upload `gesture-recognition/runs/` as artifact | After pipeline succeeds | |

- **Purpose:** Reproducible training in the cloud. You get a run folder (with `gesture_model_quant.tflite`, `metrics.json`, etc.) as a downloadable artifact.
- **Manual run:** GitHub → **Actions** → **ML Pipeline (Train & Export)** → **Run workflow**.
- **Weekly run:** If you leave the `schedule` in the workflow, it will retrain every week. If you don’t want that, delete or comment out the `schedule` block in `.github/workflows/ml-pipeline.yml`.

---

## What you need

- Repo on **GitHub** (so Actions can run).
- Default branch named `main` or `master` (the workflow uses both in `branches`).
- Training/Testing data committed (or the pipeline will run but may fail at validation/training if data is missing).

---

## After the pipeline runs

1. Open the run in the **Actions** tab.
2. At the bottom, under **Artifacts**, download **ml-run-&lt;run_id&gt;**.
3. Unzip; you get the same structure as `gesture-recognition/runs/` (e.g. `run_YYYYMMDD_HHMMSS/` with `gesture_model_quant.tflite`, `metrics.json`, etc.).
4. Deploy using **DEPLOY.md**: copy the TFLite to app assets or OTA.

---

## Turning off the weekly schedule

If you don’t want automatic weekly training, edit `.github/workflows/ml-pipeline.yml` and remove or comment out:

```yaml
  schedule:
    - cron: '0 2 * * 0'
```

Then the pipeline runs only when you click **Run workflow**.

---

## Summary

| Need | With cloud (GitHub Actions) | Without cloud |
|------|----------------------------|---------------|
| “Did my data/code break?” | CI runs on every push/PR (ml-validate). | Run `run_ci_local.ps1` or `run_ci_local.sh` before push. |
| “Train and get a model” | CD: Actions → ML Pipeline → Run workflow, download artifact. | Run `python run_pipeline.py` locally; model is in `runs/`. |
| “Use Git but run on my PC” | Use a self-hosted runner; change `runs-on` to `self-hosted`. | — |

**Bottom line:** You can do CI/CD without any cloud: run the local CI script before push, and run the pipeline locally. If you use GitHub and want “CI on every push” without cloud, add a self-hosted runner so jobs run on your machine.
