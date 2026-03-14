# MLOps Step-by-Step: Learn by Doing

This document is your **syllabus**. We implement one lesson at a time in this project. Each lesson has:
- **Why** it matters (professional reason)
- **What** we add (concrete change)
- **How** to run or check it

---

## Lesson 1: Config & Reproducibility
**Why:** In real teams, "which learning rate did we use?" and "which TensorFlow version?" must be answered from files, not memory. Reproducibility = same code + same config + same data → same model.

**What we do:**
- Put all training knobs in one **config file** (e.g. `config.yaml`).
- Pin **exact library versions** in `requirements.txt`.
- Trainer reads config so every run is documented.

**How:** Run `python trainer.py`; it uses `config.yaml`. Change `epochs` or `learning_rate` in `config.yaml` and re-run to see the effect.

---

## Lesson 2: Data Validation & Schema
**Why:** Most production ML bugs come from bad or changed data. Validating before training saves hours of debugging.

**What we do:**
- Define a **data schema** (what each JSON must contain).
- A **validation script** that runs before training: schema check, train/test leakage, minimum samples.

**How:** Run `python validate_data.py` before training. Pipeline will run it automatically in Lesson 5.

---

## Lesson 3: Experiment Tracking
**Why:** You need to compare runs: "Run 3 had better F1 than Run 1; what was different?" Without logging, you can't answer.

**What we do:**
- Each training run gets a **run_id** (e.g. timestamp or UUID).
- We save **config + metrics** (loss, accuracy, F1) under `runs/run_<id>/`.
- Optional: simple table (CSV) of all runs.

**How:** After training, look in `runs/` for `config.json` and `metrics.json` for that run.

---

## Lesson 4: Model Versioning & Registry
**Why:** Production needs one answer: "Which model is live?" A registry stores every model with its run_id and metrics so you can promote or rollback.

**What we do:**
- Save the SavedModel and TFLite under `runs/run_<id>/` (or copy there after export).
- A **registry** (e.g. `runs/registry.csv`) with columns: run_id, path, test_accuracy, test_f1, created_at.

**How:** Open `runs/registry.csv` to see all runs and pick which one to deploy.

---

## Lesson 5: One-Command ML Pipeline
**Why:** Manually running "validate → train → export → test" is error-prone. One script that does all steps is the first step toward CI/CD.

**What we do:**
- A single script **`run_pipeline.py`** (or `run_ml_pipeline.py`): validate data → train → export TFLite → run a quick inference check.
- Same run_id used for the whole pipeline.

**How:** Run `python run_pipeline.py`; it runs the full pipeline and writes to `runs/`.

---

## Lesson 6: Deployment & OTA as Release
**Why:** Deploying a model is a "release." It should be traceable: which run produced this model, and how to roll back.

**What we do:**
- Document the **release process**: copy which file from `runs/run_<id>/` to OTA or app assets.
- Add **version metadata** (e.g. in `ota/version.json`) that includes run_id or model version.

**How:** Follow the checklist in `DEPLOY.md` (or the OTA readme) when you deploy a new model.

---

## Lesson 7: CI/CD

**Why:** Yes, we need CI/CD. Without it, only the person who runs the pipeline gets the model; there’s no automatic check that data/code still work after a change. CI = run checks on every push/PR. CD = run the full pipeline in the cloud (optional) and get artifacts you can deploy.

**What we do:**
- **CI** (`.github/workflows/ml-validate.yml`): On every push/PR that touches `gesture-recognition/`, run data validation and a quick config/import check. Fast feedback, no training.
- **CD** (`.github/workflows/ml-pipeline.yml`): On manual trigger (or on a schedule), run the full pipeline in GitHub Actions, then upload the `runs/` folder as an artifact so you can download the TFLite and deploy.

**How:**
- Push (or open a PR) that changes `gesture-recognition/` → CI runs automatically. Check the **Actions** tab.
- To train in the cloud: **Actions** → **ML Pipeline (Train & Export)** → **Run workflow**. When it finishes, download the artifact `ml-run-<id>` to get the new run folder (including `gesture_model_quant.tflite`).

**No cloud?** Run CI locally: `.\run_ci_local.ps1` (Windows) or `bash run_ci_local.sh` (Linux/macOS). Full pipeline: `python run_pipeline.py` on your machine. Or use GitHub with a **self-hosted runner** so workflows run on your PC. See **CICD.md** for details.

---

## Summary Table

| Lesson | Concept              | Main artifact(s)                    |
|--------|----------------------|-------------------------------------|
| 1      | Config & repro       | `config.yaml`, `requirements.txt`   |
| 2      | Data validation      | `validate_data.py`, schema doc      |
| 3      | Experiment tracking  | `runs/run_<id>/config.json`, `metrics.json` |
| 4      | Model registry       | `runs/run_<id>/saved_model`, `registry.csv` |
| 5      | Pipeline             | `run_pipeline.py`                   |
| 6      | Deployment           | `DEPLOY.md`, OTA versioning         |
| 7      | CI/CD                | `.github/workflows/ml-validate.yml`, `ml-pipeline.yml` |

After these seven lessons, you have a minimal but real MLOps setup: reproducible runs, validated data, tracked experiments, versioned models, one-command pipeline, deployment checklist, and CI/CD.
