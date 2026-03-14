# Deployment (MLOps Lesson 6)

**Why this is part of MLOps:** Deploying a model is a *release*. You must know *which* run produced the model and how to roll back. This checklist makes that explicit.

---

## Before you deploy

1. **Pick a run**  
   Open `runs/registry.csv` and choose the run you want (e.g. highest `test_f1` or `test_accuracy`).

2. **Note the run_id**  
   Example: `run_20250314_153022`. All artifacts are in `runs/run_20250314_153022/`.

3. **Optional: quick sanity check**  
   - `runs/run_<id>/config.json` – config used for training  
   - `runs/run_<id>/metrics.json` – test_accuracy, test_f1  
   - `runs/run_<id>/gesture_model_quant.tflite` – file to deploy

---

## Option A: Bundle model in the app (new APK)

1. Copy the TFLite file:
   ```text
   copy  runs\run_<id>\gesture_model_quant.tflite  →  GestureApp\app\src\main\assets\
   ```
2. Build the APK (e.g. `.\gradlew.bat assembleDebug` from GestureApp).
3. **Record the release:** e.g. in your release notes or a `releases.txt`:  
   `2025-03-14  app version X.Y  →  model run_20250314_153022  (test_f1=0.92)`.

---

## Option B: Over-the-air (OTA) update (same APK, new model)

1. Copy the TFLite file into your OTA host:
   ```text
   copy  runs\run_<id>\gesture_model_quant.tflite  →  ota\model\
   ```
2. Update **version** (and optionally run_id) in `ota/version.json`, e.g.:
   ```json
   { "version": 2, "run_id": "run_20250314_153022" }
   ```
3. Deploy the `ota/` site (e.g. push to Vercel). The app will download the new model on next launch.
4. **Record the release:** e.g. `OTA version 2  →  run_20250314_153022`.

---

## Rollback

- **Bundled:** Ship a new APK that contains the previous `gesture_model_quant.tflite` (from an older `runs/run_<id>/`).
- **OTA:** Point `ota/version.json` (and `ota/model/`) back to the previous model file and redeploy.

Because every run is in `runs/run_<id>/` and listed in `runs/registry.csv`, you always know which model is live and which run to roll back to.
