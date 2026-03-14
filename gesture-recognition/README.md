# Gesture Recognition with TinyML + OTA

End-to-end project: train a 4-class gesture classifier (TAP, WAVE, SHAKE, IDLE) in Python, export to TFLite, run on Android, and update the model over the air (OTA) without releasing a new APK.

---

## Project structure

| Folder / file | Purpose |
|---------------|--------|
| **Training/** | JSON recordings for training (e.g. `TAP.xxx.json`, `WAVE.xxx.json`, …). |
| **Testing/** | JSON recordings for testing (same format). |
| **data_loader.py** | Loads JSONs from Training/ and Testing/; extracts label from filename. |
| **dataset.py** | Builds 39 features per 2 s window (time + FFT); 62.5 Hz, 2000 ms window, 80 ms stride. |
| **trainer.py** | Trains a small MLP (Input 39 → Dense 20 → Dense 10 → 4 classes); saves SavedModel. |
| **export_tflite.py** | Converts SavedModel to TFLite (float + int8 quantized); writes `gesture_model.tflite` and `gesture_model_quant.tflite`. |
| **saved_model/** | Keras/TF SavedModel (created by trainer). |
| **ota/** | Static site (version.json + model file) for OTA; deploy to Vercel. See **ota/README.md**. |
| **GestureApp/** | Android (Kotlin) app: accelerometer → 39 features → TFLite inference → OTA check. See **GestureApp/README.md**. |

---

## Quick start

### 1. Data

- Put training JSONs in **Training/** and test JSONs in **Testing/**.
- Filename format: `LABEL.xxx.json` (e.g. `TAP.abc123.json`). Labels: TAP, WAVE, SHAKE, IDEAL.

### 2. Train

```bash
cd AI_ML
python trainer.py
```

- Uses stratified train/val split, class weights, early stopping.
- Saves model to **saved_model/**.

### 3. Export to TFLite

```bash
python export_tflite.py
```

- Produces **gesture_model.tflite** and **gesture_model_quant.tflite** (use quantized in the app).

### 4. Android app

- Copy **gesture_model_quant.tflite** to **GestureApp/app/src/main/assets/**.
- Open **GestureApp** in Android Studio and build, or run `.\gradlew.bat assembleDebug` from GestureApp.
- APK: `GestureApp/app/build/outputs/apk/debug/app-debug.apk`.
- The app shows the current **model version** (bundled or OTA) and checks for updates on launch.

### 5. OTA (optional)

- Deploy **ota/** to Vercel (see **ota/README.md**).
- When you retrain: copy the new **gesture_model_quant.tflite** into **ota/model/**, bump **version** in **ota/version.json**, push to GitHub. The app will download the new model on next launch.

---

## Requirements

- **Python 3** with TensorFlow, NumPy; scikit-learn for stratified split.
- **Android Studio** (or Gradle + JDK) for building the app.
- **Vercel** (or any static host) for OTA.

---

## Model

- **Type:** Supervised, multi-class classification (small feedforward NN).
- **Input:** 39 features (12 time-domain + 27 spectral from accX, accY, accZ).
- **Output:** 4 classes — TAP (0), WAVE (1), SHAKE (2), IDLE (3).
- **Pipeline:** 62.5 Hz, 2 s window, 80 ms stride; FFT length 16, log spectrum; matches common impulse-style setups.

---

## Docs

- **ota/README.md** — Deploy OTA site to Vercel, update model and version.
- **GestureApp/README.md** — Build and run the Android app, copy model to assets.
