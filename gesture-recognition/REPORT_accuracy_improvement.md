# Report: Why and How Accuracy Increased from ~63% to ~96%

## Summary

| Pipeline | Features | Architecture | Test accuracy (approx.) |
|----------|----------|--------------|--------------------------|
| **Before** | 12 (time-domain only) | Input(12) → Dense(16) → Dense(3) | ~63–67% |
| **After** | 39 (time + spectral) | Input(39) → Dense(20) → Dense(10) → Dense(3) | ~96% |

The gain comes from **better input representation** (more informative features and more training samples) and a **slightly larger model** that can use that information.

---

## 1. Why the First Pipeline Had Lower Accuracy (~63%)

### 1.1 Very few training samples

- **Before:** Each **recording** = one training sample. With 9 training files (3 TAP, 3 WAVE, 3 SHAKE), the model saw only **9 examples**.
- With so few samples, the model could not learn stable patterns. Small changes in initialization or data order caused big swings in accuracy (e.g. 66% one run, 100% another).
- **Test set:** Only 3 samples (one per class), so a single wrong prediction already cost ~33% accuracy.

### 1.2 Weak feature set (12 time-domain stats only)

- **Before:** Each recording was summarized into **12 numbers**: mean, std, min, max for each of accX, accY, accZ.
- That throws away **all time and frequency structure**. The model could not see:
  - **When** peaks or shakes happened inside the gesture.
  - **How fast** the motion was (e.g. quick tap vs slow wave) — that is in the **frequency** content.
- TAP, WAVE, and SHAKE can look similar in simple stats (e.g. similar mean or range) but differ in **how** the signal changes over time and in frequency. With only 12 stats, the classifier had limited power to separate them.

### 1.3 One vector per recording

- One long recording (e.g. 600+ samples) was compressed into a single 12‑dimensional point. Any variation **within** the gesture (e.g. start vs end of a wave) was averaged away, so the model saw less detailed information per file.

---

## 2. How the New Pipeline Increased Accuracy (~96%)

### 2.1 Many more training samples (windowing)

- **After:** Each recording is split into **overlapping windows** (2000 ms window, 80 ms stride). One 10‑second file produces **many windows**, each with the same gesture label.
- **Result:** 9 recordings → **873 training windows**; 3 test recordings → **303 test windows**.
- The model now sees hundreds of examples per class instead of 3. That allows it to learn more stable, general patterns and reduces sensitivity to a single bad example.

### 2.2 Richer features: time + frequency (39 features)

- **After:** Each window is described by **39 numbers**:
  - **12 time-domain:** mean, std, min, max per axis (same idea as before, but **per window**).
  - **27 spectral:** from an **FFT** (length 16, overlapping frames, log magnitude) — 9 frequency bins × 3 axes.
- **Why this helps:**
  - **TAP:** Short, sharp impulse → energy in higher frequencies; FFT captures that.
  - **WAVE:** Slower, periodic motion → more energy in lower frequencies and a different spectral shape.
  - **SHAKE:** Rapid back‑and‑forth → strong variation in both time and frequency; std and spectral features both help.
- So the model now gets **both** “how big/variable” the signal is (time stats) and “how fast / at what frequencies” it moves (spectral). That is exactly the kind of information used in the screenshot pipeline (Spectral Analysis → Classifier).

### 2.3 Same pipeline as your screenshots

- **Time series:** 2000 ms window, 80 ms stride, zero-pad, 62.5 Hz.
- **Spectral analysis:** FFT, log spectrum, overlapping frames.
- **Classifier:** 39 inputs → Dense(20) → Dense(10) → 3 outputs, 100 epochs, learning rate 0.0005.
- Replicating this design in code gave a fair comparison and aligned the feature set with what already worked for you in the UI.

### 2.4 Slightly larger model (20 → 10 → 3)

- **Before:** 12 → 16 → 3 (very small; enough for 12 features).
- **After:** 39 → 20 → 10 → 3. More inputs and one extra layer let the model use the 39 features without underfitting. The size is still small (TinyML‑friendly).

---

## 3. What Actually Drove the Gain?

| Factor | Before | After | Effect on accuracy |
|--------|--------|--------|---------------------|
| **Number of training samples** | 9 | 873 | Much more stable learning; less overfitting to a few files. |
| **Information per sample** | 12 stats (no frequency) | 39 (time + FFT) | Model can separate TAP vs WAVE vs SHAKE using frequency and time structure. |
| **Use of time structure** | One vector per whole recording | Many windows per recording | Preserves “when” and “where” in the gesture; less averaging out of important parts. |
| **Use of frequency** | None | FFT + log spectrum | Captures speed/rhythm of motion; critical for gesture type. |

The main drivers are: **more data (windowing)** and **better features (spectral + time)**, with a **slightly larger model** to use them.

---

## 4. Conclusion

- **Why it was ~63% before:** Too few samples (9), too little information per sample (12 stats, no frequency), and no use of time structure within the recording.
- **How it reached ~96%:** Windowing gave hundreds of training samples, FFT added spectral features so the model could see “how fast” and “at what frequencies” the motion happens, and the 39‑feature vector plus a 20→10→3 network matched the pipeline that already gave you good accuracy in the screenshots.

This matches the standard approach for gesture/sensor classification: **window the signal → extract time + frequency features → train a small classifier** — and shows why moving from 12 time-only stats to 39 time+spectral features and windowing was enough to take accuracy from ~63% to ~96%.
