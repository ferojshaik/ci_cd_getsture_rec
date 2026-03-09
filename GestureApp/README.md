# Gesture Recognizer (Android)

Simple app that recognizes **TAP**, **WAVE**, and **SHAKE** using the phone’s accelerometer and the trained TFLite model.

---

## Prerequisites

- **Android Studio** (you have it installed)
- **Java 17** (bundled with Android Studio; BUILD_APK.bat will try to use it if JAVA_HOME is not set)
- **Android SDK** (installed with Android Studio; ensure `local.properties` has the correct `sdk.dir` or open the project in Android Studio once)
- The TFLite model: **gesture_model_quant.tflite** (run **export_tflite.py** in the parent folder first, then **copy_model_to_assets.bat**)

---

## One-time setup (recommended)

1. **Open the project in Android Studio**
   - File → Open → select the **GestureApp** folder (this folder).
   - Let Gradle sync (this creates/updates the Gradle wrapper and `local.properties` with your SDK path).

2. **Ensure the model is in the app**
   - From the **GestureApp** folder, run: **`copy_model_to_assets.bat`**
   - This copies `gesture_model_quant.tflite` from the parent folder (AI_ML) into `app/src/main/assets/`.
   - If you haven’t run **export_tflite.py** in the parent folder yet, do that first so the `.tflite` file exists.

3. **If the build complains about launcher icon**
   - In Android Studio: File → New → Image Asset → create a launcher icon, then build again.

---

## Build APK from terminal

From the **GestureApp** folder (same folder as this README):

```bat
BUILD_APK.bat
```

This will:

1. Copy **gesture_model_quant.tflite** into `app/src/main/assets/` (if it exists in the parent folder).
2. Run **gradlew.bat assembleDebug** to build the debug APK.

**Output APK:**  
`app\build\outputs\apk\debug\app-debug.apk`

If you see *“Gradle wrapper JAR not found”*, open the project in Android Studio once, let sync finish, then run **BUILD_APK.bat** again.

---

## Build from Android Studio

1. File → Open → select **GestureApp**.
2. Run **copy_model_to_assets.bat** (or copy `gesture_model_quant.tflite` into `app/src/main/assets/` manually).
3. Build → Build Bundle(s) / APK(s) → Build APK(s).
4. APK path: **app/build/outputs/apk/debug/app-debug.apk**.

---

## Install and run

- **USB:** Connect the phone, enable USB debugging, then:
  ```bat
  adb install -r app\build\outputs\apk\debug\app-debug.apk
  ```
- Or copy **app-debug.apk** to the phone and open it to install.

Open the app, allow any requested permissions, and do a **TAP**, **WAVE**, or **SHAKE**. The screen shows the recognized gesture.

---

## How it works

- Reads accelerometer at ~62.5 Hz (game rate).
- Keeps a 2 s (125-sample) sliding window.
- Computes the same **39 features** as in Python (12 time-domain + 27 spectral from FFT).
- Runs **gesture_model_quant.tflite** and maps output 0/1/2 to TAP / WAVE / SHAKE.
- Shows the label after a short stability check to reduce flicker.
