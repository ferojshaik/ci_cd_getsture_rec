@echo off
cd /d "%~dp0"
if not defined JAVA_HOME (
    if exist "C:\Program Files\Android\Android Studio\jbr" set "JAVA_HOME=C:\Program Files\Android\Android Studio\jbr"
    if exist "C:\Program Files\Android\Android Studio\jre" set "JAVA_HOME=C:\Program Files\Android\Android Studio\jre"
)
echo Step 1: Copy TFLite model to assets...
call copy_model_to_assets.bat
if errorlevel 1 exit /b 1
echo.
echo Step 2: Building APK (debug)...
if not exist "gradle\wrapper\gradle-wrapper.jar" (
    echo Gradle wrapper JAR not found. Trying to download...
    powershell -ExecutionPolicy Bypass -File "download_gradle_wrapper.ps1"
    if not exist "gradle\wrapper\gradle-wrapper.jar" (
        echo.
        echo Open this folder in Android Studio: File -^> Open -^> GestureApp. Let Gradle sync, then run this script again.
        exit /b 1
    )
)
call gradlew.bat assembleDebug
if errorlevel 1 (
    echo Build failed. Try opening the project in Android Studio and use Build -^> Build Bundle(s) / APK(s) -^> Build APK(s).
    exit /b 1
)
echo.
echo Build succeeded.
echo APK: app\build\outputs\apk\debug\app-debug.apk
echo Install on device: adb install -r app\build\outputs\apk\debug\app-debug.apk
pause
