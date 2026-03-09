@echo off
set ASSETS=app\src\main\assets
set MODEL=gesture_model_quant.tflite
if not exist "%ASSETS%" mkdir "%ASSETS%"
if exist "..\%MODEL%" (
    copy /Y "..\%MODEL%" "%ASSETS%\%MODEL%" >nul
    echo Copied ..\%MODEL% to %ASSETS%\
) else if exist "%MODEL%" (
    copy /Y "%MODEL%" "%ASSETS%\%MODEL%" >nul
    echo Copied %MODEL% to %ASSETS%\
) else (
    echo ERROR: %MODEL% not found.
    echo Run export_tflite.py in the parent folder first, then run this script again.
    exit /b 1
)
