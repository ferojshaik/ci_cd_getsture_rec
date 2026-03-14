# ---------------------------------------------------------------------------
# Run CI checks locally (no cloud)
# ---------------------------------------------------------------------------
# Same checks as .github/workflows/ml-validate.yml:
#   1. Validate data
#   2. Config and imports check
# Run this before you push, or use it as a pre-push hook.
# ---------------------------------------------------------------------------

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "CI (local): data validation + config/imports" -ForegroundColor Cyan
Write-Host ""

# 1. Validate data
Write-Host "[1/2] Validate data..."
& python validate_data.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "CI failed: data validation" -ForegroundColor Red
    exit 1
}

# 2. Config and imports
Write-Host ""
Write-Host "[2/2] Config and imports..."
& python -c "from mlops_config import load_config; c = load_config(); assert 'random_seed' in c; print('config OK')"
if ($LASTEXITCODE -ne 0) { exit 1 }

& python -c "from trainer import build_model; build_model(); print('trainer OK')"
if ($LASTEXITCODE -ne 0) { exit 1 }

& python -c "from export_tflite import convert_to_tflite; print('export_tflite OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "CI failed: config/imports" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "CI passed (local). Safe to push." -ForegroundColor Green
exit 0
