@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set "POS_MANIFEST=output\singing_clip_train_positive.csv"
set "NEG_MANIFEST=output\singing_clip_train_negative.csv"
set "MODEL_DIR=data\models\singing_candidate"
set "NEG_DIR=data\training_clips\not_singing"
set "SMOKE_MODEL_DIR=output\singing_model_positive_only"
set "EVAL_OUTDIR=output\eval"
set "AUTO_NEG_DIR=%NEG_DIR%\auto"
if "%EPOCHS%"=="" set "EPOCHS=1000"
if "%VALIDATION_SIZE%"=="" set "VALIDATION_SIZE=0.25"
if "%SINGING_THRESHOLD%"=="" set "SINGING_THRESHOLD=0.5"
if "%RUN_EVAL%"=="" set "RUN_EVAL=1"
if "%MAX_NEGATIVE_DURATION_SEC%"=="" (
  set "MAX_NEGATIVE_DURATION_ARG="
) else (
  set "MAX_NEGATIVE_DURATION_ARG=--max-sample-duration-sec %MAX_NEGATIVE_DURATION_SEC%"
)
if "%MAX_NEGATIVES%"=="" (
  set "MAX_NEGATIVE_ARG="
) else (
  set "MAX_NEGATIVE_ARG=--max-negatives %MAX_NEGATIVES%"
)

echo [1/7] Preparing folders...
if not exist "output" mkdir "output"
if not exist "data\training_clips" mkdir "data\training_clips"
if not exist "%NEG_DIR%" mkdir "%NEG_DIR%"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

echo [2/7] Running quick smoke checks...
"%PY%" -m py_compile scripts\build_singing_clip_manifest.py scripts\generate_negative_singing_clips.py scripts\train_singing_candidate_model.py
if errorlevel 1 goto fail
"%PY%" -m pytest tests\test_singing_training.py tests\test_singing_scorer.py -q
if errorlevel 1 goto fail

echo [3/7] Building positive manifest from non-empty output\*\clips folders...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$py='%PY%';" ^
  "$ext=@('.mp4','.wav','.mov','.mkv','.webm');" ^
  "$dirs=Get-ChildItem -Path 'output' -Directory | ForEach-Object { Join-Path $_.FullName 'clips' } | Where-Object { (Test-Path -LiteralPath $_) -and @(Get-ChildItem -LiteralPath $_ -File -ErrorAction SilentlyContinue | Where-Object { $ext -contains $_.Extension.ToLowerInvariant() }).Count -gt 0 };" ^
  "if (-not $dirs) { Write-Error 'No non-empty output\*\clips folders found.' }" ^
  "& $py 'scripts\build_singing_clip_manifest.py' --clips-dir @dirs --output '%POS_MANIFEST%' --label-singing 1"
if errorlevel 1 goto fail
for /f %%C in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "@(Import-Csv '%POS_MANIFEST%').Count"') do set "POS_COUNT=%%C"
echo Found %POS_COUNT% positive clips.

echo [4/7] Checking negative training clips in %NEG_DIR%...
for /f %%C in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "$ext=@('.mp4','.wav','.mov','.mkv','.webm'); @(Get-ChildItem -Path '%NEG_DIR%' -File -Recurse -ErrorAction SilentlyContinue | Where-Object { ($ext -contains $_.Extension.ToLowerInvariant()) -and ($_.FullName -notlike '*\auto\*') }).Count"') do set "MANUAL_NEG_COUNT=%%C"
for /f %%C in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "$ext=@('.mp4','.wav','.mov','.mkv','.webm'); @(Get-ChildItem -Path '%AUTO_NEG_DIR%' -File -Recurse -ErrorAction SilentlyContinue | Where-Object { $ext -contains $_.Extension.ToLowerInvariant() }).Count"') do set "AUTO_NEG_COUNT=%%C"
set /a NEG_COUNT=%MANUAL_NEG_COUNT%+%AUTO_NEG_COUNT%
echo Found %MANUAL_NEG_COUNT% manual negative clips and %AUTO_NEG_COUNT% auto negative clips.

if %NEG_COUNT% LSS %POS_COUNT% (
  echo.
  set /a GEN_COUNT=%POS_COUNT%-%MANUAL_NEG_COUNT%
  if not "%MAX_NEGATIVES%"=="" set "GEN_COUNT=%MAX_NEGATIVES%"
  echo Need more negative clips. Auto-generating !GEN_COUNT! negatives from VOD gaps...
  echo [5/7] Generating automatic negative clips...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "$root=(Resolve-Path '%NEG_DIR%').Path;" ^
    "$target=Join-Path $root 'auto';" ^
    "if (Test-Path -LiteralPath $target) { $resolved=(Resolve-Path -LiteralPath $target).Path; if (-not $resolved.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { throw 'Refusing to remove unexpected path' }; Remove-Item -LiteralPath $resolved -Recurse -Force }"
  if errorlevel 1 goto fail
  "%PY%" scripts\generate_negative_singing_clips.py --output-root "output" --negative-dir "%AUTO_NEG_DIR%" --manifest-output "%NEG_MANIFEST%" --max-negatives !GEN_COUNT! %MAX_NEGATIVE_DURATION_ARG%
  if errorlevel 1 goto fail
  for /f %%C in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "$ext=@('.mp4','.wav','.mov','.mkv','.webm'); @(Get-ChildItem -Path '%NEG_DIR%' -File -Recurse -ErrorAction SilentlyContinue | Where-Object { $ext -contains $_.Extension.ToLowerInvariant() }).Count"') do set "NEG_COUNT=%%C"
  echo Found !NEG_COUNT! negative clips after auto-generation.
  if "!NEG_COUNT!"=="0" goto fail
)

echo [5/7] Building negative manifest from negative clips...
"%PY%" scripts\build_singing_clip_manifest.py --clips-dir "%NEG_DIR%" --output "%NEG_MANIFEST%" --label-singing 0
if errorlevel 1 goto fail

echo [6/7] Training real singing model...
"%PY%" scripts\train_singing_candidate_model.py --manifest "%POS_MANIFEST%" "%NEG_MANIFEST%" --output-dir "%MODEL_DIR%" --epochs %EPOCHS% --validation-size %VALIDATION_SIZE%
if errorlevel 1 goto fail

if "%RUN_EVAL%"=="0" goto done

echo [7/7] Evaluating first available VOD with score mode...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$py='%PY%';" ^
  "$vod='%EVAL_FILE%';" ^
  "if (-not $vod) { $vod=Get-ChildItem -Path 'output' -Recurse -File -Filter *.mp4 | Where-Object { $_.FullName -like '*\vods\*' } | Select-Object -First 1 -ExpandProperty FullName }" ^
  "if (-not $vod) { Write-Warning 'No VOD MP4 found under output\*\vods; skipping evaluation.'; exit 0 }" ^
  "& $py -m app.main --file $vod --outdir '%EVAL_OUTDIR%' --singing-model-mode score --singing-model-path '%MODEL_DIR%' --singing-score-threshold '%SINGING_THRESHOLD%'"
if errorlevel 1 goto fail

:done
echo.
echo Done.
echo Positive manifest: %POS_MANIFEST%
echo Negative manifest: %NEG_MANIFEST%
echo Model dir: %MODEL_DIR%
echo.
echo To run real filtering after checking scores:
echo "%PY%" -m app.main --file "PATH_TO_VOD.mp4" --outdir "output\filtered" --singing-model-mode filter --singing-model-path "%MODEL_DIR%" --singing-score-threshold %SINGING_THRESHOLD%
exit /b 0

:fail
echo.
echo FAILED. Check the error above.
exit /b 1
