@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

if "%POS_MANIFEST%"=="" set "POS_MANIFEST=output\singing_clip_train_positive.csv"
if "%NEG_MANIFEST%"=="" set "NEG_MANIFEST=output\singing_clip_train_negative.csv"
if "%MODEL_DIR%"=="" set "MODEL_DIR=data\models\singing_candidate"
if "%NEG_DIR%"=="" set "NEG_DIR=data\training_clips\not_singing"
if "%AUTO_NEG_DIR%"=="" set "AUTO_NEG_DIR=%NEG_DIR%\auto"
if "%EVAL_OUTDIR%"=="" set "EVAL_OUTDIR=output\eval"
if "%BACKEND%"=="" set "BACKEND=sklearn"
if "%DEVICE%"=="" set "DEVICE=auto"
if "%EPOCHS%"=="" set "EPOCHS=1000"
if "%BATCH_SIZE%"=="" set "BATCH_SIZE=8"
if "%LEARNING_RATE%"=="" set "LEARNING_RATE=0.001"
if "%WINDOW_SEC%"=="" set "WINDOW_SEC=12"
if "%WINDOWS_PER_CLIP%"=="" set "WINDOWS_PER_CLIP=4"
if "%VALIDATION_SIZE%"=="" set "VALIDATION_SIZE=0.25"
if "%SINGING_THRESHOLD%"=="" set "SINGING_THRESHOLD=0.5"
if "%RUN_EVAL%"=="" set "RUN_EVAL=1"
set "MAX_NEGATIVE_DURATION_ARG="
set "MAX_NEGATIVE_ARG="
set "EVAL_FILE_ARG="
if not "%MAX_NEGATIVE_DURATION_SEC%"=="" set "MAX_NEGATIVE_DURATION_ARG=--max-negative-duration-sec %MAX_NEGATIVE_DURATION_SEC%"
if not "%MAX_NEGATIVES%"=="" set "MAX_NEGATIVE_ARG=--max-negatives %MAX_NEGATIVES%"
if not "%EVAL_FILE%"=="" set EVAL_FILE_ARG=--eval-file "%EVAL_FILE%"

"%PY%" scripts\train_singing_model_all.py ^
  --positive-manifest "%POS_MANIFEST%" ^
  --negative-manifest "%NEG_MANIFEST%" ^
  --negative-dir "%NEG_DIR%" ^
  --auto-negative-dir "%AUTO_NEG_DIR%" ^
  --model-dir "%MODEL_DIR%" ^
  --backend %BACKEND% ^
  --device %DEVICE% ^
  --epochs %EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --learning-rate %LEARNING_RATE% ^
  --window-sec %WINDOW_SEC% ^
  --windows-per-clip %WINDOWS_PER_CLIP% ^
  --validation-size %VALIDATION_SIZE% ^
  --run-eval %RUN_EVAL% ^
  --eval-outdir "%EVAL_OUTDIR%" ^
  --singing-score-threshold %SINGING_THRESHOLD% ^
  %MAX_NEGATIVE_DURATION_ARG% %MAX_NEGATIVE_ARG% %EVAL_FILE_ARG% %*
if errorlevel 1 (
  echo.
  echo FAILED. Check the error above.
  exit /b 1
)

echo.
echo To run real filtering after checking scores:
echo "%PY%" -m app.main --file "PATH_TO_VOD.mp4" --outdir "output\filtered" --singing-model-mode filter --singing-model-path "%MODEL_DIR%" --singing-score-threshold %SINGING_THRESHOLD%
echo If you passed --model-dir on the command line, use that same path for --singing-model-path.
exit /b 0
