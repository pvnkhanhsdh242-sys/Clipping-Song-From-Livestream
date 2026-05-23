@echo off
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
set "DOCKER_GPU_PLATFORM=linux/amd64"
set "DOCKER_DESKTOP_EXE=%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
if not exist "%DOCKER_DESKTOP_EXE%" set "DOCKER_DESKTOP_EXE=%LocalAppData%\Programs\Docker\Docker\Docker Desktop.exe"
set "IMAGE_TAG=karaoke-clipper:train-gpu"
set "SHARED_IMAGE_TAG=karaoke-clipper:gpu"
set "BUILD_IMAGE=0"
set "RUN_ARGS="

:parse_args
if "%~1"=="" goto parsed_args
if /I "%~1"=="--build" (
  set "BUILD_IMAGE=1"
  shift
  goto parse_args
)
if defined RUN_ARGS (
  set "RUN_ARGS=%RUN_ARGS% %1"
) else (
  set "RUN_ARGS=%1"
)
shift
goto parse_args

:parsed_args
call :ensure_docker_engine
if errorlevel 1 exit /b 1

if "%BUILD_IMAGE%"=="1" (
  docker build --platform %DOCKER_GPU_PLATFORM% -f Dockerfile.gpu --target base-gpu -t %SHARED_IMAGE_TAG% -t %IMAGE_TAG% "%ROOT_DIR%."
  if errorlevel 1 (
    echo GPU training image build failed.
    exit /b 1
  )
  if not defined RUN_ARGS (
    echo Built %SHARED_IMAGE_TAG% and %IMAGE_TAG% from the same image.
    exit /b 0
  )
)

docker image inspect %IMAGE_TAG% >nul 2>&1
if errorlevel 1 (
  echo Image %IMAGE_TAG% was not found. Building shared GPU image now...
  docker build --platform %DOCKER_GPU_PLATFORM% -f Dockerfile.gpu --target base-gpu -t %SHARED_IMAGE_TAG% -t %IMAGE_TAG% "%ROOT_DIR%."
  if errorlevel 1 (
    echo GPU training image build failed.
    exit /b 1
  )
)

docker run --rm --gpus all ^
  --platform %DOCKER_GPU_PLATFORM% ^
  -v "%ROOT_DIR%output:/app/output" ^
  -v "%ROOT_DIR%data:/app/data" ^
  -v "%ROOT_DIR%secret:/app/secret" ^
  -v "%ROOT_DIR%.cache:/app/.cache" ^
  %IMAGE_TAG% python scripts/container_runtime.py run --require-cuda -- python scripts/train_singing_model_all.py --backend pytorch --device cuda --epochs 5 %RUN_ARGS%

if errorlevel 1 (
  echo GPU singing training failed.
  exit /b 1
)

endlocal
exit /b 0

:ensure_docker_engine
docker info >nul 2>&1
if not errorlevel 1 exit /b 0

echo Docker engine is not ready.
if exist "%DOCKER_DESKTOP_EXE%" (
  echo Starting Docker Desktop...
  start "" "%DOCKER_DESKTOP_EXE%"
  call :wait_for_docker 120
) else (
  echo Docker Desktop executable was not found.
)

docker info >nul 2>&1
if not errorlevel 1 exit /b 0

echo Docker Linux engine is still unavailable.
echo Please open Docker Desktop, wait until it says "Engine running", then run this script again.
exit /b 1

:wait_for_docker
set "WAIT_SECONDS=%~1"
if "%WAIT_SECONDS%"=="" set "WAIT_SECONDS=120"
for /l %%i in (1,1,%WAIT_SECONDS%) do (
  docker info >nul 2>&1
  if not errorlevel 1 exit /b 0
  timeout /t 1 >nul
)
exit /b 1
