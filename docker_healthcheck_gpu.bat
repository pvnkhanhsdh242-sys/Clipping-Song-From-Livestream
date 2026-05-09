@echo off
setlocal
set "DOCKER_GPU_PLATFORM=linux/amd64"
set "DOCKER_DESKTOP_EXE=%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
if not exist "%DOCKER_DESKTOP_EXE%" set "DOCKER_DESKTOP_EXE=%LocalAppData%\Programs\Docker\Docker\Docker Desktop.exe"

call :ensure_docker_engine
if errorlevel 1 exit /b 1

docker run --rm --platform %DOCKER_GPU_PLATFORM% --gpus all karaoke-clipper:gpu python scripts/container_runtime.py healthcheck
if errorlevel 1 (
    echo Docker GPU healthcheck failed.
    exit /b 1
)

echo Docker GPU healthcheck passed.
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