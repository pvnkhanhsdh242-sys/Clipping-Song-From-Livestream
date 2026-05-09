@echo off
setlocal

set "COMPOSE_FILE=docker-compose.gpu.yml"
set "DEEP_CLEAN=0"
set "DOCKER_DESKTOP_EXE=%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
if not exist "%DOCKER_DESKTOP_EXE%" set "DOCKER_DESKTOP_EXE=%LocalAppData%\Programs\Docker\Docker\Docker Desktop.exe"

if /I "%~1"=="--deep" set "DEEP_CLEAN=1"

call :ensure_docker_engine
if errorlevel 1 exit /b 1

echo [1/4] Stopping GPU services and removing compose containers...
docker compose -f %COMPOSE_FILE% down --remove-orphans
if errorlevel 1 (
    echo Docker compose down failed.
    exit /b 1
)

echo [2/4] Removing known GPU images (current + legacy names)...
for %%I in ("karaoke-clipper:gpu" "random_project-karaoke-clipper-gpu" "random_project-karaoke-clipper-streamlit-gpu") do (
    docker image inspect "%%~I" >nul 2>&1
    if not errorlevel 1 (
        echo Removing image %%~I
        docker image rm -f "%%~I" >nul 2>&1
    ) else (
        echo Image not found: %%~I
    )
)

echo [3/4] Pruning dangling layers...
docker image prune -f >nul 2>&1

if "%DEEP_CLEAN%"=="1" (
    echo [4/6] Deep clean: pruning Docker builder cache...
    docker builder prune -a -f >nul 2>&1

    echo [5/6] Deep clean: pruning unused volumes...
    docker volume prune -f >nul 2>&1

    echo [6/6] Deep cleanup complete.
) else (
    echo [4/4] Cleanup complete.
    echo Tip: run docker_cleanup_gpu.bat --deep to also remove builder cache and unused volumes.
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