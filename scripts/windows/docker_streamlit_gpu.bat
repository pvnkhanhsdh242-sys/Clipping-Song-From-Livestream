@echo off
setlocal

for %%I in ("%~dp0..\..") do set "ROOT_DIR=%%~fI"
set "COMPOSE_FILE=%ROOT_DIR%\docker-compose.gpu.yml"
set "SERVICE=karaoke-clipper-streamlit-gpu"
set "IMAGE=karaoke-clipper:gpu"
set "FORCE_BUILD=0"
set "FOLLOW_LOGS=0"
set "APP_URL=http://localhost:8501"
set "DOCKER_GPU_PLATFORM=linux/amd64"
set "DOCKER_DESKTOP_EXE=%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
if not exist "%DOCKER_DESKTOP_EXE%" set "DOCKER_DESKTOP_EXE=%LocalAppData%\Programs\Docker\Docker\Docker Desktop.exe"

for %%A in (%*) do (
    if /I "%%~A"=="--build" set "FORCE_BUILD=1"
    if /I "%%~A"=="--logs" set "FOLLOW_LOGS=1"
)

call :ensure_docker_engine
if errorlevel 1 exit /b 1

if "%FORCE_BUILD%"=="1" goto build

docker image inspect "%IMAGE%" >nul 2>&1
if errorlevel 1 (
    echo GPU image not found. Building once before startup...
    goto build
)

goto up

:build
docker compose --project-directory "%ROOT_DIR%" -f "%COMPOSE_FILE%" build %SERVICE%
if errorlevel 1 (
    echo Docker GPU Streamlit build failed.
    exit /b 1
)

:up
docker compose --project-directory "%ROOT_DIR%" -f "%COMPOSE_FILE%" up -d %SERVICE%
if errorlevel 1 (
    echo Docker GPU Streamlit startup failed.
    exit /b 1
)

call :wait_for_streamlit 90
if errorlevel 1 (
    echo Docker GPU Streamlit started, but %APP_URL% did not respond before timeout.
    exit /b 1
)

call :open_browser

if "%FOLLOW_LOGS%"=="1" (
    docker compose --project-directory "%ROOT_DIR%" -f "%COMPOSE_FILE%" logs -f %SERVICE%
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
  powershell -NoProfile -Command "Start-Sleep -Seconds 1" >nul
)
exit /b 1

:wait_for_streamlit
set "WAIT_SECONDS=%~1"
if "%WAIT_SECONDS%"=="" set "WAIT_SECONDS=90"
for /l %%i in (1,1,%WAIT_SECONDS%) do (
    powershell -NoProfile -Command "try { $response = Invoke-WebRequest -UseBasicParsing -Uri '%APP_URL%' -TimeoutSec 1; if ($response.StatusCode -ge 200) { exit 0 } } catch { } ; exit 1" >nul 2>&1
    if not errorlevel 1 exit /b 0
    powershell -NoProfile -Command "Start-Sleep -Seconds 1" >nul
)
exit /b 1

:open_browser
set "CHROME_EXE="
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set "CHROME_EXE=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not defined CHROME_EXE if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set "CHROME_EXE=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"

if defined CHROME_EXE (
    start "Karaoke Clipper GPU" "%CHROME_EXE%" "%APP_URL%"
) else (
    start "Karaoke Clipper GPU" "%APP_URL%"
)
exit /b 0
