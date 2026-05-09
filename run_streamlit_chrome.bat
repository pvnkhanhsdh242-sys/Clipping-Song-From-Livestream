@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "APP_PATH=%ROOT_DIR%app\ui\streamlit_app.py"
set "PYTHON_EXE=%ROOT_DIR%.venv\Scripts\python.exe"
set "APP_URL=http://localhost:8501"
set "ALREADY_RUNNING=0"

powershell -NoProfile -Command "try { $response = Invoke-WebRequest -UseBasicParsing -Uri '%APP_URL%' -TimeoutSec 1; if ($response.StatusCode -ge 200) { exit 0 } } catch { } ; exit 1" >nul 2>&1
if not errorlevel 1 set "ALREADY_RUNNING=1"

if "%ALREADY_RUNNING%"=="1" (
    echo Streamlit already running at %APP_URL%. Skipping launch.
    goto :end
)

if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

start "Karaoke Clipper" /b "%PYTHON_EXE%" -m streamlit run "%APP_PATH%" --server.headless true

for /l %%i in (1,1,60) do (
    powershell -NoProfile -Command "try { $response = Invoke-WebRequest -UseBasicParsing -Uri '%APP_URL%' -TimeoutSec 1; if ($response.StatusCode -ge 200) { exit 0 } } catch { } ; exit 1" >nul 2>&1
    if not errorlevel 1 goto :open_browser
    timeout /t 1 >nul
)

:open_browser
set "CHROME_EXE="
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set "CHROME_EXE=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not defined CHROME_EXE if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set "CHROME_EXE=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"

if defined CHROME_EXE (
    start "Karaoke Clipper" "%CHROME_EXE%" "%APP_URL%"
) else (
    start "Karaoke Clipper" "%APP_URL%"
)

:end
endlocal