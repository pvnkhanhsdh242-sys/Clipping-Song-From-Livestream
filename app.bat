					@echo off
setlocal EnableDelayedExpansion

set "ROOT_DIR=%~dp0"
set "IGNORE_WARNINGS=0"
set "MODE=auto"
set "GPU_ARGS="
set "ALLOW_FALLBACK=1"
set "DRY_RUN=0"
set "SHOW_HELP=0"
for %%A in (%*) do (
	if /I "%%~A"=="--ignore-warnings" set "IGNORE_WARNINGS=1"
	if /I "%%~A"=="--docker-ui" set "MODE=docker"
	if /I "%%~A"=="--gpu" set "MODE=docker"
	if /I "%%~A"=="--cpu" set "MODE=cpu"
	if /I "%%~A"=="--local" set "MODE=cpu"
	if /I "%%~A"=="--build" set "GPU_ARGS=!GPU_ARGS! --build"
	if /I "%%~A"=="--logs" set "GPU_ARGS=!GPU_ARGS! --logs"
	if /I "%%~A"=="--no-fallback" set "ALLOW_FALLBACK=0"
	if /I "%%~A"=="--dry-run" set "DRY_RUN=1"
	if /I "%%~A"=="--help" set "SHOW_HELP=1"
	if /I "%%~A"=="-h" set "SHOW_HELP=1"
)

if "%SHOW_HELP%"=="1" (
	call :usage
	goto done
)

if "%IGNORE_WARNINGS%"=="1" (
	set "PYTHONWARNINGS=ignore::UserWarning,ignore::RuntimeWarning"
)

if "%DRY_RUN%"=="1" (
	call :dry_run
	goto done
)

if /I "%MODE%"=="cpu" goto cpu_ui
if /I "%MODE%"=="docker" goto docker_ui

echo Starting Karaoke Clipper. Trying Docker GPU UI first...
call :start_docker_ui
if not errorlevel 1 goto done
if "%ALLOW_FALLBACK%"=="0" goto docker_failed

echo.
echo Docker GPU launch failed. Falling back to local CPU Streamlit.
call :start_cpu_ui
goto done

:docker_ui
echo Starting Karaoke Clipper Docker GPU UI...
call :start_docker_ui
if not errorlevel 1 goto done
if "%ALLOW_FALLBACK%"=="0" goto docker_failed

echo.
echo Docker GPU launch failed. Falling back to local CPU Streamlit.
call :start_cpu_ui
goto done

:cpu_ui
call :start_cpu_ui
goto done

:start_docker_ui
call "%ROOT_DIR%scripts\windows\docker_streamlit_gpu.bat" %GPU_ARGS%
exit /b %errorlevel%

:start_cpu_ui
call :stop_gpu_service
set "KARAOKE_FORCE_DEVICE=cpu"
echo Starting Karaoke Clipper local CPU UI...
call "%ROOT_DIR%scripts\windows\run_streamlit_chrome.bat"
exit /b %errorlevel%

:stop_gpu_service
if exist "%ROOT_DIR%docker-compose.gpu.yml" (
	docker compose -f "%ROOT_DIR%docker-compose.gpu.yml" stop karaoke-clipper-streamlit-gpu >nul 2>&1
)
exit /b 0

:docker_failed
echo Docker GPU launch failed and fallback is disabled.
exit /b 1

:dry_run
echo Karaoke Clipper launcher dry run
echo   Mode: %MODE%
echo   Docker GPU command: "%ROOT_DIR%scripts\windows\docker_streamlit_gpu.bat"%GPU_ARGS%
if "%ALLOW_FALLBACK%"=="1" (
	echo   CPU fallback command: set KARAOKE_FORCE_DEVICE=cpu ^& "%ROOT_DIR%scripts\windows\run_streamlit_chrome.bat"
) else (
	echo   CPU fallback command: disabled
)
echo   Ignore warnings: %IGNORE_WARNINGS%
exit /b 0

:usage
echo Usage: app.bat [options]
echo.
echo Default behavior:
echo   Try Docker GPU Streamlit first. If Docker/GPU startup fails, run local CPU Streamlit.
echo.
echo Options:
echo   --cpu, --local       Run local CPU Streamlit only.
echo   --gpu, --docker-ui   Try Docker GPU Streamlit first.
echo   --build              Rebuild the Docker GPU image before startup.
echo   --logs               Follow Docker GPU logs after startup.
echo   --no-fallback        Do not fall back to CPU if Docker GPU fails.
echo   --ignore-warnings    Suppress common Python runtime warnings.
echo   --dry-run            Print the launch plan without starting anything.
echo   --help, -h           Show this help.
exit /b 0

:done

endlocal
