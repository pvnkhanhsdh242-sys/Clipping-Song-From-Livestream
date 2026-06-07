@echo off
setlocal

call "%~dp0docker_cleanup_gpu.bat" %*
if errorlevel 1 (
    echo GPU cleanup failed.
    exit /b 1
)

call "%~dp0docker_streamlit_gpu.bat" --build
if errorlevel 1 (
    echo GPU Streamlit restart failed.
    exit /b 1
)

endlocal