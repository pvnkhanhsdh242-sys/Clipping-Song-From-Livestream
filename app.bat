@echo off
setlocal

set "IGNORE_WARNINGS=0"
for %%A in (%*) do (
	if /I "%%~A"=="--ignore-warnings" set "IGNORE_WARNINGS=1"
)

if "%IGNORE_WARNINGS%"=="1" (
	set "PYTHONWARNINGS=ignore::UserWarning,ignore::RuntimeWarning"
)

call "%~dp0run_streamlit_chrome.bat"

endlocal