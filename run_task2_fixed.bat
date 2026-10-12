@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" goto missing_python
if /i "%~1"=="--plan-only" goto plan

echo Task 2: fixed taught positions, yellow-P1 / green-P2 / blue-P3.
echo Hardware preflight...
".venv\Scripts\python.exe" -m battery_sorting.fixed_sequence preflight
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m battery_sorting.fixed_sequence run --execute --auto-start
if errorlevel 1 goto failed
echo Task completed.
goto finish

:plan
".venv\Scripts\python.exe" -m battery_sorting.fixed_sequence plan
exit /b %errorlevel%

:missing_python
echo ERROR: project Python is missing: .venv\Scripts\python.exe
goto failed

:failed
echo ERROR: startup or execution failed. Read the message above.
echo The program will not automatically retry.
pause
exit /b 1

:finish
pause
exit /b 0
