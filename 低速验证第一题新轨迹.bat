@echo off
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
set PYTHONUTF8=1
".venv\Scripts\python.exe" tools\run_smooth_t0.py --execute --profile trajectory-check --wait-for-start
pause
