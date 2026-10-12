@echo off
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
".venv\Scripts\python.exe" -m battery_sorting.fixed_sequence plan
pause
