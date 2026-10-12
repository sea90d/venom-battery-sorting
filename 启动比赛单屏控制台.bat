@echo off
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m battery_sorting.competition_console --port 8768
pause

