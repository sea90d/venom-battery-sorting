@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
set PYTHONUTF8=1
echo Opening Task 2 image console: http://127.0.0.1:8768/?task=sequence
".venv\Scripts\python.exe" -u -m battery_sorting.competition_console --task sequence --port 8768
if errorlevel 1 pause
