@echo off
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
echo 先执行第一题现场预检：
".venv\Scripts\python.exe" -m battery_sorting.closed_loop preflight --task basic --config config\closed_loop.json
if errorlevel 1 (
  echo 预检未通过，未执行机械臂动作。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tools\run_smooth_t0.py --execute --profile ten-second --wait-for-start
pause
