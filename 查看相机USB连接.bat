@echo off
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
set PYTHONUTF8=1
".venv\Scripts\python.exe" -c "from battery_sorting.realsense import RealSenseCamera; devices=RealSenseCamera.list_devices(); print('=== RealSense current USB connection ==='); print('No RealSense device detected' if not devices else ''); [print('Camera: '+d.name+'\nSerial: '+d.serial+'\nUSB Type: '+(d.usb_type or 'Unknown')+'\n') for d in devices]"
pause
