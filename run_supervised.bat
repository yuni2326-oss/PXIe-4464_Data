@echo off
REM PXIe DAQ supervised launcher.
REM supervisor.py runs the app as a child process and restarts it
REM (with the saved initial conditions) on abnormal exit / hang.
cd /d "%~dp0"
python supervisor.py
echo.
echo supervisor stopped.
pause
