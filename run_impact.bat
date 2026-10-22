@echo off
REM Impact hammer test tool (separate from the monitoring app).
REM Close the monitoring app first: both cannot use the DAQ at the same time.
cd /d "%~dp0"
python -m pxie4464_daq.impact_main
echo.
echo impact tool stopped.
pause
