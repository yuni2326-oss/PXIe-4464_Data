@echo off
REM PXIe DAQ 감시 실행 — supervisor가 앱을 자식 프로세스로 띄우고
REM 비정상 종료/응답없음 시 초기 실행조건 그대로 새 프로세스로 재시작한다.
cd /d "%~dp0"
python supervisor.py
echo.
echo supervisor가 종료되었습니다.
pause
