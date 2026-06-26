#!/usr/bin/env python
"""PXIe DAQ 외부 감시(supervisor).

앱(main.py --autostart)을 자식 프로세스로 실행하고 다음을 감시·복구한다:
  - 비정상 종료(코드 != 0) 또는 크래시        → 새 프로세스로 재시작
  - heartbeat 파일이 HEARTBEAT_TIMEOUT 정체(hang) → 강제 종료 후 재시작
  - 정상 종료(코드 == 0, 사용자 종료)          → supervisor도 함께 종료

Python 3.14 + PyQt5 SIP 상태 손상은 프로세스 내(in-process) 자동 재시작으로
회복되지 않으므로(앱은 code=42로 스스로 종료), 새 프로세스를 띄우는 외부
감시가 필수다. 새 프로세스는 --autostart 로 저장된 초기 실행조건을 자동 재개한다.

사용:  python supervisor.py     (또는 run_supervised.bat)
"""
import logging
import subprocess
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE = Path(__file__).resolve().parent          # pxie4464-daq
MAIN = BASE / "pxie4464_daq" / "main.py"
HEARTBEAT = BASE / "logs" / "heartbeat.txt"
WORKDIR = BASE.parent                            # results/ 가 생성될 위치(기존 레이아웃 유지)

HEARTBEAT_TIMEOUT = 180   # 초: heartbeat 정체가 이 시간을 넘으면 hang으로 판단
POLL_SEC = 10             # 모니터링 주기
RESTART_DELAY = 5         # 재시작 전 대기
EXIT_NORMAL = 0           # 사용자 정상 종료 → supervisor 종료


def _setup_log() -> logging.Logger:
    (BASE / "logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s supervisor: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = RotatingFileHandler(BASE / "logs" / "supervisor.log",
                             maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(sh)
    root.addHandler(fh)
    return logging.getLogger("supervisor")


def _heartbeat_age():
    try:
        return time.time() - float(HEARTBEAT.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def run() -> None:
    log = _setup_log()
    log.info("=" * 50)
    log.info("supervisor 시작 — 대상: %s", MAIN)
    log.info("heartbeat=%s, timeout=%ds", HEARTBEAT, HEARTBEAT_TIMEOUT)
    restarts = 0
    while True:
        # 직전 인스턴스의 오래된 heartbeat로 즉시 hang 오판하지 않도록 초기화
        try:
            HEARTBEAT.parent.mkdir(exist_ok=True)
            HEARTBEAT.write_text(str(time.time()), encoding="utf-8")
        except Exception:
            pass

        cmd = [sys.executable, "-u", str(MAIN), "--autostart"]
        log.info("앱 실행 (재시작 #%d, cwd=%s)", restarts, WORKDIR)
        proc = subprocess.Popen(cmd, cwd=str(WORKDIR))

        while True:
            try:
                code = proc.wait(timeout=POLL_SEC)
            except subprocess.TimeoutExpired:
                age = _heartbeat_age()
                if age is not None and age > HEARTBEAT_TIMEOUT:
                    log.error("heartbeat %d초 정체 — 앱 응답없음(hang) 판단, 강제 종료", int(age))
                    proc.kill()
                    try:
                        proc.wait(timeout=10)
                    except Exception:
                        pass
                    break
                continue
            else:
                if code == EXIT_NORMAL:
                    log.info("앱 정상 종료(code=0, 사용자 종료) — supervisor 종료")
                    return
                log.warning("앱 비정상 종료(code=%s) — 재시작 예정", code)
                break

        restarts += 1
        log.info("%d초 후 재시작...", RESTART_DELAY)
        time.sleep(RESTART_DELAY)


if __name__ == "__main__":
    run()
