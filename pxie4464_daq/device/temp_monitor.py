from __future__ import annotations
import logging
import queue as _queue
import threading
import time
from datetime import datetime

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

# 온도(4353)는 저속이라 read()가 수십~수백 ms 걸릴 수 있다. 메인 스레드(GUI)를
# 막지 않도록, 모듈 임포트 시점(SIP 컨텍스트 밖)에 생성한 전용 폴링 스레드에서
# 주기적으로 읽고 결과를 Qt 시그널로 전달한다. (worker.py와 동일 패턴)
_cmd_q: "_queue.Queue" = _queue.Queue()


class TemperatureMonitor(QObject):
    """PXIe-4353 온도를 주기적으로 폴링하여 emit.

    Signals:
        reading(datetime, np.ndarray): (시각, shape (n_ch,) 섭씨 온도)
        failed(str): 오류 메시지
    """

    reading = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._daq = None
        self._interval = 2.0
        self._running = False

    def start(self, daq, interval_sec: float = 2.0) -> None:
        self._daq = daq
        self._interval = max(float(interval_sec), 0.2)
        self._running = True
        _cmd_q.put(("start", self))

    def stop(self) -> None:
        self._running = False
        _cmd_q.put(("stop", self))

    # 폴링 스레드에서 호출됨
    def _poll_once(self) -> None:
        temps = self._daq.read()
        self.reading.emit(datetime.now(), temps)


def _worker() -> None:
    active: TemperatureMonitor | None = None
    while True:
        # 대기 중인 명령 처리 (논블로킹)
        try:
            kind, mon = _cmd_q.get_nowait()
            if kind == "start":
                if active is not None and active is not mon:
                    try:
                        active._daq.stop()
                    except Exception:
                        pass
                try:
                    mon._daq.start()
                    active = mon
                    logger.info("온도 모니터 시작 (주기 %.1fs)", mon._interval)
                except Exception as exc:
                    active = None
                    try:
                        mon.failed.emit(str(exc))
                    except Exception:
                        pass
            elif kind == "stop":
                if active is not None:
                    try:
                        active._daq.stop()
                    except Exception:
                        pass
                active = None
                logger.info("온도 모니터 정지")
        except _queue.Empty:
            pass

        if active is not None and active._running:
            try:
                active._poll_once()
            except Exception as exc:
                try:
                    active.failed.emit(str(exc))
                except Exception:
                    pass
                # 읽기 실패 시 폴링 중단(장치 문제) — 정지 처리
                try:
                    active._daq.stop()
                except Exception:
                    pass
                active = None
                continue
            time.sleep(active._interval)
        else:
            time.sleep(0.3)  # 유휴


threading.Thread(target=_worker, daemon=True, name="TempMonitor").start()