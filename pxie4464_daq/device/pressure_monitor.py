from __future__ import annotations
import logging
import queue as _queue
import threading
import time
from datetime import datetime

import numpy as np  # noqa: F401  (일관성/타입용)
from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

# 압력(USB-6001)도 온도와 마찬가지로 저속이라 메인 스레드(GUI)를 막지 않도록
# 모듈 임포트 시점(SIP 밖)에 생성한 전용 폴링 스레드에서 주기적으로 읽고
# Qt 시그널로 전달한다. (temp_monitor와 동일 패턴, 별도 스레드)
_cmd_q: "_queue.Queue" = _queue.Queue()

_RETRY_MIN_SEC = 3.0    # 읽기 실패 시 재시도 최소 대기
_RECREATE_AFTER = 3     # 연속 실패가 이 횟수 이상이면 태스크 재생성(장치 복귀 대비)


class PressureMonitor(QObject):
    """USB-6001 압력을 주기적으로 폴링하여 emit.

    Signals:
        reading(datetime, np.ndarray): (시각, shape (n_ch,) bar)
        failed(str): 오류 메시지
    """

    reading = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._daq = None
        self._interval = 1.0
        self._running = False
        self._fail_streak = 0

    def start(self, daq, interval_sec: float = 1.0) -> None:
        self._daq = daq
        self._interval = max(float(interval_sec), 0.1)
        self._running = True
        self._fail_streak = 0
        _cmd_q.put(("start", self))

    def stop(self) -> None:
        self._running = False
        _cmd_q.put(("stop", self))

    def _poll_once(self) -> None:
        vals = self._daq.read()
        self.reading.emit(datetime.now(), vals)


def _worker() -> None:
    active = None
    while True:
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
                    logger.info("압력 모니터 시작 (주기 %.1fs)", mon._interval)
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
                logger.info("압력 모니터 정지")
        except _queue.Empty:
            pass

        if active is not None and active._running:
            try:
                active._poll_once()
                active._fail_streak = 0        # 성공 → 실패 카운터 리셋
            except Exception as exc:
                # 읽기 실패해도 폴링을 멈추지 않는다(장시간 무인 운전 중 일시 오류 대비).
                active._fail_streak += 1
                streak = active._fail_streak
                if streak == 1 or streak % 10 == 0:
                    logger.warning("압력 읽기 실패(%d회): %s — 재시도(로깅 계속)", streak, exc)
                    try:
                        active.failed.emit(str(exc))
                    except Exception:
                        pass
                if streak >= _RECREATE_AFTER:
                    # 지속 실패 → 태스크 재생성 시도(장치 복귀 시 자동 회복)
                    try:
                        active._daq.stop()
                    except Exception:
                        pass
                    try:
                        active._daq.start()
                    except Exception:
                        pass
                time.sleep(max(active._interval, _RETRY_MIN_SEC))
                continue
            time.sleep(active._interval)
        else:
            time.sleep(0.2)


threading.Thread(target=_worker, daemon=True, name="PressureMonitor").start()
