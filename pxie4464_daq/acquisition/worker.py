from __future__ import annotations
import logging
import queue as _queue
import threading

from PyQt5.QtCore import QObject, pyqtSignal

from pxie4464_daq.device.daq import _DAQBase

logger = logging.getLogger(__name__)

# Python 3.14 + PyQt5 SIP 디스패치 컨텍스트에서 threading.Thread() 생성 자체가 실패함.
# (threading.Thread.__init__ → Event() → Condition(Lock()) → TypeError)
# 모듈 임포트 시점(SIP 컨텍스트 밖)에 실행자 스레드를 미리 생성하고
# queue.Queue 로 작업을 전달하여 우회.
_work_q: _queue.Queue = _queue.Queue()
_done_q: _queue.Queue = _queue.Queue()


def _acq_runner() -> None:
    while True:
        fn = _work_q.get()
        if fn is None:
            return
        try:
            fn()
        finally:
            _done_q.put(None)


threading.Thread(target=_acq_runner, daemon=True, name="AcquisitionWorker").start()


class AcquisitionWorker(QObject):
    """백그라운드 연속 수집.

    모듈 임포트 시 미리 생성된 실행자 스레드(_acq_runner)에 _run()을 큐로 전달.
    SIP 디스패치 컨텍스트 안에서 threading 객체를 생성하지 않는다.

    Signals:
        data_ready(object): shape (n_channels, N) numpy 배열
        error_occurred(str): 오류 메시지
    """

    data_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, daq: _DAQBase, parent=None):
        super().__init__(parent)
        self._daq = daq
        self._running = False
        self._started = False

    def start(self) -> None:
        # 이전 stop() timeout으로 남은 stale 완료 신호를 제거
        while not _done_q.empty():
            try:
                _done_q.get_nowait()
            except _queue.Empty:
                break
        self._running = True
        self._started = True
        _work_q.put(self._run)
        logger.info("AcquisitionWorker 시작")

    def _run(self) -> None:
        try:
            self._daq.start()
            while self._running:
                data = self._daq.read()
                self.data_ready.emit(data)
        except Exception as exc:
            # 오류 발생 → 워커는 더 이상 데이터를 공급하지 않음.
            # isRunning()이 즉시 False를 반환하도록 플래그를 내려
            # Heartbeat "running" 오보 및 멈춘 버퍼 저장을 방지한다.
            self._running = False
            try:
                msg = str(exc)
            except Exception:
                msg = f"{type(exc).__name__}: (오류 메시지 변환 실패 — 장치 연결 끊김 의심)"
            try:
                logger.error("AcquisitionWorker error: %s", msg)
            except Exception:
                pass
            try:
                self.error_occurred.emit(msg)
            except Exception:
                pass
        finally:
            try:
                self._daq.stop()
            except Exception as exc:
                try:
                    logger.warning("DAQ stop error (ignored): %s", exc)
                except Exception:
                    pass

    def stop(self) -> None:
        self._running = False
        if self._started:
            self._started = False
            try:
                _done_q.get(timeout=5.0)
            except Exception:
                pass

    def isRunning(self) -> bool:
        """main_window.py 호환용 — QThread.isRunning() 동일 인터페이스."""
        return self._running
