from __future__ import annotations
import logging
import threading

from PyQt5.QtCore import QObject, pyqtSignal

from pxie4464_daq.device.daq import _DAQBase

logger = logging.getLogger(__name__)


class AcquisitionWorker(QObject):
    """백그라운드 연속 수집 스레드.

    QThread 대신 Python 표준 threading.Thread를 사용한다.
    QThread는 Python의 threading._active에 등록되지 않아 Python 3.14에서
    logging 호출 시 threading.current_thread() → KeyError 연쇄 오류가 발생한다.

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
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="AcquisitionWorker",
            daemon=True,   # 앱 종료 시 스레드 자동 정리
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            self._daq.start()
            while self._running:
                data = self._daq.read()
                self.data_ready.emit(data)
        except Exception as exc:
            # nidaqmx 내부 오류 메시지가 %d format 실패로 str(exc)를 던질 수 있음
            # (화면보호기·절전 진입 시 장치 연결 끊김 케이스)
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
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def isRunning(self) -> bool:
        """main_window.py 호환용 — QThread.isRunning() 동일 인터페이스."""
        return self._thread is not None and self._thread.is_alive()
