"""임팩트 해머 수집 워커.

AcquisitionWorker와 같은 사전 생성 실행자 스레드(worker.py)에서 돈다.
트리거 판정·레코드 조립을 수집 스레드에서 하므로 GUI가 그리기·저장으로
바빠도 타격을 놓치지 않는다. GUI에는 완성된 레코드와 레벨 표시만 보낸다.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Sequence

import numpy as np
from PyQt5.QtCore import pyqtSignal

from pxie4464_daq.acquisition.worker import AcquisitionWorker
from pxie4464_daq.analysis.impact import TriggerCapture

logger = logging.getLogger(__name__)

LEVEL_PERIOD_S = 0.2     # 레벨 표시 갱신 주기


class ImpactWorker(AcquisitionWorker):
    """Signals:
        hit_captured(dict): TriggerCapture 레코드 (data, trigger_index, trigger_time, …)
        triggered(object):  트리거 직후 시각(datetime) — "기록 중" 표시용
        level(dict):        {"peaks": [행별 최대 |x|], "capturing": bool, "progress_s": float}
    """

    hit_captured = pyqtSignal(object)
    triggered = pyqtSignal(object)
    level = pyqtSignal(object)

    def __init__(self, daq, capture: TriggerCapture, level_rows: Sequence[int], parent=None):
        super().__init__(daq, parent)
        self._capture = capture
        self._level_rows = list(level_rows)

    def _run(self) -> None:
        try:
            self._daq.start()
            # 장치가 샘플레이트를 보정했으면 레코드 메타데이터가 실제 값을 쓰도록 맞춘다
            fs = getattr(self._daq, "sample_rate", None)
            if fs:
                self._capture.fs = float(fs)
            peaks = np.zeros(len(self._level_rows))
            t_last = time.monotonic()
            while self._running:
                chunk = self._daq.read()
                recs, trig = self._capture.feed(chunk, now=datetime.now())
                if trig:
                    self.triggered.emit(datetime.now())
                for rec in recs:
                    self.hit_captured.emit(rec)
                if self._level_rows:
                    peaks = np.maximum(peaks, np.max(np.abs(chunk[self._level_rows]), axis=1))
                now = time.monotonic()
                if now - t_last >= LEVEL_PERIOD_S:
                    self.level.emit({
                        "peaks": peaks.tolist(),
                        "capturing": self._capture.capturing,
                        "progress_s": self._capture.progress_s,
                    })
                    peaks = np.zeros(len(self._level_rows))
                    t_last = now
        except Exception as exc:
            self._running = False
            try:
                msg = str(exc)
            except Exception:
                msg = f"{type(exc).__name__}: (오류 메시지 변환 실패)"
            try:
                logger.error("ImpactWorker error: %s", msg)
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
