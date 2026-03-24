from __future__ import annotations
import logging
from collections import deque
from datetime import datetime

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from pxie4464_daq.analysis.fft import compute_fft
from pxie4464_daq.analysis.features import extract_features, N_FEATURES

logger = logging.getLogger(__name__)

class FeatureCollector(QObject):
    """주기적으로 n채널 FFT 특징을 추출하여 emit.

    Signals:
        features_ready(object): shape (n_channels, N_FEATURES) numpy 배열
        raw_ready(object, object): (datetime, np.ndarray shape (n_channels, N)) — 윈도우 원시 데이터
    """

    features_ready = pyqtSignal(object)
    raw_ready = pyqtSignal(object, object)  # (datetime, np.ndarray)

    def __init__(self, sample_rate: float, collection_cycle_sec: float = 30.0,
                 window_sec: float = 5.0, n_channels: int = 4, parent=None):
        super().__init__(parent)
        self._sample_rate = sample_rate
        self._n_channels = n_channels
        self._window_samples = int(sample_rate * window_sec)
        # 채널별 rolling buffer (deque로 자동 truncation)
        self._buffers = [deque(maxlen=self._window_samples) for _ in range(n_channels)]
        self._timer = QTimer(self)
        self._timer.setInterval(int(collection_cycle_sec * 1000))
        self._timer.timeout.connect(self._extract_and_emit)

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def on_data_ready(self, data: np.ndarray) -> None:
        """AcquisitionWorker.data_ready 시그널 슬롯. data: (n_channels, N)"""
        for ch in range(self._n_channels):
            self._buffers[ch].extend(data[ch].tolist())

    def _extract_and_emit(self) -> None:
        ts = datetime.now()
        features_all = np.zeros((self._n_channels, N_FEATURES), dtype=np.float64)
        raw_arrays = []
        for ch in range(self._n_channels):
            if len(self._buffers[ch]) < 2:
                logger.warning("CH%d: 버퍼 부족 (%d 샘플)", ch, len(self._buffers[ch]))
                raw_arrays.append(np.zeros(0))
                continue
            chunk = np.array(self._buffers[ch], dtype=np.float64)
            raw_arrays.append(chunk)
            freqs, mags = compute_fft(chunk, self._sample_rate)
            features_all[ch] = extract_features(freqs, mags, raw=chunk)

        # 원시 윈도우 데이터 emit (채널 길이가 다를 수 있으므로 최소 길이로 맞춤)
        min_len = min((len(a) for a in raw_arrays), default=0)
        if min_len > 0:
            raw_data = np.array([a[:min_len] for a in raw_arrays], dtype=np.float64)
            self.raw_ready.emit(ts, raw_data)

        self.features_ready.emit(features_all)
