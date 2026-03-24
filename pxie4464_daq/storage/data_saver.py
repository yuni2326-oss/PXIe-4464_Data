from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, pyqtSlot

from pxie4464_daq.analysis.fft import compute_fft

_log = logging.getLogger(__name__)

class DataSaver(QObject):
    """주기적으로 n채널 raw 파형과 FFT 스펙트럼을 CSV로 자동 저장.

    FeatureCollector.raw_ready 시그널에 연결하여 사용.
    save_interval_sec 마다 한 번만 실제 파일을 기록한다 (기본 30분).

    Slots:
        on_raw(datetime, np.ndarray): (timestamp, shape (n, N)) 수신 시 주기 판단 후 저장
    """

    def __init__(self, sample_rate: float, save_dir: str | Path = "results",
                 save_interval_sec: float = 1800.0, parent=None):
        super().__init__(parent)
        self._sample_rate = sample_rate
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._save_interval_sec = save_interval_sec
        self._last_save_time: Optional[datetime] = None

    @pyqtSlot(object, object)
    def on_raw(self, timestamp: datetime, data: np.ndarray) -> None:
        """FeatureCollector.raw_ready 슬롯. data: shape (n, N)"""
        if self._last_save_time is not None:
            elapsed = (timestamp - self._last_save_time).total_seconds()
            if elapsed < self._save_interval_sec:
                return  # 아직 저장 주기 미도달

        stem = timestamp.strftime("%Y%m%d_%H%M%S")
        try:
            self._write_raw(stem, data)
            self._write_fft(stem, data)
            self._last_save_time = timestamp
        except OSError as exc:
            _log.warning("DataSaver: CSV 저장 실패: %s", exc)

    def _write_raw(self, stem: str, data: np.ndarray) -> None:
        """n채널 원시 데이터를 하나의 CSV에 저장 (컬럼: time_s, ch0..chN-1)."""
        n_ch = data.shape[0]
        path = self._save_dir / f"{stem}_raw.csv"
        n = data.shape[1]
        time_arr = np.arange(n) / self._sample_rate
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s"] + [f"ch{ch}" for ch in range(n_ch)])
            for i, t in enumerate(time_arr):
                writer.writerow([f"{t:.8f}"] + [f"{data[ch, i]:.8f}" for ch in range(n_ch)])
        _log.info("raw 저장: %s (%d ch, %d 샘플, %.1f s)", path.name, n_ch, n, n / self._sample_rate)

    def _write_fft(self, stem: str, data: np.ndarray) -> None:
        """n채널 FFT 스펙트럼을 채널별 CSV로 저장."""
        for ch in range(data.shape[0]):
            freqs, mags = compute_fft(data[ch], self._sample_rate)
            path = self._save_dir / f"{stem}_fft_ch{ch}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["frequency_hz", "magnitude"])
                writer.writerows(
                    (f"{freq:.4f}", f"{mag:.8f}") for freq, mag in zip(freqs, mags)
                )
