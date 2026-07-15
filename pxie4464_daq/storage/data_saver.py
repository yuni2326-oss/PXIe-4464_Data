from __future__ import annotations

import csv
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, pyqtSlot

from pxie4464_daq.analysis.fft import compute_fft

_log = logging.getLogger(__name__)

_DISK_WARN_GB = 5.0        # 여유 공간이 이 값 미만이면 경고
FFT_SAVE_MAX_HZ = 5000.0   # FFT 저장 상한 주파수 (표시와 동일, 쓰기시간 단축)


class DataSaver(QObject):
    """주기적으로 n채널 raw 파형(.npz)과 FFT 스펙트럼(0–5kHz CSV)을 자동 저장.

    FeatureCollector.raw_ready 시그널에 연결하여 사용.
    save_interval_sec 마다 한 번만 실제 파일을 기록한다 (0이면 매 사이클 저장).

    raw는 float32 압축 없는 .npz(data, sample_rate, unit)로 저장 — CSV 대비
    쓰기 ~180배 빠르고 파일 ~1/6. 코스트다운 등 연속 저장에 적합.
    FFT는 0–5kHz만 CSV로 저장하여 쓰기시간을 추가로 단축.

    .npz 읽기 예:
        d = numpy.load("..._raw.npz"); arr = d["data"]; sr = float(d["sample_rate"])

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
        self._save_count: int = 0

    @pyqtSlot(object, object)
    def on_raw(self, timestamp: datetime, data: np.ndarray) -> None:
        """FeatureCollector.raw_ready 슬롯. data: shape (n, N)"""
        if self._last_save_time is not None:
            elapsed = (timestamp - self._last_save_time).total_seconds()
            if elapsed < self._save_interval_sec:
                return  # 아직 저장 주기 미도달

        # 디스크 여유 공간 확인
        free_gb: float = float("inf")
        try:
            free_gb = shutil.disk_usage(self._save_dir).free / (1024 ** 3)
            if free_gb < _DISK_WARN_GB:
                _log.warning("[저장경고] 디스크 여유 공간 부족: %.1f GB (임계값 %.0f GB)",
                             free_gb, _DISK_WARN_GB)
        except Exception as exc:
            _log.warning("[저장경고] 디스크 용량 확인 실패: %s", exc)

        n_ch = data.shape[0]
        stem = timestamp.strftime("%Y%m%d_%H%M%S")
        t_start = time.monotonic()
        try:
            self._write_raw(stem, data)
            self._write_fft(stem, data)
            elapsed_w = time.monotonic() - t_start

            # 저장된 파일 크기 합산
            fnames = [f"{stem}_raw.npz"] + [f"{stem}_fft_ch{ch}.csv" for ch in range(n_ch)]
            size_kb = sum(
                (self._save_dir / f).stat().st_size
                for f in fnames
                if (self._save_dir / f).exists()
            ) / 1024

            self._save_count += 1
            self._last_save_time = timestamp
            _log.info(
                "[저장완료 #%d] %s | %d채널 | %.1f KB | 쓰기 %.2f초 | 디스크여유 %.1f GB",
                self._save_count, stem, n_ch, size_kb, elapsed_w, free_gb,
            )
        except OSError as exc:
            elapsed_f = time.monotonic() - t_start
            _log.error(
                "[저장실패 #%d] %s | %.2f초 경과 | 오류: %s",
                self._save_count + 1, stem, elapsed_f, exc,
            )

    def _write_raw(self, stem: str, data: np.ndarray) -> None:
        """n채널 원시 데이터를 .npz(float32)로 저장 — 시간축은 sample_rate로 복원.

        time_s[i] = i / sample_rate.  단위는 m/s².
        """
        path = self._save_dir / f"{stem}_raw.npz"
        np.savez(
            path,
            data=data.astype(np.float32),
            sample_rate=np.float64(self._sample_rate),
            unit="m/s^2",
            timestamp=stem,
        )

    def _write_fft(self, stem: str, data: np.ndarray) -> None:
        """n채널 FFT 스펙트럼(0–5kHz)을 채널별 CSV로 저장."""
        for ch in range(data.shape[0]):
            freqs, mags = compute_fft(data[ch], self._sample_rate)
            mask = freqs <= FFT_SAVE_MAX_HZ
            freqs, mags = freqs[mask], mags[mask]
            path = self._save_dir / f"{stem}_fft_ch{ch}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["frequency_hz", "magnitude"])
                writer.writerows(
                    (f"{freq:.4f}", f"{mag:.8f}") for freq, mag in zip(freqs, mags)
                )
