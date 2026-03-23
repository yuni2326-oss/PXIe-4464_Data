from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

N_CHANNELS = 4
CHANNEL_FREQS = [100, 200, 300, 400]  # MockDAQ 채널별 주파수 (Hz)


class _DAQBase(ABC):
    """PXIe4464 / MockDAQ 공통 인터페이스."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        try:
            self.stop()
        except Exception as exc:
            logger.warning("stop() during context exit raised: %s", exc)

    @abstractmethod
    def configure(self, sample_rate: float, record_length: int, voltage_range: float = 10.0) -> None: ...

    @abstractmethod
    def read(self) -> np.ndarray: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...


class MockDAQ(_DAQBase):
    """하드웨어 없이 테스트용 가상 DAQ."""

    def __init__(self):
        self._sample_rate: float = 51200.0
        self._record_length: int = 1024
        self._rng = np.random.default_rng()
        self._running = False

    def configure(self, sample_rate: float, record_length: int, voltage_range: float = 10.0) -> None:
        self._sample_rate = float(sample_rate)
        self._record_length = int(record_length)

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def read(self) -> np.ndarray:
        t = np.arange(self._record_length) / self._sample_rate
        data = np.zeros((N_CHANNELS, self._record_length), dtype=np.float64)
        for ch, freq in enumerate(CHANNEL_FREQS):
            amplitude = 1.0  # g
            noise_std = amplitude / (10 ** (40 / 20))  # SNR ~40 dB
            data[ch] = amplitude * np.sin(2 * np.pi * freq * t) + self._rng.normal(0, noise_std, self._record_length)
        return data
