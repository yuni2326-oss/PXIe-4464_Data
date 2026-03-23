from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import nidaqmx
    from nidaqmx.constants import (
        AcquisitionType,
        ExcitationSource,
        AccelSensitivityUnits,
        Edge,
    )
    from nidaqmx.stream_readers import AnalogMultiChannelReader
    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False
    logger.warning("nidaqmx not available; PXIe4464 cannot be used")

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
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if record_length <= 0:
            raise ValueError(f"record_length must be positive, got {record_length}")
        self._sample_rate = float(sample_rate)
        self._record_length = int(record_length)

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def read(self) -> np.ndarray:
        if not self._running:
            raise RuntimeError("DAQ is not running. Call start() first.")
        t = np.arange(self._record_length) / self._sample_rate
        data = np.zeros((N_CHANNELS, self._record_length), dtype=np.float64)
        for ch, freq in enumerate(CHANNEL_FREQS):
            amplitude = 1.0  # g
            noise_std = amplitude / (10 ** (40 / 20))  # SNR ~40 dB
            data[ch] = amplitude * np.sin(2 * np.pi * freq * t) + self._rng.normal(0, noise_std, self._record_length)
        return data


class PXIe4464(_DAQBase):
    """NI PXIe-4464 실제 하드웨어 드라이버."""

    def __init__(self, device_name: str = "Dev1", sensitivity: float = 100.0,
                 excit_current: float = 0.004):
        if not _NIDAQMX_AVAILABLE:
            raise RuntimeError("nidaqmx is not installed")
        self._device_name = device_name
        self._sensitivity = sensitivity          # mV/g
        self._excit_current = excit_current      # A (PXIe-4464: 0.002 or 0.004)
        self._sample_rate: float = 51200.0
        self._record_length: int = 1024
        self._voltage_range: float = 10.0
        self._task: Optional[nidaqmx.Task] = None
        self._reader: Optional[AnalogMultiChannelReader] = None
        self._buffer: Optional[np.ndarray] = None

    def configure(self, sample_rate: float, record_length: int, voltage_range: float = 10.0) -> None:
        self._sample_rate = float(sample_rate)
        self._record_length = int(record_length)
        self._voltage_range = float(voltage_range)

    def start(self) -> None:
        self._task = nidaqmx.Task()
        ch = f"{self._device_name}/ai0:3"
        self._task.ai_channels.add_ai_accel_chan(
            physical_channel=ch,
            sensitivity=self._sensitivity,
            sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G,
            current_excit_source=ExcitationSource.INTERNAL,
            current_excit_val=self._excit_current,
            min_val=-self._voltage_range,
            max_val=self._voltage_range,
        )
        self._task.timing.cfg_samp_clk_timing(
            rate=self._sample_rate,
            source="OnboardClock",
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self._record_length * 10,
        )
        self._buffer = np.zeros((N_CHANNELS, self._record_length), dtype=np.float64)
        self._reader = AnalogMultiChannelReader(self._task.in_stream)
        self._task.start()

    def stop(self) -> None:
        if self._task is not None:
            try:
                self._task.stop()
                self._task.close()
            except Exception as exc:
                logger.warning("Task cleanup error (ignored): %s", exc)
            finally:
                self._task = None
                self._reader = None
                self._buffer = None

    def read(self) -> np.ndarray:
        if self._reader is None or self._buffer is None:
            raise RuntimeError("DAQ not started. Call start() first.")
        samps_read = self._reader.read_many_sample(
            self._buffer, number_of_samples_per_channel=self._record_length
        )
        if samps_read < self._record_length:
            logger.warning("Partial read: %d/%d samples. Retrying.", samps_read, self._record_length)
            samps_read = self._reader.read_many_sample(
                self._buffer, number_of_samples_per_channel=self._record_length
            )
            if samps_read < self._record_length:
                raise RuntimeError(f"Partial read after retry: {samps_read}/{self._record_length}")
        return self._buffer.copy()
