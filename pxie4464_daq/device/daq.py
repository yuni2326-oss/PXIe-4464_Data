from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)

try:
    import nidaqmx
    from nidaqmx.constants import (
        AcquisitionType,
        ExcitationSource,
        AccelSensitivityUnits,
        AccelUnits,
        SoundPressureUnits,
        Edge,
    )
    from nidaqmx.stream_readers import AnalogMultiChannelReader
    _NIDAQMX_AVAILABLE = True
except ImportError:
    _NIDAQMX_AVAILABLE = False
    logger.warning("nidaqmx not available; hardware DAQ cannot be used")

N_CHANNELS_4464 = 4
N_CHANNELS_4492 = 8

G_TO_MS2 = 9.80665            # 1 g = 9.80665 m/s²
DEFAULT_SENSITIVITY = 10.2   # mV/(m/s²) — PCB 352C33 기본값

# 두 장비의 출력을 동일한 물리 단위(m/s²)로 통일한다:
#  - PXIe-4464: 가속도 채널이 감도로 나눠 m/s²를 반환
#  - PXIe-4492: raw 전압을 동일 감도로 나눠 m/s²로 변환
# 감도 s[mV/(m/s²)]에 대해  가속도[m/s²] = 전압[V] × 1000 / s


class _DAQBase(ABC):
    """공통 DAQ 인터페이스."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        try:
            self.stop()
        except Exception as exc:
            logger.warning("stop() during context exit raised: %s", exc)

    @abstractmethod
    def configure(self, sample_rate: float, record_length: int, voltage_range: float = 10.0,
                  sensitivity: float = DEFAULT_SENSITIVITY) -> None: ...

    @abstractmethod
    def read(self) -> np.ndarray: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...


class MockDAQ(_DAQBase):
    """하드웨어 없이 테스트용 가상 DAQ (채널 수 가변)."""

    def __init__(self, n_channels: int = 4):
        self._n_channels = n_channels
        self._sample_rate: float = 51200.0
        self._record_length: int = 1024
        self._rng = np.random.default_rng()
        self._running = False

    def configure(self, sample_rate: float, record_length: int, voltage_range: float = 10.0,
                  sensitivity: float = DEFAULT_SENSITIVITY) -> None:
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
        data = np.zeros((self._n_channels, self._record_length), dtype=np.float64)
        for ch in range(self._n_channels):
            freq = 100 * (ch + 1)  # ch0=100Hz, ch1=200Hz, ...
            amplitude = 1.0
            noise_std = amplitude / (10 ** (40 / 20))  # SNR ~40 dB
            data[ch] = amplitude * np.sin(2 * np.pi * freq * t) + self._rng.normal(0, noise_std, self._record_length)
        return data


class PXIe4464(_DAQBase):
    """NI PXIe-4464 4채널 IEPE 가속도계 드라이버."""

    def __init__(self, device_name: str = "PXI1Slot3",
                 sensitivity: float = DEFAULT_SENSITIVITY,
                 excit_current: float = 0.004):
        if not _NIDAQMX_AVAILABLE:
            raise RuntimeError("nidaqmx is not installed")
        self._device_name = device_name
        self._sensitivity = sensitivity       # mV/(m/s²)
        self._excit_current = excit_current   # A (0.002 or 0.004)
        self._sample_rate: float = 51200.0
        self._record_length: int = 1024
        self._voltage_range: float = 10.0
        self._task: Optional[nidaqmx.Task] = None
        self._reader: Optional[AnalogMultiChannelReader] = None
        self._buffer: Optional[np.ndarray] = None

    def configure(self, sample_rate: float, record_length: int, voltage_range: float = 10.0,
                  sensitivity: float = DEFAULT_SENSITIVITY) -> None:
        self._sample_rate = float(sample_rate)
        self._record_length = int(record_length)
        self._voltage_range = float(voltage_range)
        self._sensitivity = float(sensitivity)

    def start(self) -> None:
        self._task = nidaqmx.Task()
        ch = f"{self._device_name}/ai0:{N_CHANNELS_4464 - 1}"
        # 감도 mV/(m/s²) → DAQmx가 요구하는 mV/g 로 변환. 출력 단위는 m/s².
        sens_mv_per_g = self._sensitivity * G_TO_MS2
        # 입력 전압 레인지(±voltage_range V)에 해당하는 m/s² 범위로 min/max 설정
        # (레인지도 m/s² 단위로 해석되므로 전압 레인지를 그대로 넘기면 안 됨)
        sens_v_per_ms2 = self._sensitivity / 1000.0
        max_eng = self._voltage_range / sens_v_per_ms2 if sens_v_per_ms2 > 0 else self._voltage_range
        self._task.ai_channels.add_ai_accel_chan(
            physical_channel=ch,
            sensitivity=sens_mv_per_g,
            sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G,
            units=AccelUnits.METERS_PER_SECOND_SQUARED,
            current_excit_source=ExcitationSource.INTERNAL,
            current_excit_val=self._excit_current,
            min_val=-max_eng,
            max_val=max_eng,
        )
        self._task.timing.cfg_samp_clk_timing(
            rate=self._sample_rate,
            source="OnboardClock",
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self._record_length * 50,
        )
        self._buffer = np.zeros((N_CHANNELS_4464, self._record_length), dtype=np.float64)
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
            self._buffer, number_of_samples_per_channel=self._record_length, timeout=10.0
        )
        # 화면보호기·절전 진입 시 nidaqmx가 None을 반환하는 경우 대비
        if samps_read is None:
            raise RuntimeError("PXIe4464: read_many_sample returned None — 장치 연결이 끊어졌을 수 있습니다.")
        if samps_read < self._record_length:
            logger.warning("Partial read: %d/%d samples. Retrying.", samps_read, self._record_length)
            samps_read = self._reader.read_many_sample(
                self._buffer, number_of_samples_per_channel=self._record_length, timeout=10.0
            )
            if samps_read is None or samps_read < self._record_length:
                raise RuntimeError(f"Partial read after retry: {samps_read}/{self._record_length}")
        return self._buffer.copy()


class PXIe4492(_DAQBase):
    """NI PXIe-4492 8채널 전압 입력 드라이버."""

    def __init__(self, device_name: str = "PXI1Slot5", voltage_range: float = 10.0,
                 sensitivity: float = DEFAULT_SENSITIVITY, convert_to_accel: bool = True,
                 mic_channels=None, mic_sensitivity: float = 50.0,
                 mic_max_spl: float = 130.0, excit_current: float = 0.004):
        if not _NIDAQMX_AVAILABLE:
            raise RuntimeError("nidaqmx is not installed")
        self._device_name = device_name
        self._sample_rate: float = 51200.0
        self._record_length: int = 1024
        self._voltage_range: float = voltage_range
        self._sensitivity = sensitivity       # 가속도계 감도 mV/(m/s²)
        self._convert = convert_to_accel      # (호환 유지용; 드라이버가 물리단위 반환하므로 미사용)
        # 마이크로폰 채널: 4492 로컬 ai 인덱스 집합(0~7). 해당 채널은 마이크(Pa), 그 외 가속도계(m/s²).
        self._mic_locals = set(int(c) for c in (mic_channels or []))
        self._mic_sensitivity = float(mic_sensitivity)  # mV/Pa
        # 마이크 최대 SPL(dB). 감도와 조합해 입력 레인지를 정한다. 너무 높으면 장비가
        # 지원하는 레인지를 초과해 오류(-200860). 130dB≈63Pa≈3.16V(±5V IEPE 레인지 내).
        self._mic_max_spl = float(mic_max_spl)
        self._excit_current = float(excit_current)      # IEPE 전류 (A), 4464와 동일 기본 4mA
        self._task: Optional[nidaqmx.Task] = None
        self._reader: Optional[AnalogMultiChannelReader] = None
        self._buffer: Optional[np.ndarray] = None

    def configure(self, sample_rate: float, record_length: int, voltage_range: float = 10.0,
                  sensitivity: float = DEFAULT_SENSITIVITY) -> None:
        self._sample_rate = float(sample_rate)
        self._record_length = int(record_length)
        self._voltage_range = float(voltage_range)
        self._sensitivity = float(sensitivity)

    def start(self) -> None:
        self._task = nidaqmx.Task()
        # 4464와 동일하게 내부 IEPE로 구동. 채널별로 가속도계/마이크 타입을 지정하면
        # 드라이버가 IEPE 전류 공급 + 물리단위(m/s², Pa) 변환까지 수행한다.
        sens_mv_per_g = self._sensitivity * G_TO_MS2           # mV/(m/s²) → mV/g
        sens_v_per_ms2 = self._sensitivity / 1000.0
        max_eng = self._voltage_range / sens_v_per_ms2 if sens_v_per_ms2 > 0 else self._voltage_range
        for ai in range(N_CHANNELS_4492):
            chan = f"{self._device_name}/ai{ai}"
            if ai in self._mic_locals:
                # 마이크로폰(IEPE, Pa). 최대 SPL은 UI 지정값 사용(감도와 조합해 레인지 결정).
                self._task.ai_channels.add_ai_microphone_chan(
                    physical_channel=chan,
                    units=SoundPressureUnits.PA,
                    mic_sensitivity=self._mic_sensitivity,
                    max_snd_press_level=self._mic_max_spl,
                    current_excit_source=ExcitationSource.INTERNAL,
                    current_excit_val=self._excit_current,
                )
            else:
                # 가속도계(IEPE, m/s²) — 4464와 동일 방식
                self._task.ai_channels.add_ai_accel_chan(
                    physical_channel=chan,
                    sensitivity=sens_mv_per_g,
                    sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G,
                    units=AccelUnits.METERS_PER_SECOND_SQUARED,
                    current_excit_source=ExcitationSource.INTERNAL,
                    current_excit_val=self._excit_current,
                    min_val=-max_eng,
                    max_val=max_eng,
                )
        self._task.timing.cfg_samp_clk_timing(
            rate=self._sample_rate,
            source="OnboardClock",
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self._record_length * 50,
        )
        self._buffer = np.zeros((N_CHANNELS_4492, self._record_length), dtype=np.float64)
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
            self._buffer, number_of_samples_per_channel=self._record_length, timeout=10.0
        )
        if samps_read is None:
            raise RuntimeError("PXIe4492: read_many_sample returned None — 장치 연결이 끊어졌을 수 있습니다.")
        if samps_read < self._record_length:
            logger.warning("Partial read: %d/%d samples. Retrying.", samps_read, self._record_length)
            samps_read = self._reader.read_many_sample(
                self._buffer, number_of_samples_per_channel=self._record_length, timeout=10.0
            )
            if samps_read is None or samps_read < self._record_length:
                raise RuntimeError(f"Partial read after retry: {samps_read}/{self._record_length}")
        # 드라이버가 이미 물리단위(m/s² / Pa)로 반환 — 추가 스케일링 없음
        return self._buffer.copy()


class MultiDAQ(_DAQBase):
    """복수 DAQ 장치를 결합 — read() 시 채널 축 방향으로 합침."""

    def __init__(self, *daqs: _DAQBase):
        self._daqs: List[_DAQBase] = list(daqs)

    def configure(self, sample_rate: float, record_length: int, voltage_range: float = 10.0,
                  sensitivity: float = DEFAULT_SENSITIVITY) -> None:
        for daq in self._daqs:
            daq.configure(sample_rate, record_length, voltage_range, sensitivity)

    def start(self) -> None:
        for daq in self._daqs:
            daq.start()

    def stop(self) -> None:
        for daq in self._daqs:
            daq.stop()

    def read(self) -> np.ndarray:
        return np.vstack([daq.read() for daq in self._daqs])
