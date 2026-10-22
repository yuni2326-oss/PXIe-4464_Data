"""임팩트 해머 시험용 DAQ.

해머(IEPE 힘)·마이크·가속도계를 **하나의 DAQmx 태스크**로 묶어 같은 표본화 클록으로
읽는다. 해머와 응답이 샘플 단위로 맞아야 FRF·위상이 의미가 있기 때문이다.
(4492 채널만 쓰면 단일 장치 태스크, CH0–3(4464)을 함께 고르면 다중 장치 태스크 —
PXIe DSA 모듈은 같은 섀시에서 DAQmx가 자동 동기화한다.)

채널 번호는 상시 모니터링 앱과 같은 전역 번호를 쓴다:
    CH0–3  = PXIe-4464 ai0–3
    CH4–11 = PXIe-4492 ai0–7   (예: CH6 = 4492 ai2 마이크, CH7 = 4492 ai3 해머)

채널 사양은 직렬화(설정 저장·메타데이터)가 쉽도록 dict로 다룬다.
"""
from __future__ import annotations

import logging
import math
import time
from typing import List, Optional

import numpy as np

from pxie4464_daq.device.daq import _DAQBase, _NIDAQMX_AVAILABLE, G_TO_MS2, N_CHANNELS_4464

logger = logging.getLogger(__name__)

if _NIDAQMX_AVAILABLE:
    import nidaqmx
    from nidaqmx.constants import (
        AcquisitionType,
        ExcitationSource,
        AccelSensitivityUnits,
        AccelUnits,
        SoundPressureUnits,
        ForceUnits,
        ForceIEPESensorSensitivityUnits,
        Edge,
    )
    from nidaqmx.stream_readers import AnalogMultiChannelReader

KIND_FORCE = "force"
KIND_MIC = "mic"
KIND_ACCEL = "accel"
UNITS = {KIND_FORCE: "N", KIND_MIC: "Pa", KIND_ACCEL: "m/s^2"}
SENS_UNITS = {KIND_FORCE: "mV/N", KIND_MIC: "mV/Pa", KIND_ACCEL: "mV/(m/s^2)"}


def default_chunk(sample_rate: float) -> int:
    """읽기 단위: 약 50 ms (204.8 kS/s → 10240). 짧을수록 반응이 빠르고 신호 수가 늘어난다."""
    return max(512, int(round(sample_rate * 0.05)))


def physical_channel(global_ch: int, dev4464: str, dev4492: str) -> str:
    """전역 CH 번호 → 물리 채널명."""
    global_ch = int(global_ch)
    if global_ch < 0 or global_ch >= N_CHANNELS_4464 + 8:
        raise ValueError(f"채널 번호 범위 밖: CH{global_ch} (0–11)")
    if global_ch < N_CHANNELS_4464:
        return f"{dev4464}/ai{global_ch}"
    return f"{dev4492}/ai{global_ch - N_CHANNELS_4464}"


def make_spec(global_ch: int, kind: str, sensitivity: float, max_val: float,
              dev4464: str = "PXI1Slot3", dev4492: str = "PXI1Slot5",
              iepe: bool = True, excit_current: float = 0.004, label: str = "") -> dict:
    """채널 사양(dict).

    sensitivity: force mV/N, mic mV/Pa, accel mV/(m/s²)
    max_val:     force N(피크), mic 최대 SPL dB, accel m/s²(피크)
    """
    if kind not in UNITS:
        raise ValueError(f"알 수 없는 채널 종류: {kind}")
    return {
        "ch": int(global_ch),
        "name": f"CH{int(global_ch)}_{label or kind}",
        "physical": physical_channel(global_ch, dev4464, dev4492),
        "kind": kind,
        "unit": UNITS[kind],
        "sensitivity": float(sensitivity),
        "sens_unit": SENS_UNITS[kind],
        "max_val": float(max_val),
        "iepe": bool(iepe),
        "excit_current": float(excit_current),
    }


def task_channel_order(specs: List[dict]) -> tuple:
    """DAQmx 태스크에 채널을 넣을 순서와, 읽은 행을 사양 순서로 되돌리는 인덱스.

    DSA 장치는 한 태스크 안에서 물리 채널을 오름차순(ai0→ai1→…)으로만 추가할 수 있다
    (DAQmx -200618). 그래서 (장치, ai번호) 순으로 추가하고, read()에서
    data_task[row_of_spec] 로 사양 순서(해머 = 0행)를 복원한다.
    """
    def key(i):
        dev, ai = specs[i]["physical"].split("/")
        return (dev, int(ai[2:]))
    add_order = sorted(range(len(specs)), key=key)
    row_of_spec = np.argsort(add_order)
    return add_order, row_of_spec


def clip_level(spec: dict) -> float:
    """설정상 최대 입력(물리단위, 피크) — 과부하 판정 기준.

    마이크의 최대 SPL은 RMS 기준이라 피크 = √2 × p_rms.
    (130 dB → 63.2 Pa rms → 89.4 Pa 피크)
    """
    if spec["kind"] == KIND_MIC:
        return 20e-6 * 10 ** (spec["max_val"] / 20.0) * math.sqrt(2.0)
    return float(spec["max_val"])


class ImpactDAQ(_DAQBase):
    """여러 종류 채널을 한 태스크로 연속 표본화. read() → (n_ch, chunk) 물리단위."""

    def __init__(self, specs: List[dict], sample_rate: float = 204800.0,
                 chunk: Optional[int] = None):
        if not _NIDAQMX_AVAILABLE:
            raise RuntimeError("nidaqmx is not installed")
        if not specs:
            raise ValueError("채널이 없습니다.")
        phys = [s["physical"] for s in specs]
        if len(set(phys)) != len(phys):
            raise ValueError(f"같은 물리 채널이 중복되었습니다: {phys}")
        self._specs = list(specs)
        self._add_order, self._row_of_spec = task_channel_order(self._specs)
        self._fs = float(sample_rate)
        self._chunk = int(chunk or default_chunk(self._fs))
        self._task = None
        self._reader = None
        self._buffer: Optional[np.ndarray] = None

    @property
    def specs(self) -> List[dict]:
        return list(self._specs)

    @property
    def sample_rate(self) -> float:
        return self._fs

    @property
    def n_channels(self) -> int:
        return len(self._specs)

    def configure(self, sample_rate: float, record_length: int, voltage_range: float = 10.0,
                  sensitivity: float = 0.0) -> None:
        self._fs = float(sample_rate)
        self._chunk = int(record_length)

    def _add_channel(self, task, s: dict) -> None:
        excit_src = ExcitationSource.INTERNAL if s["iepe"] else ExcitationSource.NONE
        excit = s["excit_current"]
        if s["kind"] == KIND_FORCE:
            task.ai_channels.add_ai_force_iepe_chan(
                physical_channel=s["physical"],
                min_val=-s["max_val"], max_val=s["max_val"],
                units=ForceUnits.NEWTONS,
                sensitivity=s["sensitivity"],
                sensitivity_units=ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON,
                current_excit_source=excit_src, current_excit_val=excit,
            )
        elif s["kind"] == KIND_MIC:
            task.ai_channels.add_ai_microphone_chan(
                physical_channel=s["physical"],
                units=SoundPressureUnits.PA,
                mic_sensitivity=s["sensitivity"],
                max_snd_press_level=s["max_val"],
                current_excit_source=excit_src, current_excit_val=excit,
            )
        else:  # accel
            task.ai_channels.add_ai_accel_chan(
                physical_channel=s["physical"],
                min_val=-s["max_val"], max_val=s["max_val"],
                units=AccelUnits.METERS_PER_SECOND_SQUARED,
                sensitivity=s["sensitivity"] * G_TO_MS2,          # mV/(m/s²) → mV/g
                sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G,
                current_excit_source=excit_src, current_excit_val=excit,
            )

    def start(self) -> None:
        task = nidaqmx.Task()
        try:
            for i in self._add_order:          # 물리 채널 오름차순 (DAQmx -200618 방지)
                self._add_channel(task, self._specs[i])
            # source 미지정: 다중 장치 태스크면 DAQmx가 공통 클록·동기를 잡는다
            task.timing.cfg_samp_clk_timing(
                rate=self._fs,
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self._chunk * 50,
            )
            try:
                actual = float(task.timing.samp_clk_rate)
                if abs(actual - self._fs) > 1e-6 * self._fs:
                    logger.warning("샘플레이트 보정: 요청 %.1f → 실제 %.1f S/s", self._fs, actual)
                    self._fs = actual
            except Exception:
                pass
            self._buffer = np.zeros((len(self._specs), self._chunk), dtype=np.float64)
            self._reader = AnalogMultiChannelReader(task.in_stream)
            task.start()
            self._task = task
            devs = sorted({s["physical"].split("/")[0] for s in self._specs})
            logger.info("ImpactDAQ 시작: %s, %.0f S/s, 청크 %d, 장치 %s",
                        [s["name"] for s in self._specs], self._fs, self._chunk, devs)
        except Exception:
            try:
                task.close()
            except Exception:
                pass
            raise

    def stop(self) -> None:
        if self._task is not None:
            try:
                self._task.stop()
                self._task.close()
            except Exception as exc:
                logger.warning("ImpactDAQ cleanup error (ignored): %s", exc)
            finally:
                self._task = None
                self._reader = None
                self._buffer = None

    def read(self) -> np.ndarray:
        if self._reader is None or self._buffer is None:
            raise RuntimeError("ImpactDAQ not started. Call start() first.")
        n = self._reader.read_many_sample(
            self._buffer, number_of_samples_per_channel=self._chunk, timeout=10.0
        )
        if n is None or n < self._chunk:
            raise RuntimeError(f"ImpactDAQ: 부분 읽기 {n}/{self._chunk}")
        # 태스크 행(물리 채널 순) → 사양 순서(해머 = 0행). 팬시 인덱싱이라 복사본이 된다.
        return self._buffer[self._row_of_spec]


class MockImpactDAQ(_DAQBase):
    """하드웨어 없이 UI·저장·분석 경로를 확인하기 위한 모의 장치.

    주기적으로 해머 타격(반사인 펄스)을 넣고, 응답 채널에는 임펠러 계획서의
    낮은 모드(2.27·3.30·3.54·4.55·5.30 kHz, 감쇠비 0.05–0.09 %)를 합성한다.
    가끔 이중 타격(2 ms 뒤 30 %)도 넣어 품질 플래그를 확인할 수 있게 한다.
    """

    MODES = ((2270.0, 0.0008, 1.0), (3300.0, 0.0006, 0.6), (3540.0, 0.0005, 0.8),
             (4550.0, 0.0007, 0.5), (5300.0, 0.0009, 0.4))   # (f Hz, ζ, Pa @20 N)

    def __init__(self, specs: List[dict], sample_rate: float = 51200.0,
                 chunk: Optional[int] = None, hit_interval_s: float = 6.0,
                 peak_force: float = 20.0, pulse_s: float = 0.0003,
                 double_hit_prob: float = 0.15, seed: Optional[int] = None,
                 realtime: bool = True, first_hit_s: float = 1.5):
        if not specs:
            raise ValueError("채널이 없습니다.")
        self._specs = list(specs)
        self._fs = float(sample_rate)
        self._chunk = int(chunk or default_chunk(self._fs))
        self._interval = float(hit_interval_s)
        self._peak = float(peak_force)
        self._pulse = float(pulse_s)
        self._p_double = float(double_hit_prob)
        self._rng = np.random.default_rng(seed)
        self._realtime = bool(realtime)
        self._first_hit = float(first_hit_s)
        self._running = False

    @property
    def specs(self) -> List[dict]:
        return list(self._specs)

    @property
    def sample_rate(self) -> float:
        return self._fs

    @property
    def n_channels(self) -> int:
        return len(self._specs)

    def configure(self, sample_rate: float, record_length: int, voltage_range: float = 10.0,
                  sensitivity: float = 0.0) -> None:
        self._fs = float(sample_rate)
        self._chunk = int(record_length)

    def start(self) -> None:
        self._n = 0
        self._hits: list = []          # [(t_hit, amplitude)]
        self._next_hit = self._first_hit
        self._t0 = time.monotonic()
        self._running = True

    def stop(self) -> None:
        self._running = False

    def _schedule(self, t_end: float) -> None:
        while self._next_hit < t_end:
            amp = self._peak * (0.7 + 0.6 * self._rng.random())
            self._hits.append((self._next_hit, amp))
            if self._rng.random() < self._p_double:
                self._hits.append((self._next_hit + 0.002, 0.3 * amp))
            self._next_hit += self._interval * (0.8 + 0.4 * self._rng.random())

    def read(self) -> np.ndarray:
        if not self._running:
            raise RuntimeError("MockImpactDAQ not started. Call start() first.")
        n = self._chunk
        t = (self._n + np.arange(n)) / self._fs
        t_end = t[-1] + 1.0 / self._fs
        self._schedule(t_end)
        self._hits = [h for h in self._hits if t[0] - h[0] < 8.0]

        out = np.empty((len(self._specs), n))
        resp = np.zeros(n)
        force = np.zeros(n)
        for th, amp in self._hits:
            tau = t - th
            on = (tau >= 0.0) & (tau < self._pulse)
            force[on] += amp * np.sin(np.pi * tau[on] / self._pulse)
            after = tau >= 0.0
            if after.any():
                ta = tau[after]
                scale = amp / 20.0
                acc = np.zeros(ta.size)
                for f, z, g in self.MODES:
                    wn = 2.0 * np.pi * f
                    wd = wn * np.sqrt(1.0 - z * z)
                    acc += g * scale * np.exp(-z * wn * ta) * np.sin(wd * ta)
                resp[after] += acc
        for i, s in enumerate(self._specs):
            if s["kind"] == KIND_FORCE:
                out[i] = force + self._rng.normal(0.0, 0.005, n)
            elif s["kind"] == KIND_MIC:
                out[i] = resp + self._rng.normal(0.0, 0.002, n)
            else:
                out[i] = 0.05 * resp + self._rng.normal(0.0, 0.01, n)
        self._n += n

        if self._realtime:
            due = self._t0 + self._n / self._fs
            lag = due - time.monotonic()
            if lag > 0:
                time.sleep(lag)
        return out
