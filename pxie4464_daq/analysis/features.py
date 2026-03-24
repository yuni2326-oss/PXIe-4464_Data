from __future__ import annotations
import numpy as np
from typing import Optional

HARMONIC_EXCLUSION_BINS = 5  # 주도 주파수 ±5 bin 제외 (노이즈 플로어 계산용)
HIGH_FREQ_LO = 1000.0        # 고주파 에너지 비율 하한 (Hz) — 캐비테이션 지표
HIGH_FREQ_HI = 3000.0        # 고주파 에너지 비율 상한 (Hz)
N_FEATURES = 11              # 총 특징 수


def extract_features(frequencies: np.ndarray, magnitudes: np.ndarray,
                     raw: Optional[np.ndarray] = None) -> np.ndarray:
    """FFT 스펙트럼 + 시간 도메인에서 11개 특징 추출.

    Args:
        frequencies: FFT 주파수 배열 (Hz)
        magnitudes:  FFT 진폭 배열 (g)
        raw:         시간 도메인 원시 데이터 (g), 선택적

    Returns:
        np.ndarray shape (11,):
            ── FFT 기반 ──────────────────────────────────────
            [0]  dominant_freq_hz       주도 주파수
            [1]  dominant_magnitude     주도 주파수 진폭
            [2]  second_harmonic        2차 고조파 진폭
            [3]  third_harmonic         3차 고조파 진폭
            [4]  thd                    총 고조파 왜곡 (H2+H3)/H1
            [5]  noise_floor_rms        노이즈 플로어 RMS
            [6]  spectral_centroid_hz   스펙트럼 센트로이드
            [7]  high_freq_energy_ratio 1~3kHz 에너지 비율 (캐비테이션 지표)
            ── 시간 도메인 (raw 제공 시) ──────────────────────
            [8]  rms                    실효값 (전반적 진동 크기)
            [9]  kurtosis               첨도 (충격성 지표, 베어링 결함 감지)
            [10] crest_factor           파고율 = peak / rms (임펄스 크기)
    """
    # ── FFT 기반 특징 ────────────────────────────────────────────────────────

    # DC 제거 후 주도 주파수
    search_mags = magnitudes.copy()
    search_mags[0] = 0.0
    dom_idx = int(np.argmax(search_mags))
    dom_freq = float(frequencies[dom_idx])
    dom_mag  = float(magnitudes[dom_idx])

    def _harmonic_mag(n: int) -> float:
        target = dom_freq * n
        if target > frequencies[-1]:
            return 0.0
        return float(magnitudes[int(np.argmin(np.abs(frequencies - target)))])

    h2  = _harmonic_mag(2)
    h3  = _harmonic_mag(3)
    thd = (h2 + h3) / dom_mag if dom_mag > 0 else 0.0

    # 노이즈 플로어 RMS
    mask = np.ones(len(magnitudes), dtype=bool)
    lo, hi = max(0, dom_idx - HARMONIC_EXCLUSION_BINS), min(len(magnitudes), dom_idx + HARMONIC_EXCLUSION_BINS + 1)
    mask[lo:hi] = False
    noise_rms = float(np.sqrt(np.mean(magnitudes[mask] ** 2))) if mask.any() else 0.0

    # 스펙트럼 센트로이드
    total_mag = float(np.sum(magnitudes))
    centroid  = float(np.dot(frequencies, magnitudes) / total_mag) if total_mag > 0 else 0.0

    # 고주파 에너지 비율 (1~3kHz 구간 — 캐비테이션 시 급증)
    hi_mask      = (frequencies >= HIGH_FREQ_LO) & (frequencies <= HIGH_FREQ_HI)
    total_energy = float(np.sum(magnitudes ** 2))
    hi_ratio     = float(np.sum(magnitudes[hi_mask] ** 2)) / (total_energy + 1e-10) if hi_mask.any() else 0.0

    # ── 시간 도메인 특징 ─────────────────────────────────────────────────────
    if raw is not None and len(raw) > 1:
        rms = float(np.sqrt(np.mean(raw ** 2)))

        mu    = float(np.mean(raw))
        sigma = float(np.std(raw))
        kurtosis = float(np.mean(((raw - mu) / (sigma + 1e-10)) ** 4)) if sigma > 0 else 0.0

        peak         = float(np.max(np.abs(raw)))
        crest_factor = peak / (rms + 1e-10)
    else:
        rms = kurtosis = crest_factor = 0.0

    return np.array(
        [dom_freq, dom_mag, h2, h3, thd, noise_rms, centroid,
         hi_ratio, rms, kurtosis, crest_factor],
        dtype=np.float64,
    )
