import numpy as np
import pytest
from pxie4464_daq.analysis.fft import compute_fft
from pxie4464_daq.analysis.features import extract_features, N_FEATURES

SR = 51200.0
N = 4096


def make_sine(freq: float, amplitude: float = 1.0) -> tuple:
    t = np.arange(N) / SR
    data = amplitude * np.sin(2 * np.pi * freq * t)
    return compute_fft(data, SR), data


def test_extract_features_returns_N_features():
    (freqs, mags), raw = make_sine(100.0)
    features = extract_features(freqs, mags, raw=raw)
    assert features.shape == (N_FEATURES,)
    assert features.dtype == np.float64


def test_extract_features_without_raw():
    (freqs, mags), _ = make_sine(100.0)
    features = extract_features(freqs, mags)
    assert features.shape == (N_FEATURES,)
    # raw 없으면 시간 도메인 특징은 0
    assert features[8] == 0.0   # rms
    assert features[9] == 0.0   # kurtosis
    assert features[10] == 0.0  # crest_factor


def test_dominant_frequency():
    (freqs, mags), _ = make_sine(200.0)
    features = extract_features(freqs, mags)
    assert abs(features[0] - 200.0) < 5.0


def test_dominant_magnitude():
    (freqs, mags), _ = make_sine(100.0, amplitude=2.0)
    features = extract_features(freqs, mags)
    assert abs(features[1] - 2.0) < 0.05


def test_thd_pure_sine_is_low():
    """순수 사인파의 THD는 매우 낮아야 함"""
    (freqs, mags), _ = make_sine(100.0)
    features = extract_features(freqs, mags)
    assert features[4] < 0.05


def test_spectral_centroid_near_dominant():
    (freqs, mags), _ = make_sine(300.0)
    features = extract_features(freqs, mags)
    assert abs(features[6] - 300.0) < 50.0


def test_rms_matches_amplitude():
    """순수 사인파 RMS = amplitude / sqrt(2)"""
    amplitude = 2.0
    (freqs, mags), raw = make_sine(100.0, amplitude=amplitude)
    features = extract_features(freqs, mags, raw=raw)
    expected_rms = amplitude / np.sqrt(2)
    assert abs(features[8] - expected_rms) < 0.05


def test_kurtosis_sine_near_15():
    """순수 사인파 첨도 ≈ 1.5 (정규 분포보다 낮은 임펄스성)"""
    (freqs, mags), raw = make_sine(100.0)
    features = extract_features(freqs, mags, raw=raw)
    assert 1.0 < features[9] < 2.0


def test_crest_factor_sine():
    """순수 사인파 파고율 = sqrt(2) ≈ 1.414"""
    (freqs, mags), raw = make_sine(100.0, amplitude=1.0)
    features = extract_features(freqs, mags, raw=raw)
    assert abs(features[10] - np.sqrt(2)) < 0.1


def test_high_freq_energy_ratio_low_freq_signal():
    """저주파 신호(100Hz)의 고주파 에너지 비율은 낮아야 함"""
    (freqs, mags), _ = make_sine(100.0)
    features = extract_features(freqs, mags)
    assert features[7] < 0.01
