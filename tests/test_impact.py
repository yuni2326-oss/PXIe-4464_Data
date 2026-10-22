"""임팩트 해머 시험 모듈 테스트."""
from datetime import datetime

import numpy as np
import pytest

from pxie4464_daq.analysis.impact import (
    TriggerCapture, analyze_pulse, classify, exp_window, force_bandwidths,
    force_spectrum, frf_add, frf_coherence, frf_h1, frf_new, frf_remove_last,
    hit_spectra, load_hit, overload_rows, pick_peaks,
)
from pxie4464_daq.device.impact_daq import (
    KIND_FORCE, KIND_MIC, MockImpactDAQ, clip_level, make_spec, physical_channel,
    task_channel_order,
)
from pxie4464_daq.storage.impact_saver import ImpactLog, hit_filename, save_hit


def half_sine(fs, T, peak):
    n = int(round(T * fs))
    return peak * np.sin(np.pi * np.arange(n) / n)


# ── 채널 매핑 ────────────────────────────────────────────────────────────────

def test_physical_channel_mapping():
    assert physical_channel(7, "PXI1Slot3", "PXI1Slot5") == "PXI1Slot5/ai3"   # 해머
    assert physical_channel(6, "PXI1Slot3", "PXI1Slot5") == "PXI1Slot5/ai2"   # 마이크
    assert physical_channel(2, "PXI1Slot3", "PXI1Slot5") == "PXI1Slot3/ai2"
    with pytest.raises(ValueError):
        physical_channel(12, "A", "B")


def test_task_channel_order_is_ascending_and_restores_spec_order():
    # 해머 ai3, 마이크 ai2, 가속도 ai0·ai1, 4464 ai1 (다중 장치)
    specs = [make_spec(7, KIND_FORCE, 20.348, 200.0), make_spec(6, KIND_MIC, 50.0, 130.0),
             make_spec(4, "accel", 10.2, 490.0), make_spec(5, "accel", 10.2, 490.0),
             make_spec(1, "accel", 10.2, 490.0)]
    add_order, row_of_spec = task_channel_order(specs)
    added = [specs[i]["physical"] for i in add_order]
    # 장치별로 ai 번호가 오름차순 (DAQmx -200618 조건)
    for dev in ("PXI1Slot3", "PXI1Slot5"):
        ais = [int(p.split("/ai")[1]) for p in added if p.startswith(dev)]
        assert ais == sorted(ais)
    task_rows = np.arange(len(specs))[:, None] * np.ones((1, 3))   # 태스크 행 = 추가 순서
    restored = task_rows[row_of_spec]
    for spec_i in range(len(specs)):
        assert add_order[int(restored[spec_i, 0])] == spec_i      # 0행 = 해머로 복원


def test_clip_level_mic_rms_to_peak():
    s = make_spec(6, KIND_MIC, 50.0, 130.0)
    assert clip_level(s) == pytest.approx(89.44, rel=1e-3)
    assert clip_level(make_spec(7, KIND_FORCE, 20.348, 200.0)) == 200.0


# ── 트리거 캡처 ──────────────────────────────────────────────────────────────

def _signal_with_pulses(n, positions, n_ch=3):
    sig = np.zeros((n_ch, n))
    sig[1] = np.arange(n)            # 정렬 확인용 고유값
    sig[2] = -np.arange(n) * 0.5
    for p in positions:
        sig[0, p:p + 5] = 5.0        # 임계 1.0을 p에서 처음 넘는다
    return sig


def _feed_random(cap, sig, seed=0):
    rng = np.random.default_rng(seed)
    recs, pos, n = [], 0, sig.shape[1]
    while pos < n:
        m = int(rng.integers(1, 700))
        r, _ = cap.feed(sig[:, pos:pos + m], now=datetime(2026, 10, 22, 12, 0, 0))
        recs += r
        pos += m
    return recs


def test_trigger_capture_alignment_across_chunks():
    pre, post = 50, 400
    positions = [300, 1200, 2500]
    sig = _signal_with_pulses(3500, positions)
    cap = TriggerCapture(3, 1000.0, trig_row=0, threshold=1.0, pre_s=pre / 1000, post_s=post / 1000)
    recs = _feed_random(cap, sig)
    assert len(recs) == 3
    for rec, p in zip(recs, positions):
        assert rec["trigger_index"] == pre
        assert rec["trigger_abs"] == p
        assert not rec["pre_incomplete"]
        np.testing.assert_array_equal(rec["data"], sig[:, p - pre:p - pre + pre + post])


def test_trigger_pre_incomplete_at_start():
    pre, post = 50, 100
    sig = _signal_with_pulses(400, [10])
    cap = TriggerCapture(3, 1000.0, 0, 1.0, pre / 1000, post / 1000)
    recs = _feed_random(cap, sig)
    assert len(recs) == 1
    r = recs[0]
    assert r["pre_incomplete"]
    np.testing.assert_array_equal(r["data"][:, pre:], sig[:, 10:10 + post])
    np.testing.assert_array_equal(r["data"][1, pre - 10:pre], sig[1, :10])


def test_second_pulse_inside_record_is_not_a_new_trigger():
    pre, post = 20, 300
    sig = _signal_with_pulses(1500, [100, 250, 700])   # 250은 100의 기록 구간 안
    cap = TriggerCapture(3, 1000.0, 0, 1.0, pre / 1000, post / 1000)
    recs = _feed_random(cap, sig)
    assert [r["trigger_abs"] for r in recs] == [100, 700]


# ── 펄스·대역폭 ──────────────────────────────────────────────────────────────

def test_analyze_pulse_half_sine_and_double_hit():
    fs, T = 51200.0, 0.0005
    n = int(0.5 * fs)
    f = np.zeros(n)
    p = half_sine(fs, T, 20.0)
    s0 = 1000
    f[s0:s0 + p.size] = p
    trig = s0 + int(np.argmax(f[s0:] >= 2.0))
    info = analyze_pulse(f, fs, trig, threshold=2.0)
    assert info["peak_abs"] == pytest.approx(20.0, rel=0.01)
    assert info["width_s"] == pytest.approx(0.968 * T, abs=2 / fs)
    assert not info["double_hit"]

    g = f.copy()
    b0 = s0 + int(0.002 * fs)
    g[b0:b0 + p.size] += 0.3 * p                       # 2 ms 뒤 30 % 바운스
    info2 = analyze_pulse(g, fs, trig, threshold=2.0)
    assert info2["double_hit"]
    assert info2["second_peak"] == pytest.approx(6.0, rel=0.02)


def test_force_bandwidth_matches_half_sine_theory():
    fs, T = 204800.0, 0.0003
    n = int(0.2 * fs)
    f = np.zeros(n)
    p = half_sine(fs, T, 10.0)
    f[2000:2000 + p.size] = p
    info = analyze_pulse(f, fs, 2000, threshold=1.0)
    freqs, mag = force_spectrum(f, fs, info["start"], info["end"])
    bw = force_bandwidths(freqs, mag)
    # 해석해: |F(f)|/|F(0)| = |cos(π f T) / (1 − (2 f T)²)|
    grid = np.linspace(1.0, 1.4 / T, 200000)
    with np.errstate(divide="ignore", invalid="ignore"):
        ana = np.abs(np.cos(np.pi * grid * T) / (1 - (2 * grid * T) ** 2))
    f10 = grid[np.argmax(ana < 10 ** (-10 / 20))]
    assert bw[10.0] == pytest.approx(f10, rel=0.03)
    assert bw[5.0] < bw[10.0] < bw[20.0]


def test_exp_window_end_value():
    w = exp_window(1000, 100, 0.01)
    assert np.all(w[:101] == 1.0)
    assert w[-1] == pytest.approx(0.01, rel=1e-9)


# ── FRF ──────────────────────────────────────────────────────────────────────

def _sdof_record(fs, pre_s, post_s, fn, zeta, peak):
    pre, post = int(pre_s * fs), int(post_s * fs)
    n = pre + post
    t = np.arange(n) / fs
    wn = 2 * np.pi * fn
    wd = wn * np.sqrt(1 - zeta ** 2)
    h = np.exp(-zeta * wn * t) * np.sin(wd * t) / wd
    force = np.zeros(n)
    p = half_sine(fs, 0.0002, peak)
    force[pre:pre + p.size] = p
    m = 1 << int(np.ceil(np.log2(2 * n)))
    resp = np.fft.irfft(np.fft.rfft(force, m) * np.fft.rfft(h, m), m)[:n] / fs
    return np.vstack([force, resp]), pre


def test_frf_sdof_peak_coherence_and_remove_last():
    fs, fn, zeta = 51200.0, 1000.0, 0.001
    rec, trig = _sdof_record(fs, 0.02, 2.0, fn, zeta, 10.0)
    pulse = analyze_pulse(rec[0], fs, trig, 1.0)
    sp = hit_spectra(rec, fs, 0, [1], pulse, trig, fmax=5000.0, exp_end=None)
    st = frf_new(sp["freqs"].size, 1)
    for k in range(3):
        frf_add(st, sp["F"], sp["X"], hit_id=f"h{k}")
    H = frf_h1(st)[0]
    f_pk = sp["freqs"][np.argmax(np.abs(H))]
    assert abs(f_pk - fn) < 1.0
    coh = frf_coherence(st)[0]
    assert coh[np.argmax(np.abs(H))] == pytest.approx(1.0, abs=1e-6)
    peaks = pick_peaks(sp["freqs"], H, fmin=200, fmax=5000)
    assert any(abs(p["f"] - fn) < 1.0 for p in peaks)

    sff_before = st["Sff"].copy()
    assert frf_remove_last(st)
    assert st["n"] == 2
    assert np.all(st["Sff"] < sff_before + 1e-30)
    np.testing.assert_allclose(np.abs(frf_h1(st)[0]), np.abs(H), rtol=1e-9)
    assert not frf_remove_last(st)          # 직전 1회만 뺄 수 있다


def test_pick_peaks_resolves_close_doublet():
    df = 0.25
    f = np.arange(0, 8000, df)
    H = np.full(f.size, 1e-4, dtype=complex)
    for fc in (3000.0, 3002.0, 5000.0):
        H += 1.0 / (1 + 1j * (f - fc) / 0.2)
    peaks = pick_peaks(f, H, fmin=200, fmax=7000, min_prom_db=6)
    got = [round(p["f"], 2) for p in peaks]
    assert 3000.0 in got and 3002.0 in got and 5000.0 in got


def test_overload_and_classify():
    rec = np.zeros((2, 100))
    rec[0, 10] = 199.0
    rec[1, 20] = 10.0
    assert overload_rows(rec, [200.0, 89.4]) == [0]
    pulse = {"double_hit": False}
    assert classify(pulse, [0], False) == "overload"
    assert classify({"double_hit": True}, [], False) == "double_hit"
    assert classify(pulse, [], True) == "pre_incomplete"
    assert classify(pulse, [], False) == "ok"


# ── 저장 ─────────────────────────────────────────────────────────────────────

def test_save_and_load_hit_roundtrip(tmp_path):
    data = np.random.default_rng(1).normal(size=(3, 1000))
    meta = {
        "sample_rate": 204800.0, "session": 2, "location": "back_vane3_xi0.3",
        "channel_names": ["CH7_hammer", "CH6_mic", "CH4_acc"],
        "temperature_C": [23.4, 23.6], "trigger_time": datetime(2026, 10, 22, 9, 5, 7, 123000),
        "double_hit": False, "notes": None,
    }
    name = hit_filename(datetime(2026, 10, 22, 9, 5, 7, 123000), 2, "back_vane3_xi0.3", "M1", 7)
    assert name == "20261022_090507_123_S2_back_vane3_xi0.3_M1_H007.npz"
    path = save_hit(tmp_path / "impact_test", name, data, meta)
    assert path.exists() and not list((tmp_path / "impact_test").glob("*.part"))
    d = load_hit(path)
    assert d["data"].dtype == np.float32
    np.testing.assert_allclose(d["data"], data.astype(np.float32))
    assert d["session"] == 2 and d["location"] == "back_vane3_xi0.3"
    assert list(d["channel_names"]) == ["CH7_hammer", "CH6_mic", "CH4_acc"]
    np.testing.assert_allclose(d["temperature_C"], [23.4, 23.6])
    assert d["trigger_time"] == "2026-10-22T09:05:07.123"
    assert d["double_hit"] is False and d["notes"] == ""


def test_impact_log_status_and_numbering(tmp_path):
    p = tmp_path / "impact_log.csv"
    log = ImpactLog(p)
    for k in range(1, 4):
        log.append({"file": f"f{k}.npz", "session": 1, "location": "inlet_rim",
                    "mic_position": "M1", "hit": k, "peak_force_N": 12.5,
                    "double_hit": False, "quality": "ok", "status": "accepted"})
    log.append({"file": "g1.npz", "session": 1, "location": "back_vane1_xi0.3",
                "mic_position": "M1", "hit": 1, "status": "flagged"})
    assert log.next_hit_number(1, "inlet_rim", "M1") == 4
    assert log.next_hit_number(2, "inlet_rim", "M1") == 1
    assert log.set_status("f2.npz", "rejected")
    assert log.count(1, "inlet_rim", "M1", "accepted") == 2

    again = ImpactLog(p)                      # 재시작해도 이어진다
    assert len(again.rows) == 4
    assert again.count(1, "inlet_rim", "M1", "rejected") == 1
    assert again.next_hit_number(1, "inlet_rim", "M1") == 4
    assert again.rows[0]["double_hit"] == "0"


# ── 모의 장치 + 캡처 연동 ────────────────────────────────────────────────────

def test_mock_daq_hits_are_captured_and_show_first_mode():
    fs = 51200.0
    specs = [make_spec(7, KIND_FORCE, 20.348, 200.0, label="hammer"),
             make_spec(6, KIND_MIC, 50.0, 130.0, label="mic")]
    daq = MockImpactDAQ(specs, sample_rate=fs, hit_interval_s=1.2, first_hit_s=0.3,
                        double_hit_prob=0.0, seed=3, realtime=False)
    cap = TriggerCapture(2, fs, 0, threshold=2.0, pre_s=0.02, post_s=0.8)
    daq.start()
    recs = []
    for _ in range(int(3.2 * fs / daq._chunk)):
        r, _ = cap.feed(daq.read())
        recs += r
    daq.stop()
    assert len(recs) >= 2
    rec = recs[0]
    pulse = analyze_pulse(rec["data"][0], fs, rec["trigger_index"], 2.0)
    assert 10.0 < pulse["peak_abs"] < 30.0 and not pulse["double_hit"]
    sp = hit_spectra(rec["data"], fs, 0, [1], pulse, rec["trigger_index"], fmax=8000.0)
    st = frf_new(sp["freqs"].size, 1)
    frf_add(st, sp["F"], sp["X"])
    peaks = pick_peaks(sp["freqs"], frf_h1(st)[0], fmin=1000, fmax=8000)
    assert any(abs(p["f"] - 2270.0) < 3.0 for p in peaks)
