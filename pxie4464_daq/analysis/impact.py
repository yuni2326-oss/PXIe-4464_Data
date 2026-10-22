"""임팩트 해머 시험 분석 (numpy 전용).

- TriggerCapture  : 연속 청크에서 해머 힘 트리거 → [프리트리거 + 포스트트리거] 레코드
- analyze_pulse   : 피크 힘, 펄스 폭, 이중 타격(바운스) 판정
- force_bandwidths: 힘 스펙트럼이 저주파 수준보다 N dB 떨어지는 주파수(가진 대역)
- FRF 누적        : H1 = Σ F*·X / Σ|F|², 코히어런스 γ² = |Σ F*·X|² / (Σ|F|² Σ|X|²)
- pick_peaks      : |H1| 피크(주파수·크기·위상) — 마이크 고정 + 해머 이동 시 베인별
                    위상(부호) 패턴으로 원주 차수 h를 가리는 데 쓴다
- load_hit        : 저장된 타격 파일(.npz) 읽기

Python 3.14 + PyQt5 환경에서 슬롯 안 사용자 클래스 생성이 드물게 실패한 이력이 있어,
GUI 스레드에서 타격마다 호출되는 경로는 dict와 numpy만 쓴다.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── 트리거 캡처 ─────────────────────────────────────────────────────────────

class TriggerCapture:
    """해머 힘 |f| ≥ threshold 인 첫 샘플을 트리거로 잡아 레코드를 만든다.

    레코드 = 트리거 이전 pre_s + 트리거 이후 post_s (트리거 샘플은 index = pre).
    청크 경계와 무관하게 동작하며, 프리트리거는 이전 청크에서 가져온다.
    기록 중(post_s 동안) 다시 넘는 힘은 새 트리거가 아니라 같은 레코드 안에
    들어가므로 analyze_pulse가 이중 타격으로 표시한다.
    """

    def __init__(self, n_channels: int, sample_rate: float, trig_row: int,
                 threshold: float, pre_s: float = 0.02, post_s: float = 4.0):
        self.n_ch = int(n_channels)
        self.fs = float(sample_rate)
        self.trig_row = int(trig_row)
        self.threshold = float(threshold)
        self.pre = max(1, int(round(pre_s * self.fs)))
        self.post = max(1, int(round(post_s * self.fs)))
        self.total = self.pre + self.post
        self.reset()

    def reset(self) -> None:
        self._hist = np.zeros((self.n_ch, self.pre))
        self._hist_valid = 0
        self._rec: Optional[np.ndarray] = None
        self._filled = 0
        self._abs = 0                 # 다음 청크 첫 샘플의 절대 인덱스
        self._trig_abs = 0
        self._trig_time: Optional[datetime] = None
        self._pre_incomplete = False

    @property
    def capturing(self) -> bool:
        return self._rec is not None

    @property
    def progress_s(self) -> float:
        """기록 중인 레코드의 트리거 이후 경과(초)."""
        if self._rec is None:
            return 0.0
        return max(0, self._filled - self.pre) / self.fs

    def _push_hist(self, seg: np.ndarray) -> None:
        m = seg.shape[1]
        if m == 0:
            return
        if m >= self.pre:
            self._hist = seg[:, -self.pre:].copy()
        else:
            self._hist = np.concatenate([self._hist[:, m:], seg], axis=1)
        self._hist_valid = min(self.pre, self._hist_valid + m)

    def feed(self, chunk: np.ndarray, now: Optional[datetime] = None) -> Tuple[List[dict], bool]:
        """청크 (n_ch, n) 투입 → (완성된 레코드 목록, 이번 청크에서 트리거 발생 여부).

        now: 이 청크를 읽은 시각(트리거 시각 추정용). 생략 시 datetime.now().
        """
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[0] != self.n_ch:
            raise ValueError(f"청크 모양 {chunk.shape} ≠ ({self.n_ch}, n)")
        n = chunk.shape[1]
        now = now or datetime.now()
        out: List[dict] = []
        triggered = False
        pos = 0
        while pos < n:
            if self._rec is None:
                f = chunk[self.trig_row, pos:]
                over = np.flatnonzero(np.abs(f) >= self.threshold)
                if over.size == 0:
                    self._push_hist(chunk[:, pos:])
                    pos = n
                    break
                t = pos + int(over[0])
                self._push_hist(chunk[:, pos:t])
                rec = np.empty((self.n_ch, self.total))
                rec[:, :self.pre] = self._hist
                self._pre_incomplete = self._hist_valid < self.pre
                self._rec = rec
                self._filled = self.pre
                self._trig_abs = self._abs + t
                # 청크 끝 = now 로 보고 트리거 샘플 시각을 역산
                self._trig_time = now - timedelta(seconds=(n - t) / self.fs)
                triggered = True
                pos = t
            else:
                take = min(self.total - self._filled, n - pos)
                self._rec[:, self._filled:self._filled + take] = chunk[:, pos:pos + take]
                self._filled += take
                pos += take
                if self._filled == self.total:
                    out.append({
                        "data": self._rec,
                        "sample_rate": self.fs,
                        "trigger_index": self.pre,
                        "trigger_abs": int(self._trig_abs),
                        "trigger_time": self._trig_time,
                        "pre_incomplete": bool(self._pre_incomplete),
                        "threshold": self.threshold,
                    })
                    # 다음 트리거의 프리트리거 = 방금 레코드의 마지막 pre 샘플(연속)
                    self._hist = self._rec[:, -self.pre:].copy()
                    self._hist_valid = self.pre
                    self._rec = None
                    self._filled = 0
        self._abs += n
        return out, triggered


# ── 펄스·대역폭 ─────────────────────────────────────────────────────────────

def analyze_pulse(force: np.ndarray, fs: float, trig_index: int, threshold: float,
                  edge_frac: float = 0.05, search_s: float = 0.005,
                  guard_s: float = 0.0002, double_ratio: float = 0.1) -> dict:
    """주 펄스(피크·시작·끝·폭)와 이중 타격.

    시작/끝 = 피크의 edge_frac(5 %) 이하로 떨어지는 지점.
    이중 타격 = 펄스 끝 + guard 이후 레코드 안에서 max(threshold, double_ratio×피크)
    이상의 힘이 다시 나타나면 True.
    """
    f = np.asarray(force, dtype=np.float64)
    n = f.size
    a = np.abs(f)
    i0 = int(min(max(trig_index, 0), n - 1))
    i1 = min(n, i0 + max(1, int(round(search_s * fs))))
    pk_i = i0 + int(np.argmax(a[i0:i1]))
    pk = float(f[pk_i])
    apk = abs(pk)
    lev = edge_frac * apk
    before = np.flatnonzero(a[:pk_i] <= lev)
    start = int(before[-1] + 1) if before.size else 0
    after = np.flatnonzero(a[pk_i:] <= lev)
    end = int(pk_i + after[0]) if after.size else n
    g = min(n, end + int(round(guard_s * fs)))
    if g < n:
        j = g + int(np.argmax(a[g:]))
        sp = float(a[j])
        sp_t = (j - trig_index) / fs
    else:
        sp, sp_t = 0.0, float("nan")
    return {
        "peak": pk,
        "peak_abs": apk,
        "peak_index": int(pk_i),
        "start": start,
        "end": end,
        "width_s": (end - start) / fs,
        "second_peak": sp,
        "second_peak_t": float(sp_t),
        "double_hit": bool(sp >= max(threshold, double_ratio * apk)),
    }


def force_window(n: int, start: int, end: int, fs: float,
                 margin_s: float = 0.0001, taper_s: float = 0.0002) -> np.ndarray:
    """펄스 구간만 1인 힘 창(앞뒤 여유 + 코사인 테이퍼). 펄스 밖 잡음을 지운다."""
    w = np.zeros(n)
    m = int(round(margin_s * fs))
    s = max(0, start - m)
    e = min(n, end + m)
    w[s:e] = 1.0
    nt = int(round(taper_s * fs))
    if nt > 0:
        k = np.arange(max(0, s - nt), s)
        w[k] = 0.5 * (1.0 - np.cos(np.pi * (k - (s - nt) + 1) / (nt + 1)))
        k = np.arange(e, min(n, e + nt))
        w[k] = 0.5 * (1.0 + np.cos(np.pi * (k - e + 1) / (nt + 1)))
    return w


def exp_window(n: int, start: int, end_value: float = 0.01) -> np.ndarray:
    """응답용 지수창: start부터 감쇠해 레코드 끝에서 end_value.

    추가 감쇠율 σ = 1/τ [1/s] → 모드 감쇠비에 σ/ω 가 더해진다(라이브 확인용).
    """
    w = np.ones(n)
    m = n - int(start)
    if end_value >= 1.0 or m <= 1:
        return w
    tau = (m - 1) / np.log(1.0 / end_value)
    w[int(start):] = np.exp(-np.arange(m) / tau)
    return w


def force_spectrum(force: np.ndarray, fs: float, start: int, end: int,
                   min_df: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """힘 창을 씌운 펄스 구간만 짧게 FFT → (freqs, |F|). 대역폭·표시용."""
    f = np.asarray(force, dtype=np.float64)
    n = f.size
    w = force_window(n, start, end, fs)
    nz = np.flatnonzero(w > 0)
    if nz.size == 0:
        return np.array([0.0]), np.array([0.0])
    seg = f[nz[0]:nz[-1] + 1] * w[nz[0]:nz[-1] + 1]
    nfft = 1 << int(np.ceil(np.log2(max(seg.size, fs / min_df))))
    mag = np.abs(np.fft.rfft(seg, n=nfft))
    return np.fft.rfftfreq(nfft, 1.0 / fs), mag


def force_bandwidths(freqs: np.ndarray, mag: np.ndarray,
                     drops_db: Sequence[float] = (5.0, 10.0, 20.0),
                     ref_band: Tuple[float, float] = (50.0, 500.0)) -> Dict[float, float]:
    """|F|가 저주파 기준(ref_band 중앙값)보다 drop dB 떨어지는 첫 주파수.

    끝까지 안 떨어지면 나이퀴스트(=freqs[-1])."""
    sel = (freqs >= ref_band[0]) & (freqs <= ref_band[1])
    ref = float(np.median(mag[sel])) if sel.any() else float(np.max(mag))
    out: Dict[float, float] = {}
    above = freqs > ref_band[1]
    for d in drops_db:
        thr = ref * 10 ** (-d / 20.0)
        idx = np.flatnonzero(above & (mag < thr))
        out[float(d)] = float(freqs[idx[0]]) if idx.size else float(freqs[-1])
    return out


# ── FRF ─────────────────────────────────────────────────────────────────────

def hit_spectra(record: np.ndarray, fs: float, force_row: int, resp_rows: Sequence[int],
                pulse: dict, trig_index: int, fmax: float,
                exp_end: Optional[float] = 0.01) -> dict:
    """한 타격의 힘·응답 스펙트럼 (fmax 이하만).

    힘: 힘 창, 응답: 지수창(exp_end=None이면 창 없음). 같은 변환을 쓰므로 X/F 비는
    FFT 스케일과 무관하다.
    """
    n = record.shape[1]
    fw = force_window(n, pulse["start"], pulse["end"], fs)
    rw = exp_window(n, trig_index, exp_end) if exp_end else np.ones(n)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    keep = freqs <= fmax
    F = np.fft.rfft(record[force_row] * fw)[keep]
    X = np.empty((len(resp_rows), int(keep.sum())), dtype=np.complex128)
    for k, r in enumerate(resp_rows):
        X[k] = np.fft.rfft(record[r] * rw)[keep]
    return {"freqs": freqs[keep], "F": F, "X": X}


def frf_new(n_bins: int, n_resp: int) -> dict:
    return {
        "n": 0,
        "Sff": np.zeros(n_bins),
        "Sfx": np.zeros((n_resp, n_bins), dtype=np.complex128),
        "Sxx": np.zeros((n_resp, n_bins)),
        "last": None,
        "last_id": None,
    }


def frf_add(state: dict, F: np.ndarray, X: np.ndarray, hit_id: str = "") -> None:
    sff = np.abs(F) ** 2
    sfx = np.conj(F)[None, :] * X
    sxx = np.abs(X) ** 2
    state["Sff"] += sff
    state["Sfx"] += sfx
    state["Sxx"] += sxx
    state["n"] += 1
    state["last"] = (sff, sfx, sxx)
    state["last_id"] = hit_id


def frf_remove_last(state: dict) -> bool:
    """마지막으로 더한 타격을 평균에서 뺀다(직전 타격 제외용)."""
    if state.get("last") is None:
        return False
    sff, sfx, sxx = state["last"]
    state["Sff"] -= sff
    state["Sfx"] -= sfx
    state["Sxx"] -= sxx
    state["n"] -= 1
    state["last"] = None
    state["last_id"] = None
    return True


def frf_h1(state: dict) -> np.ndarray:
    tiny = np.finfo(float).tiny
    return state["Sfx"] / np.maximum(state["Sff"], tiny)[None, :]


def frf_coherence(state: dict) -> np.ndarray:
    tiny = np.finfo(float).tiny
    num = np.abs(state["Sfx"]) ** 2
    den = np.maximum(state["Sff"][None, :] * state["Sxx"], tiny)
    return np.clip(num / den, 0.0, 1.0)


def db20(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def pick_peaks(freqs: np.ndarray, H: np.ndarray, fmin: float = 200.0,
               fmax: float = 20000.0, n_peaks: int = 12, min_prom_db: float = 10.0,
               win_hz: float = 40.0, min_sep_hz: float = 0.5) -> List[dict]:
    """|H| 피크 → [{f, db, phase_deg}] (주파수 오름차순).

    돌출도 = 피크 높이 − 주변 ±win_hz 최솟값(dB). 가까운 doublet도 min_sep_hz
    이상 떨어지면 둘 다 잡는다.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    H = np.asarray(H)
    sel = (freqs >= fmin) & (freqs <= fmax)
    f = freqs[sel]
    h = H[sel]
    if f.size < 3:
        return []
    y = db20(h)
    df = f[1] - f[0]
    w = max(1, int(round(win_hz / df)))
    cand = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1
    if cand.size == 0:
        return []
    pad = np.pad(y, w, mode="edge")
    mins = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1).min(axis=1)
    prom = y[cand] - mins[cand]
    cand = cand[prom >= min_prom_db]
    order = cand[np.argsort(y[cand])[::-1]]
    chosen: List[int] = []
    for i in order:
        if all(abs(f[i] - f[j]) >= min_sep_hz for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= n_peaks:
            break
    chosen.sort()
    return [{"f": float(f[i]), "db": float(y[i]),
             "phase_deg": float(np.degrees(np.angle(h[i])))} for i in chosen]


# ── 판정·파일 ───────────────────────────────────────────────────────────────

def overload_rows(record: np.ndarray, clip_levels: Sequence[float], frac: float = 0.98) -> List[int]:
    """설정 최대값의 frac 이상에 닿은 채널 행 번호."""
    rows = []
    for i, c in enumerate(clip_levels):
        if c and c > 0 and float(np.max(np.abs(record[i]))) >= frac * c:
            rows.append(i)
    return rows


def classify(pulse: dict, overloads: Sequence[int], pre_incomplete: bool) -> str:
    if overloads:
        return "overload"
    if pulse["double_hit"]:
        return "double_hit"
    if pre_incomplete:
        return "pre_incomplete"
    return "ok"


def load_hit(path) -> dict:
    """저장된 타격(.npz) → dict.

    0차원 배열은 파이썬 스칼라/문자열로, 1차원 문자열 배열(채널명·단위 등)은
    파이썬 str 리스트로 바꾼다. data·수치 배열은 numpy 그대로.
        d = load_hit(p); force = d["data"][0]; fs = d["sample_rate"]
    """
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    for k, v in list(out.items()):
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                out[k] = v.item()
            elif v.ndim == 1 and v.dtype.kind == "U":
                out[k] = [str(x) for x in v.tolist()]
    return out
