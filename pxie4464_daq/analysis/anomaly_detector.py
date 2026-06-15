from __future__ import annotations
import logging
import queue as _queue
import threading
from enum import Enum, auto
from typing import List

import numpy as np
from sklearn.ensemble import IsolationForest
from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

# Python 3.14 + PyQt5 SIP 디스패치 컨텍스트에서 ThreadPoolExecutor.submit()이 실패함.
# (submit() → Future() → threading.Condition() → TypeError)
# 모듈 임포트 시점(SIP 컨텍스트 밖)에 queue.Queue와 sklearn 전용 스레드를 미리 생성.
# SIP 컨텍스트 안에서는 put()/get()만 호출하며, 새로운 threading 객체를 생성하지 않는다.
#
# [큐 안전성 설계]
# 공유 응답 큐(_SKL_RES)를 사용하면 timeout 발생 시 stale 결과가 남아
# 이후 호출이 잘못된 결과를 받는 버그가 있다.
# 해결: 모듈 임포트 시점에 _N_RQ개의 결과 큐를 미리 생성(SIP 밖 → 안전).
# 각 호출은 순환 방식으로 전용 큐를 배정받으므로 timeout이 발생해도
# stale 결과가 다음 호출에 영향을 주지 않는다.
_N_RQ = 64  # 순환 풀 크기 (동시 호출은 1개이므로 64면 충분)
_SKL_REQ: _queue.Queue = _queue.Queue()
_SKL_RESULT_POOL: list = [_queue.Queue() for _ in range(_N_RQ)]
_rq_index: int = 0


def _sklearn_loop() -> None:
    while True:
        fn, args, rq = _SKL_REQ.get()
        try:
            rq.put(('ok', fn(*args)))
        except Exception as e:
            rq.put(('err', e))


threading.Thread(target=_sklearn_loop, daemon=True, name="sklearn").start()


def _call_sklearn(fn, *args, timeout: float = 30.0):
    """sklearn 함수를 SIP 컨텍스트 밖 스레드에서 실행하고 결과를 동기적으로 반환."""
    global _rq_index
    rq = _SKL_RESULT_POOL[_rq_index % _N_RQ]
    _rq_index += 1

    # 64회 전에 timeout이 발생했을 경우 stale 결과를 제거
    while not rq.empty():
        try:
            rq.get_nowait()
        except _queue.Empty:
            break

    _SKL_REQ.put((fn, args, rq))
    try:
        status, value = rq.get(timeout=timeout)
    except _queue.Empty:
        logger.error("sklearn 호출 timeout (%.1fs) — 채널 업데이트 건너뜀", timeout)
        raise TimeoutError(f"sklearn timed out after {timeout}s")
    if status == 'err':
        raise value
    return value


ZSCORE_WARNING = 3.0
ZSCORE_ALARM = 5.0
NORM_DEV_WARNING = -2.0  # 베이스라인 score_samples 분포 대비 정규화 편차 임계값
NORM_DEV_ALARM = -3.0
WARNING_HOLDOFF = 3      # 연속 n회 이상 판정 시에만 WARNING 발동


class State(Enum):
    LEARNING = auto()
    NORMAL = auto()
    WARNING = auto()
    ALARM = auto()


class ChannelAnomalyDetector:
    """단일 채널 이상 감지기."""

    def __init__(self, baseline_count: int = 20):
        self._baseline_count = baseline_count
        self._baseline: list = []
        self._model: IsolationForest | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._score_mean: float = 0.0   # 베이스라인 score_samples 평균
        self._score_std: float = 1.0    # 베이스라인 score_samples 표준편차
        self.state: State = State.LEARNING
        self._norm_dev: float = 0.0     # 정규화 편차 (IF 이상도 지표)
        self._zscore_max: float = 0.0
        self._warning_streak: int = 0

    def update(self, features: np.ndarray) -> State:
        if self.state == State.LEARNING:
            self._baseline.append(features.copy())
            if len(self._baseline) >= self._baseline_count:
                self._fit_model()
                self.state = State.NORMAL
            return self.state

        # Z-score 계산
        zscores = np.abs((features - self._mean) / (self._std + 1e-10))
        self._zscore_max = float(np.max(zscores))

        # IsolationForest score_samples → 베이스라인 분포 대비 정규화 편차
        X = features.reshape(1, -1)
        if_raw = float(_call_sklearn(self._model.score_samples, X, timeout=5.0)[0])
        self._norm_dev = (if_raw - self._score_mean) / self._score_std

        # 상태 결정 (AND 로직: 두 조건 모두 만족해야 낮은 심각도 유지)
        if self._zscore_max < ZSCORE_WARNING and self._norm_dev > NORM_DEV_WARNING:
            new_state = State.NORMAL
        elif self._zscore_max < ZSCORE_ALARM and self._norm_dev > NORM_DEV_ALARM:
            new_state = State.WARNING
        else:
            new_state = State.ALARM

        # 홀드오프: WARNING/ALARM 3회 연속이어야 발동
        if new_state in (State.WARNING, State.ALARM):
            self._warning_streak += 1
        else:
            self._warning_streak = 0

        if self._warning_streak >= WARNING_HOLDOFF:
            self.state = new_state
        else:
            self.state = State.NORMAL

        return self.state

    @property
    def if_score(self) -> float:
        """베이스라인 대비 정규화 편차 (낮을수록 이상)."""
        return self._norm_dev

    @property
    def zscore_max(self) -> float:
        return self._zscore_max

    def _fit_model(self) -> None:
        data = np.array(self._baseline)
        self._mean = data.mean(axis=0)
        self._std = data.std(axis=0)
        self._model = IsolationForest(contamination=0.05, random_state=42)
        _call_sklearn(self._model.fit, data, timeout=30.0)
        baseline_scores = _call_sklearn(self._model.score_samples, data, timeout=10.0)
        self._score_mean = float(baseline_scores.mean())
        self._score_std = float(baseline_scores.std()) + 1e-9


class AnomalyDetector(QObject):
    """n채널 통합 이상 감지 관리자."""

    state_changed = pyqtSignal(object)  # list[State] (n채널)

    def __init__(self, n_channels: int = 4, baseline_count: int = 20, parent=None):
        super().__init__(parent)
        self._detectors = [ChannelAnomalyDetector(baseline_count) for _ in range(n_channels)]

    def update(self, features: np.ndarray) -> List[State]:
        """features: shape (n_channels, N_FEATURES)"""
        states = []
        for ch in range(len(self._detectors)):
            try:
                states.append(self._detectors[ch].update(features[ch]))
            except TimeoutError:
                # sklearn timeout — 이전 상태 유지하고 계속 진행
                states.append(self._detectors[ch].state)
            except Exception as exc:
                logger.error("CH%d 이상감지 업데이트 오류 (건너뜀): %s", ch, exc)
                states.append(self._detectors[ch].state)
        self.state_changed.emit(states)
        return states

    def if_scores(self) -> List[float]:
        return [d.if_score for d in self._detectors]

    def zscore_maxes(self) -> List[float]:
        return [d.zscore_max for d in self._detectors]
