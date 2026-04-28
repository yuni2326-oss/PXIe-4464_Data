from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from typing import List

import numpy as np
from sklearn.ensemble import IsolationForest
from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

# Python 3.14 + PyQt5 SIP dispatch context에서 sklearn dataclass(Tags) __init__
# 반환값 검사가 오작동함. sklearn 연산을 별도 Python 스레드에서 실행하여 우회.
_sklearn_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sklearn")

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
        # SIP 컨텍스트 밖 스레드에서 실행 (Python 3.14 + PyQt5 SIP 호환성)
        X = features.reshape(1, -1)
        if_raw = float(_sklearn_pool.submit(self._model.score_samples, X).result(timeout=5.0)[0])
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
        # fit과 score_samples 모두 SIP 컨텍스트 밖 스레드에서 실행
        _sklearn_pool.submit(self._model.fit, data).result(timeout=30.0)
        baseline_scores = _sklearn_pool.submit(self._model.score_samples, data).result(timeout=10.0)
        self._score_mean = float(baseline_scores.mean())
        self._score_std = float(baseline_scores.std()) + 1e-9


class AnomalyDetector(QObject):
    """4채널 통합 이상 감지 관리자."""

    state_changed = pyqtSignal(object)  # list[State] (4채널)

    def __init__(self, n_channels: int = 4, baseline_count: int = 20, parent=None):
        super().__init__(parent)
        self._detectors = [ChannelAnomalyDetector(baseline_count) for _ in range(n_channels)]

    def update(self, features: np.ndarray) -> List[State]:
        """features: shape (n_channels, 7)"""
        states = [self._detectors[ch].update(features[ch]) for ch in range(len(self._detectors))]
        self.state_changed.emit(states)
        return states

    def if_scores(self) -> List[float]:
        return [d.if_score for d in self._detectors]

    def zscore_maxes(self) -> List[float]:
        return [d.zscore_max for d in self._detectors]
