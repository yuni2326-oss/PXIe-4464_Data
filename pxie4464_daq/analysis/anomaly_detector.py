from __future__ import annotations
import logging
from enum import Enum, auto
from typing import List

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

# [Python 3.14 호환성 이력]
# 이전 버전은 sklearn IsolationForest.score_samples 를 사용했으나,
# Python 3.14 + PyQt5 SIP 디스패치 컨텍스트에서 sklearn 내부 __init__ 호출이
# "__init__() should return None, not 'NoneType'" 오류로 매 사이클·매 채널 100% 실패했다.
# (worker 스레드 우회도 효과 없었음 → 로그상 33,744건/일 에러)
#
# numpy 연산은 동일 컨텍스트에서 정상 동작함이 검증되어,
# 이상도 지표를 sklearn IsolationForest → 정규화 마할라노비스 거리(순수 numpy)로 교체.
# 마할라노비스 거리는 특징 간 상관구조를 반영하므로, 단일 특징 Z-score가 놓치는
# 다특징 확산 드리프트(펌프 열화 등)도 포착한다.

ZSCORE_WARNING = 3.0
ZSCORE_ALARM = 5.0
NORM_DEV_WARNING = -2.0  # 베이스라인 거리분포 대비 정규화 편차 임계값 (낮을수록 이상)
NORM_DEV_ALARM = -3.0
WARNING_HOLDOFF = 3      # 연속 n회 이상 판정 시에만 WARNING/ALARM 발동
_COV_RIDGE = 1e-3        # 공분산 정칙화 계수 (소표본·특이행렬 대비)


class State(Enum):
    LEARNING = auto()
    NORMAL = auto()
    WARNING = auto()
    ALARM = auto()


class ChannelAnomalyDetector:
    """단일 채널 이상 감지기 (Z-score + 정규화 마할라노비스 거리)."""

    def __init__(self, baseline_count: int = 20):
        self._baseline_count = baseline_count
        self._baseline: list = []
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None   # 정칙화된 공분산 역행렬
        self._dist_mean: float = 0.0              # 베이스라인 마할라노비스 거리 평균
        self._dist_std: float = 1.0               # 베이스라인 마할라노비스 거리 표준편차
        self.state: State = State.LEARNING
        self._norm_dev: float = 0.0               # 정규화 편차 (이상도 지표, 낮을수록 이상)
        self._zscore_max: float = 0.0
        self._warning_streak: int = 0

    def update(self, features: np.ndarray) -> State:
        if self.state == State.LEARNING:
            self._baseline.append(features.copy())
            if len(self._baseline) >= self._baseline_count:
                self._fit_model()
                self.state = State.NORMAL
            return self.state

        # Z-score (특징별 편차의 최대값) — 단일 특징 스파이크 포착
        zscores = np.abs((features - self._mean) / (self._std + 1e-10))
        self._zscore_max = float(np.max(zscores))

        # 정규화 마할라노비스 거리 — 다특징 상관 드리프트 포착
        dist = self._mahalanobis(features)
        # 거리가 클수록 이상 → 부호 반전하여 "낮을수록 이상" 규약 유지 (기존 임계값 호환)
        self._norm_dev = -(dist - self._dist_mean) / self._dist_std

        # 상태 결정 (AND 로직: 두 지표 모두 정상 범위여야 낮은 심각도 유지)
        if self._zscore_max < ZSCORE_WARNING and self._norm_dev > NORM_DEV_WARNING:
            new_state = State.NORMAL
        elif self._zscore_max < ZSCORE_ALARM and self._norm_dev > NORM_DEV_ALARM:
            new_state = State.WARNING
        else:
            new_state = State.ALARM

        # 홀드오프: WARNING/ALARM 연속 n회여야 발동 (순간 노이즈 무시)
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
        """베이스라인 대비 정규화 편차 (낮을수록 이상). 호환을 위해 이름 유지."""
        return self._norm_dev

    @property
    def zscore_max(self) -> float:
        return self._zscore_max

    def _mahalanobis(self, x: np.ndarray) -> float:
        diff = x - self._mean
        d2 = float(diff @ self._cov_inv @ diff)
        return float(np.sqrt(max(d2, 0.0)))

    def _fit_model(self) -> None:
        data = np.array(self._baseline)
        self._mean = data.mean(axis=0)
        self._std = data.std(axis=0)

        n_features = data.shape[1]
        cov = np.cov(data, rowvar=False)
        if cov.ndim == 0:  # 특징 1개인 경우 스칼라 방지
            cov = cov.reshape(1, 1)
        # 정칙화: 평균 분산에 비례한 ridge 추가 → 소표본/특이행렬에서도 안정
        ridge = (np.trace(cov) / n_features) * _COV_RIDGE
        cov_reg = cov + ridge * np.eye(n_features)
        self._cov_inv = np.linalg.pinv(cov_reg)  # 의사역행렬 — 특이행렬 안전

        # 베이스라인 거리 분포로 정규화 기준 수립
        dists = np.array([self._mahalanobis(row) for row in data])
        self._dist_mean = float(dists.mean())
        self._dist_std = float(dists.std()) + 1e-9


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
            except Exception as exc:
                logger.error("CH%d 이상감지 업데이트 오류 (건너뜀): %s", ch, exc)
                states.append(self._detectors[ch].state)
        self.state_changed.emit(states)
        return states

    def if_scores(self) -> List[float]:
        return [d.if_score for d in self._detectors]

    def zscore_maxes(self) -> List[float]:
        return [d.zscore_max for d in self._detectors]
