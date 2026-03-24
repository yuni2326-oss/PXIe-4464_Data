from __future__ import annotations
from typing import List
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QBrush
from pxie4464_daq.analysis.anomaly_detector import State

STATE_COLORS = {
    State.LEARNING: "#2196F3",
    State.NORMAL:   "#4CAF50",
    State.WARNING:  "#FF9800",
    State.ALARM:    "#F44336",
}
STATE_LABELS = {
    State.LEARNING: "LEARNING",
    State.NORMAL:   "NORMAL",
    State.WARNING:  "WARNING",
    State.ALARM:    "ALARM",
}
STATE_PRIORITY = [State.LEARNING, State.NORMAL, State.WARNING, State.ALARM]


class _Bulb(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = QColor(STATE_COLORS[State.LEARNING])
        self.setFixedSize(40, 40)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(self._color))
        p.drawEllipse(4, 4, 32, 32)


class StatusLight(QWidget):
    """전체 최악 상태 신호등 + 채널별 상태 텍스트 (동적 채널 수)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._bulb = _Bulb()
        self._overall_label = QLabel("LEARNING")
        self._overall_label.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._bulb, alignment=Qt.AlignCenter)
        self._layout.addWidget(self._overall_label, alignment=Qt.AlignCenter)
        self._ch_labels: List[QLabel] = []
        self._active: List[int] = []

    def reconfigure(self, channel_indices: List[int]) -> None:
        # 기존 채널 레이블 제거
        for lbl in self._ch_labels:
            self._layout.removeWidget(lbl)
            lbl.deleteLater()
        self._ch_labels = []
        self._active = list(channel_indices)
        for ch in channel_indices:
            lbl = QLabel(f"CH{ch}: LEARNING")
            self._layout.addWidget(lbl)
            self._ch_labels.append(lbl)

    def update_states(self, states: List[State], channel_indices: List[int]) -> None:
        if not states:
            return
        worst = max(states, key=lambda s: STATE_PRIORITY.index(s))
        self._bulb.set_color(STATE_COLORS[worst])
        self._overall_label.setText(STATE_LABELS[worst])
        for local_i, ch_idx in enumerate(channel_indices):
            if local_i < len(self._ch_labels):
                self._ch_labels[local_i].setText(f"CH{ch_idx}: {STATE_LABELS[states[local_i]]}")
