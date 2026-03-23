from __future__ import annotations
from typing import List
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QBrush
from pxie4464_daq.analysis.anomaly_detector import State

STATE_COLORS = {
    State.LEARNING: "#2196F3",   # 파란색
    State.NORMAL:   "#4CAF50",   # 초록색
    State.WARNING:  "#FF9800",   # 노란색
    State.ALARM:    "#F44336",   # 빨간색
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
    """4채널 최악 상태 기준 신호등 + 채널별 상태 텍스트."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._bulb = _Bulb()
        self._label = QLabel("LEARNING")
        self._label.setAlignment(Qt.AlignCenter)
        self._ch_labels = [QLabel(f"CH{i}: LEARNING") for i in range(4)]
        layout.addWidget(self._bulb, alignment=Qt.AlignCenter)
        layout.addWidget(self._label, alignment=Qt.AlignCenter)
        for lbl in self._ch_labels:
            layout.addWidget(lbl)

    def update_states(self, states: List[State]) -> None:
        worst = max(states, key=lambda s: STATE_PRIORITY.index(s))
        self._bulb.set_color(STATE_COLORS[worst])
        self._label.setText(STATE_LABELS[worst])
        for ch, state in enumerate(states):
            self._ch_labels[ch].setText(f"CH{ch}: {STATE_LABELS[state]}")
