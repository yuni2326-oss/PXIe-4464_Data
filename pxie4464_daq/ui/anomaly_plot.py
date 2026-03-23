from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout

COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
HISTORY = 50


class AnomalyPlot(QWidget):
    """채널별 IsolationForest 이상 점수 히스토리 (50샘플 rolling)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._plot_widget = pg.PlotWidget(title="Anomaly Score History")
        self._plot_widget.setLabel("left", "IF Score")
        self._plot_widget.setLabel("bottom", "Sample")
        self._plot_widget.addLegend()
        # 임계선
        self._plot_widget.addLine(y=-0.05, pen=pg.mkPen("y", style=pg.QtCore.Qt.DashLine))
        self._plot_widget.addLine(y=-0.15, pen=pg.mkPen("r", style=pg.QtCore.Qt.DashLine))
        layout.addWidget(self._plot_widget)
        self._curves = [
            self._plot_widget.plot(pen=pg.mkPen(color=c, width=1), name=f"CH{i}")
            for i, c in enumerate(COLORS)
        ]
        self._history = np.zeros((4, HISTORY))

    def update(self, if_scores: list) -> None:
        """if_scores: list[float] (4채널 IsolationForest 점수)"""
        self._history = np.roll(self._history, -1, axis=1)
        for ch, score in enumerate(if_scores):
            self._history[ch, -1] = score
        x = np.arange(HISTORY)
        for ch, curve in enumerate(self._curves):
            curve.setData(x, self._history[ch])
