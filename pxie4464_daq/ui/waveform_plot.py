from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout

COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]  # CH0~3
DISPLAY_SAMPLES = 4096  # 화면에 표시할 최대 샘플 수


class WaveformPlot(QWidget):
    """4채널 실시간 시간 파형 플롯."""

    def __init__(self, sample_rate: float = 51200.0, parent=None):
        super().__init__(parent)
        self._sample_rate = sample_rate
        layout = QVBoxLayout(self)
        self._plot_widget = pg.PlotWidget(title="Waveform (g)")
        self._plot_widget.setLabel("left", "Acceleration", "g")
        self._plot_widget.setLabel("bottom", "Time", "s")
        self._plot_widget.addLegend()
        layout.addWidget(self._plot_widget)
        self._curves = [
            self._plot_widget.plot(pen=pg.mkPen(color=c, width=1), name=f"CH{i}")
            for i, c in enumerate(COLORS)
        ]
        self._buffer = np.zeros((4, DISPLAY_SAMPLES))

    def update(self, data: np.ndarray) -> None:
        """data: (4, N)"""
        n = data.shape[1]
        if n >= DISPLAY_SAMPLES:
            self._buffer = data[:, -DISPLAY_SAMPLES:]
        else:
            self._buffer = np.roll(self._buffer, -n, axis=1)
            self._buffer[:, -n:] = data
        t = np.arange(DISPLAY_SAMPLES) / self._sample_rate
        for ch, curve in enumerate(self._curves):
            curve.setData(t, self._buffer[ch])
