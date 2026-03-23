from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout

COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]


class FFTPlot(QWidget):
    """4채널 실시간 FFT 스펙트럼 플롯."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._plot_widget = pg.PlotWidget(title="FFT Spectrum")
        self._plot_widget.setLabel("left", "Magnitude", "g")
        self._plot_widget.setLabel("bottom", "Frequency", "Hz")
        self._plot_widget.setLogMode(x=False, y=False)
        self._plot_widget.addLegend()
        layout.addWidget(self._plot_widget)
        self._curves = [
            self._plot_widget.plot(pen=pg.mkPen(color=c, width=1), name=f"CH{i}")
            for i, c in enumerate(COLORS)
        ]

    def update(self, frequencies: list, magnitudes: list) -> None:
        """
        frequencies: list of np.ndarray (4채널)
        magnitudes:  list of np.ndarray (4채널)
        """
        for ch, (freqs, mags) in enumerate(zip(frequencies, magnitudes)):
            self._curves[ch].setData(freqs, mags)
