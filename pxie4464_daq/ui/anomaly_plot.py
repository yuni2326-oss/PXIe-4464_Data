from __future__ import annotations
from typing import List
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from pxie4464_daq.ui.waveform_plot import COLORS, MAX_CHANNELS

HISTORY = 50


class AnomalyPlot(QWidget):
    """채널별 정규화 편차(norm_dev) 이상 점수 히스토리 (50샘플 rolling)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._plot_widget = pg.PlotWidget(title="Anomaly Score History (norm_dev)")
        self._plot_widget.setLabel("left", "Norm. Deviation")
        self._plot_widget.setLabel("bottom", "Sample")
        self._plot_widget.addLegend()
        # 임계선: WARNING=-2.0, ALARM=-3.0
        self._plot_widget.addLine(y=-2.0, pen=pg.mkPen("y", style=pg.QtCore.Qt.DashLine))
        self._plot_widget.addLine(y=-3.0, pen=pg.mkPen("r", style=pg.QtCore.Qt.DashLine))
        layout.addWidget(self._plot_widget)
        self._curves = {}
        self._history = np.zeros((MAX_CHANNELS, HISTORY))
        self._active: List[int] = []

    def reconfigure(self, channel_indices: List[int]) -> None:
        self._plot_widget.clear()
        self._plot_widget.addLegend()
        self._plot_widget.addLine(y=-2.0, pen=pg.mkPen("y", style=pg.QtCore.Qt.DashLine))
        self._plot_widget.addLine(y=-3.0, pen=pg.mkPen("r", style=pg.QtCore.Qt.DashLine))
        self._active = list(channel_indices)
        self._curves = {
            ch: self._plot_widget.plot(
                pen=pg.mkPen(color=COLORS[ch % MAX_CHANNELS], width=1),
                name=f"CH{ch}"
            )
            for ch in channel_indices
        }
        self._history = np.zeros((MAX_CHANNELS, HISTORY))

    def update(self, scores: list, channel_indices: List[int]) -> None:
        """scores: 활성 채널 순서 norm_dev 리스트; channel_indices: 원래 채널 번호"""
        x = np.arange(HISTORY)
        for local_i, ch_idx in enumerate(channel_indices):
            self._history[ch_idx] = np.roll(self._history[ch_idx], -1)
            self._history[ch_idx, -1] = scores[local_i]
            if ch_idx in self._curves:
                self._curves[ch_idx].setData(x, self._history[ch_idx])
