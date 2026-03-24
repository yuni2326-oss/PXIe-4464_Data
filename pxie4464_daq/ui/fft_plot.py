from __future__ import annotations
from typing import List
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from pxie4464_daq.ui.waveform_plot import COLORS, MAX_CHANNELS

FREQ_MAX_HZ = 5000.0  # FFT 표시 상한 주파수


class FFTPlot(QWidget):
    """n채널 실시간 FFT 스펙트럼 플롯."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._plot_widget = pg.PlotWidget(title="FFT Spectrum")
        self._plot_widget.setLabel("left", "Magnitude", "g")
        self._plot_widget.setLabel("bottom", "Frequency", "Hz")
        self._plot_widget.addLegend()
        layout.addWidget(self._plot_widget)
        self._plot_widget.setXRange(0, FREQ_MAX_HZ)
        self._curves = {}
        self._active: List[int] = []

    def reconfigure(self, channel_indices: List[int]) -> None:
        self._plot_widget.clear()
        self._plot_widget.addLegend()
        self._active = list(channel_indices)
        self._curves = {
            ch: self._plot_widget.plot(
                pen=pg.mkPen(color=COLORS[ch % MAX_CHANNELS], width=1),
                name=f"CH{ch}"
            )
            for ch in channel_indices
        }
        self._plot_widget.setXRange(0, FREQ_MAX_HZ)

    def update(self, frequencies: list, magnitudes: list, channel_indices: List[int]) -> None:
        """frequencies/magnitudes: 활성 채널 순서 리스트; channel_indices: 원래 채널 번호"""
        for local_i, ch_idx in enumerate(channel_indices):
            if ch_idx in self._curves:
                freqs = frequencies[local_i]
                mags  = magnitudes[local_i]
                mask  = freqs <= FREQ_MAX_HZ
                self._curves[ch_idx].setData(freqs[mask], mags[mask])
