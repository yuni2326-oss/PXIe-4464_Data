from __future__ import annotations
from typing import List
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout

COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",   # CH0-3  (4464)
    "#FF9F43", "#EE5A24", "#0652DD", "#1289A7",    # CH4-7  (4492)
    "#C4E538", "#A3CB38", "#FDA7DF", "#D980FA",    # CH8-11 (4492)
]
MAX_CHANNELS = len(COLORS)
DISPLAY_SAMPLES = 4096  # 화면에 표시할 최대 샘플 수


class WaveformPlot(QWidget):
    """n채널 실시간 시간 파형 플롯."""

    def __init__(self, sample_rate: float = 51200.0, parent=None):
        super().__init__(parent)
        self._sample_rate = sample_rate
        layout = QVBoxLayout(self)
        self._plot_widget = pg.PlotWidget(title="Waveform")
        self._plot_widget.setLabel("left", "Acceleration", "g")
        self._plot_widget.setLabel("bottom", "Time", "s")
        self._plot_widget.addLegend()
        layout.addWidget(self._plot_widget)
        self._curves: List[pg.PlotDataItem] = []
        self._active: List[int] = []
        self._buffer = np.zeros((MAX_CHANNELS, DISPLAY_SAMPLES))

    def reconfigure(self, channel_indices: List[int]) -> None:
        """활성 채널 변경 시 호출 — 커브 재생성."""
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
        self._buffer = np.zeros((MAX_CHANNELS, DISPLAY_SAMPLES))

    def update(self, data: np.ndarray, channel_indices: List[int]) -> None:
        """data: (n_active, N); channel_indices: 원래 채널 번호 목록"""
        n = data.shape[1]
        t = np.arange(DISPLAY_SAMPLES) / self._sample_rate
        for local_i, ch_idx in enumerate(channel_indices):
            if n >= DISPLAY_SAMPLES:
                self._buffer[ch_idx] = data[local_i, -DISPLAY_SAMPLES:]
            else:
                self._buffer[ch_idx] = np.roll(self._buffer[ch_idx], -n)
                self._buffer[ch_idx, -n:] = data[local_i]
            if ch_idx in self._curves:
                self._curves[ch_idx].setData(t, self._buffer[ch_idx])
