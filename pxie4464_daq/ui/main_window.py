from __future__ import annotations
import logging
from datetime import datetime
from typing import Optional, List

import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QMessageBox, QGridLayout, QScrollArea
)
from PyQt5.QtCore import Qt

from pxie4464_daq.device.daq import (
    MockDAQ, PXIe4464, PXIe4492, MultiDAQ, _DAQBase,
    N_CHANNELS_4464, N_CHANNELS_4492,
)
from pxie4464_daq.acquisition.worker import AcquisitionWorker
from pxie4464_daq.analysis.fft import compute_fft
from pxie4464_daq.analysis.feature_collector import FeatureCollector
from pxie4464_daq.analysis.anomaly_detector import AnomalyDetector
from pxie4464_daq.storage.csv_writer import save_raw, save_fft
from pxie4464_daq.storage.data_saver import DataSaver
from pxie4464_daq.ui.waveform_plot import WaveformPlot
from pxie4464_daq.ui.fft_plot import FFTPlot
from pxie4464_daq.ui.anomaly_plot import AnomalyPlot
from pxie4464_daq.ui.status_light import StatusLight

logger = logging.getLogger(__name__)

VOLTAGE_RANGES = [1.0, 3.16, 10.0, 31.6]
N_TOTAL_MAX = N_CHANNELS_4464 + N_CHANNELS_4492  # 12


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PXIe-4464 / 4492 진동 데이터 수집")
        self._daq: Optional[_DAQBase] = None
        self._worker: Optional[AcquisitionWorker] = None
        self._collector: Optional[FeatureCollector] = None
        self._detector: Optional[AnomalyDetector] = None
        self._data_saver: Optional[DataSaver] = None
        self._last_data: Optional[np.ndarray] = None
        self._last_freqs: Optional[List] = None
        self._last_mags: Optional[List] = None
        self._enabled_indices: List[int] = list(range(N_CHANNELS_4464))  # 초기: 4464 4채널
        self._setup_ui()

    # ── UI 구성 ─────────────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # 좌측: 스크롤 가능한 제어 패널
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(280)
        ctrl_container = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_container)
        ctrl_layout.addWidget(self._make_device_panel())
        ctrl_layout.addWidget(self._make_channel_panel())
        ctrl_layout.addWidget(self._make_action_panel())
        ctrl_layout.addStretch()
        scroll.setWidget(ctrl_container)
        root.addWidget(scroll)

        # 우측: 플롯 영역
        plots = QVBoxLayout()
        top_plots = QHBoxLayout()
        self._waveform_plot = WaveformPlot()
        self._fft_plot = FFTPlot()
        top_plots.addWidget(self._waveform_plot)
        top_plots.addWidget(self._fft_plot)
        plots.addLayout(top_plots)

        bottom_plots = QHBoxLayout()
        self._anomaly_plot = AnomalyPlot()
        self._status_light = StatusLight()
        bottom_plots.addWidget(self._anomaly_plot, stretch=3)
        bottom_plots.addWidget(self._status_light, stretch=1)
        plots.addLayout(bottom_plots)

        root.addLayout(plots, stretch=4)

    def _make_device_panel(self) -> QGroupBox:
        group = QGroupBox("장치 설정")
        layout = QGridLayout(group)
        row = 0

        # ── PXIe-4464 ──
        layout.addWidget(QLabel("PXIe-4464 장치명"), row, 0)
        self._dev4464_edit = QLineEdit("PXI1Slot3")
        layout.addWidget(self._dev4464_edit, row, 1)
        row += 1

        # ── PXIe-4492 ──
        self._use_4492_check = QCheckBox("PXIe-4492 사용")
        self._use_4492_check.setChecked(False)
        self._use_4492_check.toggled.connect(self._on_4492_toggled)
        layout.addWidget(self._use_4492_check, row, 0, 1, 2)
        row += 1

        layout.addWidget(QLabel("PXIe-4492 장치명"), row, 0)
        self._dev4492_edit = QLineEdit("PXI1Slot5")
        self._dev4492_edit.setEnabled(False)
        layout.addWidget(self._dev4492_edit, row, 1)
        row += 1

        # ── 공통 파라미터 ──
        layout.addWidget(QLabel("샘플레이트 (S/s)"), row, 0)
        self._sample_rate_edit = QLineEdit("51200")
        layout.addWidget(self._sample_rate_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("청크 크기 (샘플)"), row, 0)
        self._chunk_edit = QLineEdit("1024")
        layout.addWidget(self._chunk_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("전압 범위 (±V)"), row, 0)
        self._voltage_combo = QComboBox()
        for v in VOLTAGE_RANGES:
            self._voltage_combo.addItem(f"±{v}V", v)
        self._voltage_combo.setCurrentIndex(2)  # ±10V
        layout.addWidget(self._voltage_combo, row, 1)
        row += 1

        layout.addWidget(QLabel("수집 주기 (s)"), row, 0)
        self._cycle_sec_edit = QLineEdit("30")
        layout.addWidget(self._cycle_sec_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("수집 시간 (s/회)"), row, 0)
        self._window_sec_edit = QLineEdit("5")
        layout.addWidget(self._window_sec_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("학습 누적 횟수"), row, 0)
        self._baseline_count_edit = QLineEdit("20")
        layout.addWidget(self._baseline_count_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("저장 주기 (분)"), row, 0)
        self._save_interval_edit = QLineEdit("30")
        layout.addWidget(self._save_interval_edit, row, 1)
        row += 1

        self._mock_check = QCheckBox("Mock 모드")
        self._mock_check.setChecked(True)
        layout.addWidget(self._mock_check, row, 0, 1, 2)

        return group

    def _make_channel_panel(self) -> QGroupBox:
        group = QGroupBox("채널 활성화")
        layout = QGridLayout(group)

        self._ch_checks: List[QCheckBox] = []

        # 4464 채널 (CH0~3)
        layout.addWidget(QLabel("── PXIe-4464 ──"), 0, 0, 1, 4)
        for i in range(N_CHANNELS_4464):
            cb = QCheckBox(f"CH{i}")
            cb.setChecked(True)
            layout.addWidget(cb, 1, i)
            self._ch_checks.append(cb)

        # 4492 채널 (CH4~11) — 초기에는 비활성화
        layout.addWidget(QLabel("── PXIe-4492 ──"), 2, 0, 1, 4)
        for i in range(N_CHANNELS_4492):
            ch_idx = N_CHANNELS_4464 + i
            cb = QCheckBox(f"CH{ch_idx}")
            cb.setChecked(True)
            cb.setEnabled(False)
            layout.addWidget(cb, 3 + i // 4, i % 4)
            self._ch_checks.append(cb)

        return group

    def _make_action_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self._connect_btn = QPushButton("연결")
        self._connect_btn.clicked.connect(self._on_connect)
        self._start_btn = QPushButton("▶ 시작")
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start_stop)
        self._save_btn = QPushButton("CSV 저장")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_csv)
        layout.addWidget(self._connect_btn)
        layout.addWidget(self._start_btn)
        layout.addWidget(self._save_btn)
        return w

    # ── 슬롯 ────────────────────────────────────────────────────────────────

    def _on_4492_toggled(self, checked: bool) -> None:
        self._dev4492_edit.setEnabled(checked)
        for i in range(N_CHANNELS_4492):
            self._ch_checks[N_CHANNELS_4464 + i].setEnabled(checked)

    def _on_connect(self):
        try:
            sample_rate = float(self._sample_rate_edit.text())
            chunk = int(self._chunk_edit.text())
            voltage_range = self._voltage_combo.currentData()
            cycle_sec = float(self._cycle_sec_edit.text())
            window_sec = float(self._window_sec_edit.text())
            baseline_count = int(self._baseline_count_edit.text())
            save_interval_sec = float(self._save_interval_edit.text()) * 60.0
            use_4492 = self._use_4492_check.isChecked()
            mock = self._mock_check.isChecked()

            # 활성 채널 인덱스 계산
            n_total = N_CHANNELS_4464 + (N_CHANNELS_4492 if use_4492 else 0)
            self._enabled_indices = [
                i for i in range(n_total) if self._ch_checks[i].isChecked()
            ]
            if not self._enabled_indices:
                QMessageBox.warning(self, "설정 오류", "활성화된 채널이 없습니다.")
                return
            n_active = len(self._enabled_indices)

            # DAQ 장치 생성
            if mock:
                daq_4464 = MockDAQ(n_channels=N_CHANNELS_4464)
                if use_4492:
                    daq_4492 = MockDAQ(n_channels=N_CHANNELS_4492)
                    self._daq = MultiDAQ(daq_4464, daq_4492)
                else:
                    self._daq = daq_4464
            else:
                daq_4464 = PXIe4464(device_name=self._dev4464_edit.text())
                if use_4492:
                    daq_4492 = PXIe4492(device_name=self._dev4492_edit.text(),
                                        voltage_range=voltage_range)
                    self._daq = MultiDAQ(daq_4464, daq_4492)
                else:
                    self._daq = daq_4464

            self._daq.configure(sample_rate=sample_rate, record_length=chunk,
                                voltage_range=voltage_range)

            # 플롯 재구성
            self._waveform_plot._sample_rate = sample_rate
            self._waveform_plot.reconfigure(self._enabled_indices)
            self._fft_plot.reconfigure(self._enabled_indices)
            self._anomaly_plot.reconfigure(self._enabled_indices)
            self._status_light.reconfigure(self._enabled_indices)

            # 분석 파이프라인 생성
            self._collector = FeatureCollector(
                sample_rate=sample_rate,
                collection_cycle_sec=cycle_sec,
                window_sec=window_sec,
                n_channels=n_active,
            )
            self._detector = AnomalyDetector(n_channels=n_active, baseline_count=baseline_count)
            self._detector.state_changed.connect(self._on_state_changed)
            self._collector.features_ready.connect(self._detector.update)

            self._data_saver = DataSaver(sample_rate=sample_rate, save_dir="results",
                                         save_interval_sec=save_interval_sec)
            self._collector.raw_ready.connect(self._data_saver.on_raw)

            self._start_btn.setEnabled(True)
            self._connect_btn.setEnabled(False)
            logger.info("연결 완료: %s, 활성 채널=%s, sr=%.0f",
                        type(self._daq).__name__, self._enabled_indices, sample_rate)

        except Exception as exc:
            QMessageBox.critical(self, "연결 오류", str(exc))

    def _on_start_stop(self):
        if self._worker is None or not self._worker.isRunning():
            self._start_acquisition()
        else:
            self._stop_acquisition()

    def _start_acquisition(self):
        self._worker = AcquisitionWorker(self._daq)
        self._worker.data_ready.connect(self._on_data_ready)
        self._worker.error_occurred.connect(self._on_error)
        self._collector.start()
        self._worker.start()
        self._start_btn.setText("■ 정지")
        self._save_btn.setEnabled(True)

    def _stop_acquisition(self):
        if self._worker:
            self._worker.stop()
            self._worker = None
        if self._collector:
            self._collector.stop()
        self._start_btn.setText("▶ 시작")

    def _on_data_ready(self, data: np.ndarray):
        # 활성 채널만 필터링
        filtered = data[self._enabled_indices, :]
        self._last_data = filtered
        self._waveform_plot.update(filtered, self._enabled_indices)

        sample_rate = float(self._sample_rate_edit.text())
        freqs_list, mags_list = [], []
        for ch_data in filtered:
            freqs, mags = compute_fft(ch_data, sample_rate)
            freqs_list.append(freqs)
            mags_list.append(mags)
        self._last_freqs = freqs_list
        self._last_mags = mags_list
        self._fft_plot.update(freqs_list, mags_list, self._enabled_indices)

        self._collector.on_data_ready(filtered)

    def _on_state_changed(self, states):
        self._status_light.update_states(states, self._enabled_indices)
        if self._detector:
            self._anomaly_plot.update(self._detector.if_scores(), self._enabled_indices)

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "수집 오류", msg)
        self._stop_acquisition()

    def _on_save_csv(self):
        if self._last_data is None:
            QMessageBox.warning(self, "저장 실패", "저장할 데이터가 없습니다.")
            return
        ts = datetime.now()
        sample_rate = float(self._sample_rate_edit.text())
        try:
            save_raw(self._last_data, sample_rate=sample_rate, timestamp=ts)
            if self._last_freqs and self._last_mags:
                save_fft(self._last_freqs, self._last_mags, timestamp=ts)
            QMessageBox.information(self, "저장 완료", f"CSV 저장 완료: {ts.strftime('%Y%m%d_%H%M%S')}")
        except Exception as exc:
            QMessageBox.critical(self, "저장 오류", str(exc))

    def closeEvent(self, event):
        self._stop_acquisition()
        super().closeEvent(event)
