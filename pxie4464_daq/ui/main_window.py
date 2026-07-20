from __future__ import annotations
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QMessageBox, QGridLayout, QScrollArea, QApplication
)
from PyQt5.QtCore import Qt, QTimer

from pxie4464_daq.device.daq import (
    MockDAQ, PXIe4464, PXIe4492, MultiDAQ, _DAQBase,
    N_CHANNELS_4464, N_CHANNELS_4492, DEFAULT_SENSITIVITY,
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

# NAS 전송 기본 경로 (네트워크 드라이브 UNC)
DEFAULT_NAS_DIR = r"\\10.130.121.158\ai_gpu_001\GY_DATA\pump data"


def _parse_int_list(text: str) -> list:
    """쉼표/공백 구분 정수 목록 파싱. 잘못된 항목은 무시."""
    out = []
    for tok in (text or "").replace(",", " ").split():
        try:
            out.append(int(tok))
        except ValueError:
            pass
    return out

# 자동 재시작 백오프 파라미터
RESTART_BASE_SEC = 5       # 첫 재시도 지연
RESTART_MAX_SEC = 300      # 지연 상한 (5분)
STABLE_RESET_SEC = 120     # 이 시간 무오류 운전 시 재시도 카운터 초기화

# 프로세스 내 복구가 이 횟수만큼 연속 실패하면(= Python 3.14+SIP 상태 손상으로
# 같은 프로세스에서 회복 불가) 종료 코드 EXIT_NEEDS_RESTART로 빠져나가
# 외부 supervisor가 새 프로세스로 재시작하게 한다.
MAX_INPROC_RESTART_FAILS = 3
EXIT_NEEDS_RESTART = 42

# 패키지 루트(pxie4464-daq) 기준 절대경로 — 작업 디렉터리와 무관하게 daq.log와 동일 위치
_BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = _BASE_DIR / "config" / "last_session.json"
HEARTBEAT_PATH = _BASE_DIR / "logs" / "heartbeat.txt"
HEARTBEAT_FILE_SEC = 20    # supervisor 감시용 heartbeat 파일 기록 주기


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
        self._active_sample_rate: Optional[float] = None  # 현재 가동 중인 샘플레이트(위젯 편집과 분리)
        self._plot_warned: set = set()  # 플롯 갱신 실패 1회 경고 추적

        # 자동 재시작(워치독): DAQ 오류 시 초기 실행조건 그대로 재연결+재시작
        self._start_config: Optional[dict] = None   # 초기 실행조건 스냅샷
        self._auto_restart_enabled: bool = True
        self._restart_attempts: int = 0
        self._consecutive_restart_failures: int = 0  # 프로세스 내 복구 연속 실패 횟수
        self._setup_ui()
        self._restart_timer = QTimer(self)
        self._restart_timer.setSingleShot(True)
        self._restart_timer.timeout.connect(self._attempt_restart)
        self._stable_timer = QTimer(self)
        self._stable_timer.setSingleShot(True)
        self._stable_timer.timeout.connect(self._on_stable)
        self._start_heartbeat()
        self._start_heartbeat_file()

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

        # 센서 감도 (mV/(m/s²)) — 4464 가속도 변환 + 4492 가속도계 채널 변환에 적용.
        # PCB 352C33 = 10.2 mV/(m/s²) (≈100 mV/g)
        layout.addWidget(QLabel("가속도계 감도 (mV/(m/s²))"), row, 0)
        self._sensitivity_edit = QLineEdit("10.2")
        layout.addWidget(self._sensitivity_edit, row, 1)
        row += 1

        # 마이크로폰 채널 — 4492에서 마이크가 연결된 CH 번호(쉼표구분). 해당 채널은 Pa로 변환.
        # 예: Crysound 333T1(IEPE, 50 mV/Pa)을 CH6에 연결 → "6"
        layout.addWidget(QLabel("마이크 채널 (CH)"), row, 0)
        self._mic_ch_edit = QLineEdit("")
        self._mic_ch_edit.setPlaceholderText("예: 6 (비우면 없음)")
        layout.addWidget(self._mic_ch_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("마이크 감도 (mV/Pa)"), row, 0)
        self._mic_sens_edit = QLineEdit("50")
        layout.addWidget(self._mic_sens_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("FFT 표시 상한 (Hz)"), row, 0)
        self._fft_max_edit = QLineEdit("5000")
        layout.addWidget(self._fft_max_edit, row, 1)
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

        # 저장 하위폴더 — 비우면 results/ 루트, 입력 시 results/<이름>/ 에 저장
        layout.addWidget(QLabel("저장 하위폴더"), row, 0)
        self._subdir_edit = QLineEdit("")
        self._subdir_edit.setPlaceholderText("비우면 results/ 루트")
        layout.addWidget(self._subdir_edit, row, 1)
        row += 1

        # NAS 전송 경로 — 비우면 전송 안 함. 로컬 저장 직후 백그라운드로 복사.
        layout.addWidget(QLabel("NAS 전송 경로"), row, 0)
        self._nas_edit = QLineEdit(DEFAULT_NAS_DIR)
        self._nas_edit.setPlaceholderText("비우면 NAS 전송 안 함")
        layout.addWidget(self._nas_edit, row, 1)
        row += 1

        self._nas_delete_check = QCheckBox("NAS 전송 후 로컬 삭제")
        self._nas_delete_check.setChecked(False)
        layout.addWidget(self._nas_delete_check, row, 0, 1, 2)
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
        self._auto_restart_check = QCheckBox("오류 시 자동 재시작")
        self._auto_restart_check.setChecked(True)
        self._auto_restart_check.toggled.connect(self._on_auto_restart_toggled)
        layout.addWidget(self._connect_btn)
        layout.addWidget(self._start_btn)
        layout.addWidget(self._save_btn)
        layout.addWidget(self._auto_restart_check)
        return w

    # ── Heartbeat ───────────────────────────────────────────────────────────

    def _start_heartbeat(self) -> None:
        self._heartbeat_timer = QTimer(self)
        self._heartbeat_timer.setInterval(3600 * 1000)  # 1시간마다
        self._heartbeat_timer.timeout.connect(self._on_heartbeat)
        self._heartbeat_timer.start()

    def _start_heartbeat_file(self) -> None:
        """supervisor 감시용 heartbeat 파일을 주기적으로 갱신한다.
        메인 스레드(GUI 이벤트 루프)가 살아있다는 증거 — 멈추면 supervisor가 재시작."""
        self._hb_file_timer = QTimer(self)
        self._hb_file_timer.setInterval(HEARTBEAT_FILE_SEC * 1000)
        self._hb_file_timer.timeout.connect(self._write_heartbeat_file)
        self._hb_file_timer.start()
        self._write_heartbeat_file()

    def _write_heartbeat_file(self) -> None:
        try:
            HEARTBEAT_PATH.parent.mkdir(exist_ok=True)
            HEARTBEAT_PATH.write_text(str(time.time()), encoding="utf-8")
        except Exception:
            pass

    def _on_heartbeat(self) -> None:
        try:
            worker_state = "running" if (self._worker and self._worker.isRunning()) else "stopped"
            save_count = getattr(self._data_saver, "_save_count", 0) if self._data_saver else 0
            last_save = (
                self._data_saver._last_save_time.strftime("%H:%M:%S")
                if (self._data_saver and self._data_saver._last_save_time)
                else "없음"
            )
            logger.info(
                "[Heartbeat] worker=%s | DAQ=%s | 저장횟수=%d | 마지막저장=%s",
                worker_state,
                type(self._daq).__name__ if self._daq else "None",
                save_count,
                last_save,
            )
        except Exception as exc:
            logger.warning("[Heartbeat] 상태 기록 실패: %s", exc)

    # ── 슬롯 ────────────────────────────────────────────────────────────────

    def _on_4492_toggled(self, checked: bool) -> None:
        self._dev4492_edit.setEnabled(checked)
        for i in range(N_CHANNELS_4492):
            self._ch_checks[N_CHANNELS_4464 + i].setEnabled(checked)

    def _read_config(self) -> dict:
        """현재 UI 위젯 값을 초기 실행조건 스냅샷(dict)으로 읽는다."""
        use_4492 = self._use_4492_check.isChecked()
        n_total = N_CHANNELS_4464 + (N_CHANNELS_4492 if use_4492 else 0)
        enabled = [i for i in range(n_total) if self._ch_checks[i].isChecked()]
        return {
            "sample_rate": float(self._sample_rate_edit.text()),
            "chunk": int(self._chunk_edit.text()),
            "voltage_range": self._voltage_combo.currentData(),
            "cycle_sec": float(self._cycle_sec_edit.text()),
            "window_sec": float(self._window_sec_edit.text()),
            "baseline_count": int(self._baseline_count_edit.text()),
            "save_interval_sec": float(self._save_interval_edit.text()) * 60.0,
            "sensitivity": float(self._sensitivity_edit.text()),
            "mic_channels": _parse_int_list(self._mic_ch_edit.text()),
            "mic_sensitivity": float(self._mic_sens_edit.text() or 50.0),
            "fft_max_hz": float(self._fft_max_edit.text() or 5000.0),
            "save_subdir": self._subdir_edit.text().strip(),
            "nas_dir": self._nas_edit.text().strip(),
            "nas_delete": self._nas_delete_check.isChecked(),
            "use_4492": use_4492,
            "mock": self._mock_check.isChecked(),
            "dev4464": self._dev4464_edit.text(),
            "dev4492": self._dev4492_edit.text(),
            "enabled_indices": enabled,
        }

    def _build_pipeline(self, cfg: dict) -> None:
        """설정 스냅샷(cfg)으로 DAQ·플롯·분석 파이프라인을 구성한다.

        UI 대화창을 띄우지 않고 실패 시 예외를 올린다. 수동 연결과
        자동 재시작이 동일한 코드로 '초기 실행조건 그대로' 재구성된다.
        """
        if not cfg["enabled_indices"]:
            raise ValueError("활성화된 채널이 없습니다.")
        self._enabled_indices = list(cfg["enabled_indices"])
        n_active = len(self._enabled_indices)
        sample_rate = cfg["sample_rate"]
        self._active_sample_rate = sample_rate

        # DAQ 장치 생성
        if cfg["mock"]:
            daq_4464 = MockDAQ(n_channels=N_CHANNELS_4464)
            if cfg["use_4492"]:
                self._daq = MultiDAQ(daq_4464, MockDAQ(n_channels=N_CHANNELS_4492))
            else:
                self._daq = daq_4464
        else:
            sensitivity = cfg.get("sensitivity", DEFAULT_SENSITIVITY)
            daq_4464 = PXIe4464(device_name=cfg["dev4464"], sensitivity=sensitivity)
            if cfg["use_4492"]:
                # 마이크 채널: 전역 CH번호 → 4492 로컬 ai 인덱스(CH - 4464채널수)
                mic_locals = [c - N_CHANNELS_4464 for c in cfg.get("mic_channels", [])
                              if c >= N_CHANNELS_4464]
                daq_4492 = PXIe4492(device_name=cfg["dev4492"],
                                    voltage_range=cfg["voltage_range"],
                                    sensitivity=sensitivity,
                                    mic_channels=mic_locals,
                                    mic_sensitivity=cfg.get("mic_sensitivity", 50.0))
                self._daq = MultiDAQ(daq_4464, daq_4492)
            else:
                self._daq = daq_4464

        self._daq.configure(sample_rate=sample_rate, record_length=cfg["chunk"],
                            voltage_range=cfg["voltage_range"],
                            sensitivity=cfg.get("sensitivity", DEFAULT_SENSITIVITY))

        # 플롯 재구성
        self._waveform_plot._sample_rate = sample_rate
        self._waveform_plot.reconfigure(self._enabled_indices)
        self._fft_plot.set_freq_max(cfg.get("fft_max_hz", 5000.0))
        self._fft_plot.reconfigure(self._enabled_indices)
        self._anomaly_plot.reconfigure(self._enabled_indices)
        self._status_light.reconfigure(self._enabled_indices)

        # 분석 파이프라인 생성
        self._collector = FeatureCollector(
            sample_rate=sample_rate,
            collection_cycle_sec=cfg["cycle_sec"],
            window_sec=cfg["window_sec"],
            n_channels=n_active,
        )
        self._detector = AnomalyDetector(n_channels=n_active, baseline_count=cfg["baseline_count"])
        self._detector.state_changed.connect(self._on_state_changed)
        self._collector.features_ready.connect(self._detector.update)

        subdir = (cfg.get("save_subdir") or "").strip()
        save_dir = str(Path("results") / subdir) if subdir else "results"
        # NAS 대상: 하위폴더 구조를 그대로 미러링 (<NAS>/<subdir>)
        nas_root = (cfg.get("nas_dir") or "").strip()
        nas_dir = None
        if nas_root:
            nas_dir = str(Path(nas_root) / subdir) if subdir else nas_root
        self._data_saver = DataSaver(sample_rate=sample_rate, save_dir=save_dir,
                                     save_interval_sec=cfg["save_interval_sec"],
                                     nas_dir=nas_dir,
                                     delete_after_upload=cfg.get("nas_delete", False))
        logger.info("저장 폴더: %s | NAS: %s", save_dir, nas_dir or "(사용 안 함)")
        self._collector.raw_ready.connect(self._data_saver.on_raw)
        logger.info("파이프라인 구성: %s, 활성 채널=%s, sr=%.0f",
                    type(self._daq).__name__, self._enabled_indices, sample_rate)

    def _on_connect(self):
        try:
            cfg = self._read_config()
            self._build_pipeline(cfg)
        except Exception as exc:
            QMessageBox.critical(self, "연결 오류", str(exc))
            return
        self._start_config = cfg          # 초기 실행조건 스냅샷 저장
        self._restart_attempts = 0
        self._consecutive_restart_failures = 0
        self._save_session_config(cfg)    # 새 프로세스 자동시작용 파일 저장
        self._start_btn.setEnabled(True)
        self._connect_btn.setEnabled(False)
        logger.info("연결 완료 (자동 재시작 %s)",
                    "ON" if self._auto_restart_enabled else "OFF")

    # ── 세션 설정 영속화 / 자동 시작 ─────────────────────────────────────────

    def _save_session_config(self, cfg: dict) -> None:
        """초기 실행조건을 파일로 저장 (supervisor 재시작 시 자동 재개용)."""
        try:
            CONFIG_PATH.parent.mkdir(exist_ok=True)
            CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        except Exception as exc:
            logger.warning("세션 설정 저장 실패: %s", exc)

    def _load_session_config(self) -> Optional[dict]:
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _apply_config_to_widgets(self, cfg: dict) -> None:
        """저장된 설정을 UI 위젯에 반영 (표시 일관성)."""
        def _fmt(v):
            return str(int(v)) if float(v).is_integer() else str(v)
        self._dev4464_edit.setText(cfg.get("dev4464", "PXI1Slot3"))
        self._dev4492_edit.setText(cfg.get("dev4492", "PXI1Slot5"))
        self._use_4492_check.setChecked(bool(cfg.get("use_4492", False)))
        self._sample_rate_edit.setText(_fmt(cfg["sample_rate"]))
        self._chunk_edit.setText(str(int(cfg["chunk"])))
        idx = self._voltage_combo.findData(cfg.get("voltage_range"))
        if idx >= 0:
            self._voltage_combo.setCurrentIndex(idx)
        self._cycle_sec_edit.setText(_fmt(cfg["cycle_sec"]))
        self._window_sec_edit.setText(_fmt(cfg["window_sec"]))
        self._baseline_count_edit.setText(str(int(cfg["baseline_count"])))
        self._save_interval_edit.setText(_fmt(cfg["save_interval_sec"] / 60.0))
        self._sensitivity_edit.setText(_fmt(cfg.get("sensitivity", DEFAULT_SENSITIVITY)))
        self._mic_ch_edit.setText(" ".join(str(c) for c in cfg.get("mic_channels", [])))
        self._mic_sens_edit.setText(_fmt(cfg.get("mic_sensitivity", 50.0)))
        self._fft_max_edit.setText(_fmt(cfg.get("fft_max_hz", 5000.0)))
        self._subdir_edit.setText(cfg.get("save_subdir", ""))
        self._nas_edit.setText(cfg.get("nas_dir", DEFAULT_NAS_DIR))
        self._nas_delete_check.setChecked(bool(cfg.get("nas_delete", False)))
        self._mock_check.setChecked(bool(cfg.get("mock", True)))
        enabled = set(cfg.get("enabled_indices", []))
        for i, cb in enumerate(self._ch_checks):
            cb.setChecked(i in enabled)

    def autostart(self) -> None:
        """--autostart: 저장된 초기 실행조건으로 자동 연결+시작.
        supervisor가 새 프로세스를 띄울 때 무인 자동 재개를 가능케 한다."""
        cfg = self._load_session_config()
        if not cfg:
            logger.info("[autostart] 저장된 세션 설정 없음 — 수동 설정 대기")
            return
        logger.info("[autostart] 저장된 초기 실행조건으로 자동 시작")
        try:
            self._apply_config_to_widgets(cfg)
            # 위젯 반영 후 설정을 다시 읽어 새로 추가된 항목(NAS 경로 등)의 기본값이
            # 구버전 config에도 자동 반영되게 한다(스키마 진화 자가 치유).
            cfg = self._read_config()
            self._build_pipeline(cfg)
        except Exception as exc:
            logger.error("[autostart] 파이프라인 구성 실패: %s", exc)
            return
        self._start_config = cfg
        self._save_session_config(cfg)  # 갱신된 설정 저장(다음 재시작부터 정상)
        self._restart_attempts = 0
        self._consecutive_restart_failures = 0
        self._start_btn.setEnabled(True)
        self._connect_btn.setEnabled(False)
        self._start_acquisition()

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
        logger.info("수집 시작: DAQ=%s, 채널=%s", type(self._daq).__name__, self._enabled_indices)

    def _stop_acquisition(self, manual: bool = True):
        # 수동 정지(버튼/종료)는 자동 재시작 사이클도 함께 취소한다.
        # 오류 정지(manual=False)는 호출자가 재시작을 예약하므로 취소하지 않는다.
        if manual:
            self._restart_timer.stop()
            self._stable_timer.stop()
            self._restart_attempts = 0
        if self._worker:
            self._worker.stop()
            self._worker = None
        if self._collector:
            self._collector.stop()
        self._start_btn.setText("▶ 시작")
        logger.info("수집 정지%s", "" if manual else " (오류)")

    def _on_data_ready(self, data: np.ndarray):
        # 활성 채널만 필터링
        filtered = data[self._enabled_indices, :]
        self._last_data = filtered

        # ① 핵심 경로 우선: 수집기에 먼저 공급한다. 플롯 그리기(pyqtgraph)가
        #    Python 3.14+SIP 비호환으로 실패해도 저장·이상감지 데이터 흐름이
        #    끊기지 않도록 보장한다. (플롯이 먼저면 실패 시 collector 미공급 → 저장 중단)
        self._collector.on_data_ready(filtered)

        # ② FFT 계산(numpy, 안전) — CSV 저장 버튼용 캐시 갱신
        sample_rate = self._active_sample_rate or float(self._sample_rate_edit.text())
        freqs_list, mags_list = [], []
        for ch_data in filtered:
            freqs, mags = compute_fft(ch_data, sample_rate)
            freqs_list.append(freqs)
            mags_list.append(mags)
        self._last_freqs = freqs_list
        self._last_mags = mags_list

        # ③ 플롯 갱신 — 표시 전용. 실패해도 슬롯 밖으로 전파되지 않게 보호.
        self._safe_plot(lambda: self._waveform_plot.update(filtered, self._enabled_indices), "waveform")
        self._safe_plot(lambda: self._fft_plot.update(freqs_list, mags_list, self._enabled_indices), "fft")

    def _on_state_changed(self, states):
        self._safe_plot(lambda: self._status_light.update_states(states, self._enabled_indices), "status")
        if self._detector:
            self._safe_plot(
                lambda: self._anomaly_plot.update(self._detector.if_scores(), self._enabled_indices),
                "anomaly",
            )

    def _safe_plot(self, fn, name: str) -> None:
        """플롯 갱신을 보호 실행. pyqtgraph TypeError(Python 3.14+SIP)가
        Qt 슬롯 밖으로 전파돼 앱이 죽는 것을 막는다. 동일 경고는 1회만 기록."""
        try:
            fn()
        except Exception as exc:
            if name not in self._plot_warned:
                self._plot_warned.add(name)
                logger.warning("플롯 '%s' 갱신 실패 (이후 동일 경고 생략, 데이터 수집은 계속): %s",
                               name, exc)

    def _on_error(self, msg: str):
        logger.error("DAQ 수집 오류 발생: %s", msg)
        # ① 정지를 먼저 실행한다. (모달창이 정지 로직을 막아 멈춘 버퍼가
        #    계속 저장되던 버그 방지 — 무인 운전 시 치명적)
        self._stop_acquisition(manual=False)
        # ② 알림은 비모달로 표시한다. exec_()/모달은 이벤트 루프를 블록하여
        #    무인 운전 중 OK 클릭이 없으면 앱 로직이 영구 정지된다.
        self._show_error_nonmodal(msg)
        # ③ 자동 재시작 예약 (초기 실행조건 그대로)
        if self._auto_restart_enabled and self._start_config is not None:
            self._schedule_restart()

    def _show_error_nonmodal(self, msg: str) -> None:
        """비차단(non-modal) 오류 알림. 반복 오류는 기존 창 텍스트만 갱신."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full = f"[{ts}] 수집이 중단되었습니다.\n\n{msg}"
        box = getattr(self, "_error_box", None)
        if box is None:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle("수집 오류")
            box.setStandardButtons(QMessageBox.Ok)
            box.setModal(False)  # 비모달 — 이벤트 루프를 막지 않음
            self._error_box = box
        box.setText(full)
        box.show()
        box.raise_()

    # ── 자동 재시작 (워치독) ─────────────────────────────────────────────────

    def _on_auto_restart_toggled(self, checked: bool) -> None:
        self._auto_restart_enabled = checked
        if not checked:
            self._restart_timer.stop()
        logger.info("자동 재시작 %s", "활성화" if checked else "비활성화")

    def _schedule_restart(self) -> None:
        """지수 백오프로 재시작을 예약한다 (상한 RESTART_MAX_SEC)."""
        self._restart_attempts += 1
        delay = min(RESTART_BASE_SEC * (2 ** (self._restart_attempts - 1)), RESTART_MAX_SEC)
        logger.warning("[자동재시작] %d초 후 재시도 예약 (누적 시도 #%d)",
                       delay, self._restart_attempts)
        self._restart_timer.start(int(delay * 1000))

    def _attempt_restart(self) -> None:
        """초기 실행조건 스냅샷으로 재연결 후 수집을 재개한다."""
        if not self._auto_restart_enabled or self._start_config is None:
            return
        logger.info("[자동재시작] 시도 #%d — 초기 실행조건으로 재연결", self._restart_attempts)
        try:
            self._build_pipeline(self._start_config)
        except Exception as exc:
            self._consecutive_restart_failures += 1
            logger.warning("[자동재시작] 재연결 실패 (#%d, 연속 %d/%d): %s",
                           self._restart_attempts, self._consecutive_restart_failures,
                           MAX_INPROC_RESTART_FAILS, exc)
            # 프로세스 내 복구가 반복 실패 = Python 3.14+SIP 상태 손상으로
            # 같은 프로세스에서 회복 불가. 종료하여 supervisor가 새 프로세스로 재시작.
            if self._consecutive_restart_failures >= MAX_INPROC_RESTART_FAILS:
                logger.critical(
                    "[자동재시작] 프로세스 내 복구 %d회 연속 실패 — "
                    "새 프로세스 재시작 필요. 종료(code=%d). "
                    "(supervisor 미사용 시 수동 재시작 요망)",
                    MAX_INPROC_RESTART_FAILS, EXIT_NEEDS_RESTART,
                )
                app = QApplication.instance()
                if app is not None:
                    app.exit(EXIT_NEEDS_RESTART)
                return
            self._schedule_restart()
            return
        self._start_acquisition()
        self._consecutive_restart_failures = 0
        logger.info("[자동재시작] 성공 (#%d). %d초 무오류 시 카운터 초기화",
                    self._restart_attempts, STABLE_RESET_SEC)
        self._stable_timer.start(STABLE_RESET_SEC * 1000)

    def _on_stable(self) -> None:
        if self._restart_attempts:
            logger.info("[자동재시작] %d초 안정 운전 확인 — 재시도 카운터 초기화",
                        STABLE_RESET_SEC)
        self._restart_attempts = 0
        self._consecutive_restart_failures = 0

    def _on_save_csv(self):
        if self._last_data is None:
            QMessageBox.warning(self, "저장 실패", "저장할 데이터가 없습니다.")
            return
        ts = datetime.now()
        sample_rate = self._active_sample_rate or float(self._sample_rate_edit.text())
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
