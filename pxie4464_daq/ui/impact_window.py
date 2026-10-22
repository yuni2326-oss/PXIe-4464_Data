"""임팩트 해머 시험 창 (EX2 계획서 §1·§4·§6).

절차: 마이크 고정 + 해머 이동. 위치마다 목표 횟수(기본 10회) 타격 → '다음 위치'.
타격마다 [프리트리거 + 기록 길이] 레코드를 .npz로 저장하고 impact_log.csv에 기록,
NAS(하위폴더 미러링)로 백그라운드 업로드한다.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)

from pxie4464_daq.acquisition.impact_worker import ImpactWorker
from pxie4464_daq.analysis.impact import (
    TriggerCapture, analyze_pulse, classify, db20, force_bandwidths, force_spectrum,
    frf_add, frf_coherence, frf_h1, frf_new, frf_remove_last, hit_spectra,
    overload_rows, pick_peaks,
)
from pxie4464_daq.device.daq import PXIe4353
from pxie4464_daq.device.impact_daq import (
    KIND_ACCEL, KIND_FORCE, KIND_MIC, ImpactDAQ, MockImpactDAQ, clip_level, make_spec,
)
from pxie4464_daq.device.temp_monitor import TemperatureMonitor
from pxie4464_daq.storage import nas_uploader
from pxie4464_daq.storage.impact_saver import LOG_NAME, ImpactLog, hit_filename, save_hit

logger = logging.getLogger(__name__)

TOOL_VERSION = "pxie4464_daq impact v1"
# C:\Project\Data_aquistion\results (작업 디렉터리와 무관)
RESULTS_ROOT_DEFAULT = Path(__file__).resolve().parents[3] / "results"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "impact_session.json"
DEFAULT_NAS_DIR = r"\\10.130.121.158\ai_gpu_001\GY_DATA\pump data"   # 상시 모니터링 앱과 동일

SAMPLE_RATES = [51200, 102400, 204800]
TC_TYPES = ["J", "K", "T", "E", "N", "R", "S", "B"]
TEMP_INTERVAL_S = 10.0

# 계획서 §4 가진점: 뒤판 바깥면 베인 용접점 6베인 × ξ0.3·0.6, 앞판 베인선 ξ0.6 6곳, 흡입구 테.
# §6 E1-C 쿠폰: 클램프 블록 타격.
LOCATION_PRESETS = (
    [f"back_vane{v}_xi0.3" for v in range(1, 7)]
    + [f"back_vane{v}_xi0.6" for v in range(1, 7)]
    + [f"front_vane{v}_xi0.6" for v in range(1, 7)]
    + ["inlet_rim", "coupon_clamp_block"]
)
MIC_PRESETS = ["M1", "M2", "M3", "coupon_tip_center", "coupon_tip_edge"]
HAMMER_TIPS = ["", "steel", "aluminum", "nylon", "rubber"]

HELP_TEXT = (
    "절차(계획서 §4): ① 임펠러 자유-자유 매달기 ② 마이크 고정(판에서 5±0.5 mm) "
    "③ 위치 선택 → 같은 위치 목표 횟수 타격, 타격 간격은 기록 길이(4 s) 이상 "
    "④ '다음 위치 ▶' ⑤ 세션마다 다시 매달고 세션 번호를 올린다. "
    "앞판 베이는 치지 않는다(다중 접촉). 상시 모니터링 프로그램은 먼저 종료할 것."
)


class ImpactWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("임팩트 해머 시험 — LC-00KE (CH7) · 마이크 (CH6)")
        self._worker: Optional[ImpactWorker] = None
        self._capture: Optional[TriggerCapture] = None
        self._specs: List[dict] = []
        self._resp_rows: List[int] = []
        self._clip: List[float] = []
        self._run_cfg: dict = {}
        self._folder: Optional[Path] = None
        self._nas_dir: Optional[Path] = None
        self._log: Optional[ImpactLog] = None
        self._frf: Optional[dict] = None
        self._frf_key = None
        self._last: Optional[dict] = None          # 마지막 타격(그래프 다시 그리기용)
        self._last_saved: Optional[dict] = None    # 직전 타격 제외용
        self._pending = deque()                    # 트리거 순간의 실험정보 스냅샷
        self._acq_start: Optional[datetime] = None
        self._last_temp: List[float] = []
        self._last_temp_t = 0.0
        self._temp_channels: List[int] = []
        self._plot_warned = set()
        self._temp_monitor = TemperatureMonitor(self)
        self._temp_monitor.reading.connect(self._on_temperature)
        self._temp_monitor.failed.connect(self._on_temp_error)
        self._setup_ui()
        self._load_settings()
        self._update_location_status()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(370)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setSpacing(6)
        v.setContentsMargins(6, 6, 6, 6)
        v.addWidget(self._make_action_group())
        v.addWidget(self._make_experiment_group())
        v.addWidget(self._make_device_group())
        v.addWidget(self._make_trigger_group())
        v.addWidget(self._make_save_group())
        v.addWidget(self._make_temp_group())
        help_lbl = QLabel(HELP_TEXT)
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color: gray;")
        v.addWidget(help_lbl)
        v.addStretch()
        scroll.setWidget(panel)
        root.addWidget(scroll)

        right = QVBoxLayout()
        top = QHBoxLayout()
        self._status = QLabel("정지됨 — '▶ 시작'을 누르면 트리거 대기")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size: 13pt; font-weight: bold;")
        top.addWidget(self._status, stretch=1)
        top.addWidget(QLabel("표시 응답 채널"))
        self._resp_combo = QComboBox()
        self._resp_combo.currentIndexChanged.connect(lambda _i: self._redraw())
        top.addWidget(self._resp_combo)
        right.addLayout(top)

        grid = QGridLayout()
        self._p_force = pg.PlotWidget(title="해머 힘 — 트리거 주변")
        self._p_force.setLabel("bottom", "시간", "ms")
        self._p_force.setLabel("left", "힘", "N")
        self._p_resp = pg.PlotWidget(title="응답 시간이력")
        self._p_resp.setLabel("bottom", "시간(트리거 기준)", "s")
        self._p_resp.setDownsampling(auto=True, mode="peak")
        self._p_resp.setClipToView(True)
        self._p_fspec = pg.PlotWidget(title="힘 스펙트럼 (dB, 저주파 기준)")
        self._p_fspec.setLabel("bottom", "주파수", "Hz")
        self._p_frf = pg.PlotWidget(title="FRF |H1|")
        self._p_frf.setLabel("bottom", "주파수", "Hz")
        self._p_frf.setLabel("left", "dB")
        self._p_frf.setDownsampling(auto=True, mode="peak")
        self._p_frf.setClipToView(True)
        self._p_coh = pg.PlotWidget(title="코히어런스 (같은 위치 평균)")
        self._p_coh.setYRange(0.0, 1.05)
        self._p_coh.setXLink(self._p_frf)
        self._p_coh.setDownsampling(auto=True, mode="peak")
        self._p_coh.setClipToView(True)
        self._peaks_text = QPlainTextEdit()
        self._peaks_text.setReadOnly(True)
        self._peaks_text.setStyleSheet("font-family: Consolas, monospace;")
        grid.addWidget(self._p_force, 0, 0)
        grid.addWidget(self._p_resp, 0, 1)
        grid.addWidget(self._p_fspec, 1, 0)
        grid.addWidget(self._p_frf, 1, 1)
        grid.addWidget(self._peaks_text, 2, 0)
        grid.addWidget(self._p_coh, 2, 1)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 3)
        grid.setRowStretch(2, 2)
        right.addLayout(grid, stretch=5)

        self._hit_text = QPlainTextEdit()
        self._hit_text.setReadOnly(True)
        self._hit_text.setMaximumBlockCount(500)
        self._hit_text.setStyleSheet("font-family: Consolas, monospace;")
        self._hit_text.setPlaceholderText("타격 기록")
        right.addWidget(self._hit_text, stretch=1)
        root.addLayout(right, stretch=1)

        # 그래프 항목은 여기(슬롯 밖)에서 한 번만 만들고, 슬롯에서는 갱신만 한다.
        self._c_force = self._p_force.plot(pen=pg.mkPen("y", width=1.5))
        self._l_thr_pos = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("r", style=Qt.DashLine))
        self._l_thr_neg = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("r", style=Qt.DashLine))
        self._p_force.addItem(self._l_thr_pos)
        self._p_force.addItem(self._l_thr_neg)
        self._c_resp = self._p_resp.plot(pen=pg.mkPen("c", width=1))
        self._c_fspec = self._p_fspec.plot(pen=pg.mkPen("y", width=1.5))
        self._l_bw = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("g", style=Qt.DashLine))
        self._p_fspec.addItem(self._l_bw)
        for lvl in (-10.0, -20.0):
            self._p_fspec.addItem(pg.InfiniteLine(pos=lvl, angle=0, movable=False,
                                                  pen=pg.mkPen((120, 120, 120), style=Qt.DotLine)))
        self._c_frf_hit = self._p_frf.plot(pen=pg.mkPen((140, 140, 140), width=1))
        self._c_frf_avg = self._p_frf.plot(pen=pg.mkPen("c", width=1.5))
        self._s_peaks = pg.ScatterPlotItem(size=9, pen=pg.mkPen("r"), brush=pg.mkBrush(255, 60, 60, 140))
        self._p_frf.addItem(self._s_peaks)
        self._c_coh = self._p_coh.plot(pen=pg.mkPen("g", width=1))

    def _form(self, title: str) -> tuple:
        group = QGroupBox(title)
        lay = QGridLayout(group)
        return group, lay

    def _row(self, lay: QGridLayout, label: str, widget) -> None:
        r = lay.rowCount()
        lay.addWidget(QLabel(label), r, 0)
        lay.addWidget(widget, r, 1)

    def _make_action_group(self) -> QGroupBox:
        group, lay = self._form("실행")
        self._start_btn = QPushButton("▶ 시작 (트리거 대기)")
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn = QPushButton("■ 정지")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        self._reject_btn = QPushButton("직전 타격 제외")
        self._reject_btn.setEnabled(False)
        self._reject_btn.clicked.connect(self._on_reject)
        lay.addWidget(self._start_btn, 0, 0, 1, 2)
        lay.addWidget(self._stop_btn, 1, 0)
        lay.addWidget(self._reject_btn, 1, 1)
        self._level_label = QLabel("레벨: —")
        self._level_label.setWordWrap(True)
        lay.addWidget(self._level_label, 2, 0, 1, 2)
        return group

    def _make_experiment_group(self) -> QGroupBox:
        group, lay = self._form("실험 정보 (타격 순간 값으로 기록)")
        self._session_spin = QSpinBox()
        self._session_spin.setRange(1, 99)
        self._row(lay, "세션", self._session_spin)
        self._loc_combo = QComboBox()
        self._loc_combo.setEditable(True)
        self._loc_combo.addItems(LOCATION_PRESETS)
        self._row(lay, "타격 위치", self._loc_combo)
        self._next_btn = QPushButton("다음 위치 ▶")
        self._next_btn.clicked.connect(self._on_next_location)
        lay.addWidget(self._next_btn, lay.rowCount(), 1)
        self._mic_pos_combo = QComboBox()
        self._mic_pos_combo.setEditable(True)
        self._mic_pos_combo.addItems(MIC_PRESETS)
        self._row(lay, "마이크 위치", self._mic_pos_combo)
        self._target_spin = QSpinBox()
        self._target_spin.setRange(1, 999)
        self._target_spin.setValue(10)
        self._row(lay, "위치당 목표 타격", self._target_spin)
        self._notes_edit = QLineEdit("")
        self._row(lay, "메모", self._notes_edit)
        self._loc_status = QLabel("")
        self._loc_status.setWordWrap(True)
        self._loc_status.setStyleSheet("font-weight: bold;")
        lay.addWidget(self._loc_status, lay.rowCount(), 0, 1, 2)
        self._session_spin.valueChanged.connect(lambda _v: self._update_location_status())
        self._loc_combo.currentTextChanged.connect(lambda _t: self._update_location_status())
        self._mic_pos_combo.currentTextChanged.connect(lambda _t: self._update_location_status())
        self._target_spin.valueChanged.connect(lambda _v: self._update_location_status())
        return group

    def _make_device_group(self) -> QGroupBox:
        group, lay = self._form("장치 · 채널 (CH0–3=4464, CH4–11=4492)")
        self._mock_check = QCheckBox("Mock 모드 (하드웨어 없이 확인)")
        lay.addWidget(self._mock_check, 0, 0, 1, 2)
        self._dev4492_edit = QLineEdit("PXI1Slot5")
        self._row(lay, "PXIe-4492 장치명", self._dev4492_edit)
        self._dev4464_edit = QLineEdit("PXI1Slot3")
        self._row(lay, "PXIe-4464 장치명", self._dev4464_edit)
        self._fs_combo = QComboBox()
        for r in SAMPLE_RATES:
            self._fs_combo.addItem(f"{r:,} S/s", r)
        self._fs_combo.setCurrentIndex(SAMPLE_RATES.index(204800))
        self._row(lay, "샘플레이트", self._fs_combo)
        self._iepe_edit = QLineEdit("4")
        self._row(lay, "IEPE 전류 (mA)", self._iepe_edit)

        lay.addWidget(QLabel("── 해머 (IEPE 힘) ──"), lay.rowCount(), 0, 1, 2)
        self._ham_ch = QSpinBox()
        self._ham_ch.setRange(0, 11)
        self._ham_ch.setValue(7)
        self._row(lay, "채널 (CH)", self._ham_ch)
        self._ham_sens = QLineEdit("20.348")
        self._row(lay, "감도 (mV/N)", self._ham_sens)
        self._ham_max = QLineEdit("200")
        self._row(lay, "최대 힘 (N)", self._ham_max)
        self._ham_model = QLineEdit("LC-00KE")
        self._row(lay, "모델", self._ham_model)
        self._ham_tip = QComboBox()
        self._ham_tip.setEditable(True)
        self._ham_tip.addItems(HAMMER_TIPS)
        self._row(lay, "팁", self._ham_tip)

        lay.addWidget(QLabel("── 마이크 (주 응답) ──"), lay.rowCount(), 0, 1, 2)
        self._mic_use = QCheckBox("사용")
        self._mic_use.setChecked(True)
        lay.addWidget(self._mic_use, lay.rowCount(), 0, 1, 2)
        self._mic_ch = QSpinBox()
        self._mic_ch.setRange(0, 11)
        self._mic_ch.setValue(6)
        self._row(lay, "채널 (CH)", self._mic_ch)
        self._mic_sens = QLineEdit("50")
        self._row(lay, "감도 (mV/Pa)", self._mic_sens)
        self._mic_spl = QLineEdit("130")
        self._row(lay, "최대 SPL (dB)", self._mic_spl)

        lay.addWidget(QLabel("── 가속도계 (기준·틀 감시) ──"), lay.rowCount(), 0, 1, 2)
        self._acc_chs = QLineEdit("4, 5")
        self._acc_chs.setPlaceholderText("비우면 없음. CH0–3 포함 시 4464 함께 동기 수집")
        self._row(lay, "채널 (CH, 쉼표)", self._acc_chs)
        self._acc_sens = QLineEdit("10.2")
        self._row(lay, "감도 (mV/(m/s²))", self._acc_sens)
        self._acc_max = QLineEdit("490")
        self._row(lay, "최대 (m/s²)", self._acc_max)
        return group

    def _make_trigger_group(self) -> QGroupBox:
        group, lay = self._form("트리거 · 분석")
        self._thr_edit = QLineEdit("2.0")
        self._row(lay, "트리거 임계 (N)", self._thr_edit)
        self._pre_edit = QLineEdit("20")
        self._row(lay, "프리트리거 (ms)", self._pre_edit)
        self._post_edit = QLineEdit("4.0")
        self._row(lay, "기록 길이 (s/타격)", self._post_edit)
        self._fmax_edit = QLineEdit("20000")
        self._row(lay, "분석 상한 (Hz)", self._fmax_edit)
        self._expwin_check = QCheckBox("FRF 응답 지수창 (끝 1 %)")
        self._expwin_check.setChecked(True)
        lay.addWidget(self._expwin_check, lay.rowCount(), 0, 1, 2)
        return group

    def _make_save_group(self) -> QGroupBox:
        group, lay = self._form("저장 · NAS")
        self._root_edit = QLineEdit(str(RESULTS_ROOT_DEFAULT))
        self._row(lay, "저장 루트", self._root_edit)
        self._subdir_edit = QLineEdit("impact_test")
        self._row(lay, "하위폴더", self._subdir_edit)
        self._nas_edit = QLineEdit(DEFAULT_NAS_DIR)
        self._nas_edit.setPlaceholderText("비우면 NAS 전송 안 함")
        self._row(lay, "NAS 경로", self._nas_edit)
        return group

    def _make_temp_group(self) -> QGroupBox:
        group, lay = self._form("온도 (PXIe-4353, 타격마다 기록)")
        self._temp_use = QCheckBox("온도 기록")
        self._temp_use.setChecked(True)
        lay.addWidget(self._temp_use, 0, 0, 1, 2)
        self._temp_dev = QLineEdit("PXI1Slot8")
        self._row(lay, "장치명", self._temp_dev)
        self._temp_chs = QLineEdit("0, 1")
        self._row(lay, "채널 (ai)", self._temp_chs)
        self._temp_tc = QComboBox()
        for t in TC_TYPES:
            self._temp_tc.addItem(f"{t}형", t)
        self._row(lay, "열전대", self._temp_tc)
        self._temp_label = QLabel("온도: —")
        lay.addWidget(self._temp_label, lay.rowCount(), 0, 1, 2)
        return group

    # ── 설정 ────────────────────────────────────────────────────────────────

    def _ui_values(self) -> dict:
        return {
            "mock": self._mock_check.isChecked(),
            "dev4492": self._dev4492_edit.text().strip(),
            "dev4464": self._dev4464_edit.text().strip(),
            "fs": int(self._fs_combo.currentData()),
            "iepe_mA": self._iepe_edit.text().strip(),
            "ham_ch": self._ham_ch.value(),
            "ham_sens": self._ham_sens.text().strip(),
            "ham_max": self._ham_max.text().strip(),
            "ham_model": self._ham_model.text().strip(),
            "ham_tip": self._ham_tip.currentText().strip(),
            "mic_use": self._mic_use.isChecked(),
            "mic_ch": self._mic_ch.value(),
            "mic_sens": self._mic_sens.text().strip(),
            "mic_spl": self._mic_spl.text().strip(),
            "acc_chs": self._acc_chs.text().strip(),
            "acc_sens": self._acc_sens.text().strip(),
            "acc_max": self._acc_max.text().strip(),
            "threshold": self._thr_edit.text().strip(),
            "pre_ms": self._pre_edit.text().strip(),
            "post_s": self._post_edit.text().strip(),
            "fmax": self._fmax_edit.text().strip(),
            "expwin": self._expwin_check.isChecked(),
            "root": self._root_edit.text().strip(),
            "subdir": self._subdir_edit.text().strip(),
            "nas": self._nas_edit.text().strip(),
            "session": self._session_spin.value(),
            "location": self._loc_combo.currentText().strip(),
            "mic_position": self._mic_pos_combo.currentText().strip(),
            "target": self._target_spin.value(),
            "notes": self._notes_edit.text(),
            "temp_use": self._temp_use.isChecked(),
            "temp_dev": self._temp_dev.text().strip(),
            "temp_chs": self._temp_chs.text().strip(),
            "temp_tc": self._temp_tc.currentData(),
        }

    def _apply_values(self, d: dict) -> None:
        def s(w, key):
            if key in d and d[key] is not None:
                w.setText(str(d[key]))

        def combo_text(w, key):
            if d.get(key):
                i = w.findText(str(d[key]))
                if i >= 0:
                    w.setCurrentIndex(i)
                else:
                    w.setEditText(str(d[key]))

        self._mock_check.setChecked(bool(d.get("mock", False)))
        s(self._dev4492_edit, "dev4492")
        s(self._dev4464_edit, "dev4464")
        if d.get("fs") in SAMPLE_RATES:
            self._fs_combo.setCurrentIndex(SAMPLE_RATES.index(d["fs"]))
        s(self._iepe_edit, "iepe_mA")
        if "ham_ch" in d:
            self._ham_ch.setValue(int(d["ham_ch"]))
        s(self._ham_sens, "ham_sens")
        s(self._ham_max, "ham_max")
        s(self._ham_model, "ham_model")
        combo_text(self._ham_tip, "ham_tip")
        self._mic_use.setChecked(bool(d.get("mic_use", True)))
        if "mic_ch" in d:
            self._mic_ch.setValue(int(d["mic_ch"]))
        s(self._mic_sens, "mic_sens")
        s(self._mic_spl, "mic_spl")
        s(self._acc_chs, "acc_chs")
        s(self._acc_sens, "acc_sens")
        s(self._acc_max, "acc_max")
        s(self._thr_edit, "threshold")
        s(self._pre_edit, "pre_ms")
        s(self._post_edit, "post_s")
        s(self._fmax_edit, "fmax")
        self._expwin_check.setChecked(bool(d.get("expwin", True)))
        s(self._root_edit, "root")
        s(self._subdir_edit, "subdir")
        s(self._nas_edit, "nas")
        if "session" in d:
            self._session_spin.setValue(int(d["session"]))
        combo_text(self._loc_combo, "location")
        combo_text(self._mic_pos_combo, "mic_position")
        if "target" in d:
            self._target_spin.setValue(int(d["target"]))
        s(self._notes_edit, "notes")
        self._temp_use.setChecked(bool(d.get("temp_use", True)))
        s(self._temp_dev, "temp_dev")
        s(self._temp_chs, "temp_chs")
        i = self._temp_tc.findData(d.get("temp_tc", "J"))
        if i >= 0:
            self._temp_tc.setCurrentIndex(i)

    def _load_settings(self) -> None:
        try:
            self._apply_values(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("임팩트 설정 읽기 실패(기본값 사용): %s", exc)

    def _save_settings(self) -> None:
        try:
            CONFIG_PATH.parent.mkdir(exist_ok=True)
            CONFIG_PATH.write_text(json.dumps(self._ui_values(), ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        except Exception as exc:
            logger.warning("임팩트 설정 저장 실패: %s", exc)

    @staticmethod
    def _ints(text: str) -> List[int]:
        out = []
        for tok in (text or "").replace(",", " ").split():
            out.append(int(tok))
        return out

    def _parse(self) -> dict:
        """UI 값 검증·변환 → 실행 설정."""
        u = self._ui_values()
        c = dict(u)
        c["iepe_mA"] = float(u["iepe_mA"] or 4.0)
        c["ham_sens"] = float(u["ham_sens"])
        c["ham_max"] = float(u["ham_max"])
        c["mic_sens"] = float(u["mic_sens"])
        c["mic_spl"] = float(u["mic_spl"])
        c["acc_chs"] = self._ints(u["acc_chs"])
        c["acc_sens"] = float(u["acc_sens"] or 10.2)
        c["acc_max"] = float(u["acc_max"] or 490.0)
        c["threshold"] = float(u["threshold"])
        c["pre_s"] = float(u["pre_ms"]) / 1000.0
        c["post_s"] = float(u["post_s"])
        c["fmax"] = min(float(u["fmax"]), c["fs"] / 2.0)
        c["temp_chs"] = self._ints(u["temp_chs"])
        if c["ham_sens"] <= 0 or c["ham_max"] <= 0 or c["threshold"] <= 0:
            raise ValueError("해머 감도·최대 힘·트리거 임계는 0보다 커야 합니다.")
        if c["threshold"] >= c["ham_max"]:
            raise ValueError("트리거 임계가 최대 힘보다 작아야 합니다.")
        if c["post_s"] <= 0 or c["pre_s"] < 0:
            raise ValueError("기록 길이·프리트리거를 확인하세요.")
        if not u["subdir"]:
            raise ValueError("하위폴더를 지정하세요 (예: E2I_S1).")
        return c

    def _build_specs(self, c: dict) -> List[dict]:
        common = dict(dev4464=c["dev4464"], dev4492=c["dev4492"], iepe=True,
                      excit_current=c["iepe_mA"] / 1000.0)
        specs = [make_spec(c["ham_ch"], KIND_FORCE, c["ham_sens"], c["ham_max"],
                           label="hammer", **common)]
        if c["mic_use"]:
            specs.append(make_spec(c["mic_ch"], KIND_MIC, c["mic_sens"], c["mic_spl"],
                                   label="mic", **common))
        for ch in sorted(set(c["acc_chs"])):
            specs.append(make_spec(ch, KIND_ACCEL, c["acc_sens"], c["acc_max"],
                                   label="acc", **common))
        chs = [s["ch"] for s in specs]
        if len(set(chs)) != len(chs):
            raise ValueError(f"채널이 겹칩니다: {chs} (해머·마이크·가속도계는 서로 다른 CH)")
        return specs

    # ── 실행 ────────────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        if self._worker is not None:
            return
        try:
            c = self._parse()
            specs = self._build_specs(c)
        except Exception as exc:
            self._set_status(f"설정 오류: {exc}", "error")
            return

        root = Path(c["root"]) if c["root"] else RESULTS_ROOT_DEFAULT
        self._folder = root / c["subdir"]
        try:
            self._folder.mkdir(parents=True, exist_ok=True)
            self._log = ImpactLog(self._folder / LOG_NAME)
        except Exception as exc:
            self._set_status(f"저장 폴더를 만들 수 없습니다: {exc}", "error")
            return
        self._nas_dir = (Path(c["nas"]) / c["subdir"]) if c["nas"] else None

        fs = float(c["fs"])
        try:
            if c["mock"]:
                daq = MockImpactDAQ(specs, sample_rate=fs, hit_interval_s=c["post_s"] + 2.0)
            else:
                daq = ImpactDAQ(specs, sample_rate=fs)
        except Exception as exc:
            self._set_status(f"장치 구성 실패: {exc}", "error")
            return

        self._specs = specs
        self._resp_rows = list(range(1, len(specs)))
        self._clip = [clip_level(s) for s in specs]
        self._run_cfg = c
        self._capture = TriggerCapture(len(specs), fs, trig_row=0, threshold=c["threshold"],
                                       pre_s=c["pre_s"], post_s=c["post_s"])
        level_rows = [0] + ([1] if c["mic_use"] else [])
        self._worker = ImpactWorker(daq, self._capture, level_rows)
        self._worker.hit_captured.connect(self._on_hit)
        self._worker.triggered.connect(self._on_triggered)
        self._worker.level.connect(self._on_level)
        self._worker.error_occurred.connect(self._on_error)

        self._resp_combo.blockSignals(True)
        self._resp_combo.clear()
        for r in self._resp_rows:
            self._resp_combo.addItem(f"{specs[r]['name']} [{specs[r]['unit']}]", r)
        self._resp_combo.blockSignals(False)
        self._frf = None
        self._frf_key = None
        self._last = None
        self._last_saved = None
        self._pending.clear()
        self._reject_btn.setEnabled(False)
        self._l_thr_pos.setValue(c["threshold"])
        self._l_thr_neg.setValue(-c["threshold"])
        self._acq_start = datetime.now()

        self._worker.start()
        self._start_temperature(c)
        self._save_settings()
        self._set_inputs_enabled(False)
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        devs = sorted({s["physical"].split("/")[0] for s in specs})
        self._set_status(f"트리거 대기 중 — 해머 |F| ≥ {c['threshold']:g} N 이면 기록 "
                         f"({c['post_s']:g} s). 저장: {self._folder}", "ok")
        logger.info("임팩트 시험 시작: %s, %.0f S/s, 장치 %s, 폴더 %s, NAS %s%s",
                    [s["name"] for s in specs], fs, devs, self._folder,
                    self._nas_dir or "(없음)", " [MOCK]" if c["mock"] else "")
        self._update_location_status()

    def _start_temperature(self, c: dict) -> None:
        self._last_temp = []
        self._last_temp_t = 0.0
        self._temp_channels = c["temp_chs"]
        if not c["temp_use"] or c["mock"] or not c["temp_chs"]:
            self._temp_label.setText("온도: (기록 안 함)")
            return
        try:
            tdaq = PXIe4353(device_name=c["temp_dev"], channels=c["temp_chs"], tc_type=c["temp_tc"])
            self._temp_monitor.start(tdaq, TEMP_INTERVAL_S)
            self._temp_label.setText("온도: 측정 대기…")
        except Exception as exc:
            self._temp_label.setText(f"온도: 시작 실패 ({exc})")
            logger.warning("온도 모니터 시작 실패: %s", exc)

    def _on_stop(self) -> None:
        self._stop_acquisition("정지됨")

    def _stop_acquisition(self, msg: str, level: str = "ok") -> None:
        if self._worker is not None:
            try:
                self._worker.stop()
            except Exception as exc:
                logger.warning("워커 정지 오류(무시): %s", exc)
            self._worker = None
        try:
            self._temp_monitor.stop()
        except Exception:
            pass
        self._pending.clear()
        self._set_inputs_enabled(True)
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._level_label.setText("레벨: —")
        self._set_status(msg, level)
        logger.info("임팩트 시험 정지: %s", msg)

    def _set_inputs_enabled(self, on: bool) -> None:
        # 기록 중에도 바꿀 수 있어야 하는 것: 세션·위치·마이크 위치·목표·메모·표시 채널
        for w in (self._mock_check, self._dev4492_edit, self._dev4464_edit, self._fs_combo,
                  self._iepe_edit, self._ham_ch, self._ham_sens, self._ham_max, self._ham_model,
                  self._ham_tip, self._mic_use, self._mic_ch, self._mic_sens, self._mic_spl,
                  self._acc_chs, self._acc_sens, self._acc_max, self._thr_edit, self._pre_edit,
                  self._post_edit, self._fmax_edit, self._expwin_check, self._root_edit,
                  self._subdir_edit, self._nas_edit, self._temp_use, self._temp_dev,
                  self._temp_chs, self._temp_tc):
            w.setEnabled(on)

    def _on_error(self, msg: str) -> None:
        hint = ""
        low = msg.lower()
        if "reserved" in low or "-50103" in msg:
            hint = " → 다른 프로그램(상시 모니터링 등)이 장치를 쓰고 있습니다. 먼저 종료하세요."
        elif "cannot be accessed" in low or "-201003" in msg:
            hint = " → 장치 연결·전원을 확인하세요."
        elif any(s["ch"] < 4 for s in self._specs):
            hint = " → CH0–3(4464)을 빼고 4492 채널만으로 다시 시도해 보세요."
        logger.error("임팩트 수집 오류: %s", msg)
        self._stop_acquisition(f"수집 오류로 정지: {msg.splitlines()[0] if msg else msg}{hint}", "error")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("수집 오류")
        box.setText(msg + ("\n\n" + hint.strip(" →") if hint else ""))
        box.setModal(False)      # 비모달 — 이벤트 루프를 막지 않는다
        box.show()

    # ── 수집 신호 ────────────────────────────────────────────────────────────

    def _snapshot(self) -> dict:
        return {
            "session": self._session_spin.value(),
            "location": self._loc_combo.currentText().strip() or "unspecified",
            "mic_position": self._mic_pos_combo.currentText().strip() or "-",
            "notes": self._notes_edit.text(),
        }

    def _on_triggered(self, _t) -> None:
        self._pending.append(self._snapshot())
        post = self._run_cfg.get("post_s", 4.0)
        self._set_status(f"● 타격 감지 — {post:g} s 기록 중 (끝날 때까지 치지 마세요)", "busy")

    def _on_level(self, d: dict) -> None:
        pk = d.get("peaks", [])
        txt = f"해머 {pk[0]:.2f} N" if pk else "해머 —"
        if len(pk) > 1:
            txt += f" · 마이크 {pk[1]:.3f} Pa"
        if d.get("capturing"):
            txt += f" · 기록 중 {d.get('progress_s', 0.0):.1f}/{self._run_cfg.get('post_s', 4.0):g} s"
        else:
            txt += " · 트리거 대기"
            thr = self._run_cfg.get("threshold", 0.0)
            if pk and thr and pk[0] > 0.5 * thr:
                txt += "  ⚠ 잡음이 임계의 50 % 이상 — 임계를 올리세요"
        self._level_label.setText(txt)

    def _fresh_temperature(self) -> List[float]:
        if self._last_temp and time.monotonic() - self._last_temp_t < 3 * TEMP_INTERVAL_S:
            return list(self._last_temp)
        return []

    def _on_hit(self, rec: dict) -> None:
        c = self._run_cfg
        info = self._pending.popleft() if self._pending else self._snapshot()
        data = rec["data"]
        fs = float(rec["sample_rate"])
        ti = int(rec["trigger_index"])
        ts = rec.get("trigger_time") or datetime.now()

        # ① 펄스·대역폭·품질 (가벼운 계산)
        pulse, ffreq, fmag, bw = None, None, None, {5.0: float("nan"), 10.0: float("nan"), 20.0: float("nan")}
        try:
            pulse = analyze_pulse(data[0], fs, ti, c["threshold"])
            ffreq, fmag = force_spectrum(data[0], fs, pulse["start"], pulse["end"])
            bw = force_bandwidths(ffreq, fmag)
        except Exception as exc:
            logger.exception("펄스 분석 실패: %s", exc)
        ovl = overload_rows(data, self._clip)
        quality = classify(pulse, ovl, bool(rec.get("pre_incomplete"))) if pulse else "analysis_error"
        status = "accepted" if quality == "ok" else "flagged"
        session, loc, mic = info["session"], info["location"], info["mic_position"]
        hit_no = self._log.next_hit_number(session, loc, mic) if self._log else 1
        temps = self._fresh_temperature()
        resp_row = 1 if len(self._specs) > 1 else None
        resp_peak = float(np.max(np.abs(data[resp_row, ti:]))) if resp_row is not None else float("nan")

        meta = {
            "tool": TOOL_VERSION, "mock": bool(c.get("mock")),
            "sample_rate": fs, "trigger_index": ti, "pretrigger_s": ti / fs,
            "record_s": c["post_s"], "threshold_N": c["threshold"],
            "channel_names": [s["name"] for s in self._specs],
            "channel_kinds": [s["kind"] for s in self._specs],
            "units": [s["unit"] for s in self._specs],
            "physical_channels": [s["physical"] for s in self._specs],
            "sensitivities": [s["sensitivity"] for s in self._specs],
            "sensitivity_units": [s["sens_unit"] for s in self._specs],
            "max_vals": [s["max_val"] for s in self._specs],
            "iepe_mA": c["iepe_mA"],
            "timestamp": ts, "trigger_abs_sample": int(rec.get("trigger_abs", 0)),
            "acq_start_time": self._acq_start,
            "session": int(session), "location": loc, "mic_position": mic,
            "hit": int(hit_no), "notes": info["notes"],
            "hammer_model": c["ham_model"], "hammer_sensitivity_mV_per_N": c["ham_sens"],
            "hammer_tip": c["ham_tip"],
            "temperature_C": temps, "temperature_channels": [f"ai{ch}" for ch in self._temp_channels],
            "peak_force_N": pulse["peak_abs"] if pulse else float("nan"),
            "pulse_width_ms": pulse["width_s"] * 1e3 if pulse else float("nan"),
            "pulse_start_index": pulse["start"] if pulse else -1,
            "pulse_end_index": pulse["end"] if pulse else -1,
            "bw_5dB_Hz": bw[5.0], "bw_10dB_Hz": bw[10.0], "bw_20dB_Hz": bw[20.0],
            "second_peak_N": pulse["second_peak"] if pulse else float("nan"),
            "second_peak_ms": pulse["second_peak_t"] * 1e3 if pulse else float("nan"),
            "double_hit": bool(pulse["double_hit"]) if pulse else False,
            "overload": bool(ovl), "overload_channels": [self._specs[i]["name"] for i in ovl],
            "pre_incomplete": bool(rec.get("pre_incomplete")),
            "quality": quality,
        }
        fname = hit_filename(ts, session, loc, mic, hit_no)

        # ② 원시 데이터 저장이 최우선
        try:
            path = save_hit(self._folder, fname, data, meta)
        except Exception as exc:
            logger.error("타격 저장 실패: %s", exc)
            self._set_status(f"⚠ 저장 실패: {exc}", "error")
            return
        if self._nas_dir is not None:
            nas_uploader.enqueue([path], self._nas_dir, delete_after=False)

        # ③ 로그
        try:
            self._log.append({
                "file": fname, "timestamp": ts.isoformat(timespec="milliseconds"),
                "session": session, "location": loc, "mic_position": mic, "hit": hit_no,
                "peak_force_N": meta["peak_force_N"], "pulse_width_ms": meta["pulse_width_ms"],
                "bw_5dB_Hz": bw[5.0], "bw_10dB_Hz": bw[10.0], "bw_20dB_Hz": bw[20.0],
                "resp_peak": resp_peak,
                "resp_unit": self._specs[resp_row]["unit"] if resp_row is not None else "",
                "second_peak_N": meta["second_peak_N"], "double_hit": meta["double_hit"],
                "overload": meta["overload"], "temp_C": temps, "quality": quality,
                "status": status, "notes": info["notes"],
            })
            if self._nas_dir is not None:
                nas_uploader.enqueue([self._log.path], self._nas_dir, delete_after=False)
        except Exception as exc:
            logger.error("impact_log 기록 실패: %s", exc)

        key = (session, loc, mic)
        self._last_saved = {"file": fname, "key": key, "added": False, "hit": hit_no}

        # ④ FRF·그래프 (실패해도 데이터는 이미 저장됨)
        if pulse is not None:
            try:
                self._update_analysis(data, fs, ti, pulse, ffreq, fmag, bw, status, fname, key)
            except Exception as exc:
                logger.exception("FRF/그래프 갱신 실패(데이터는 저장됨): %s", exc)

        # ⑤ 표시
        self._reject_btn.setEnabled(True)
        acc = self._log.count(session, loc, mic, "accepted") if self._log else 0
        tgt = self._target_spin.value()
        line = (f"{ts:%H:%M:%S}  S{session} {loc} {mic} #{hit_no:03d}  "
                f"{meta['peak_force_N']:6.1f} N  {meta['pulse_width_ms']:5.2f} ms  "
                f"BW10 {bw[10.0] / 1e3:5.1f} kHz  resp {resp_peak:.3g}  [{status}]")
        if quality != "ok":
            line += f" {quality}"
            if meta["double_hit"]:
                line += f" (2차 {meta['second_peak_N']:.1f} N @{meta['second_peak_ms']:.1f} ms)"
            if ovl:
                line += f" ({', '.join(meta['overload_channels'])})"
        self._hit_text.appendPlainText(line)
        if quality == "ok":
            msg = (f"저장 #{hit_no:03d} ({loc}, {mic}) — 피크 {meta['peak_force_N']:.1f} N, "
                   f"폭 {meta['pulse_width_ms']:.2f} ms, BW(−10 dB) {bw[10.0] / 1e3:.1f} kHz → 채택 {acc}/{tgt}")
            self._set_status(msg + ("   [목표 완료 — '다음 위치 ▶']" if acc >= tgt else ""), "ok")
        else:
            self._set_status(f"저장 #{hit_no:03d} — 자동 플래그: {quality} (평균 제외, 파일은 보존). "
                             f"다시 치세요.", "warn")
        self._update_location_status()

    def _update_analysis(self, data, fs, ti, pulse, ffreq, fmag, bw, status, fname, key) -> None:
        c = self._run_cfg
        sp = hit_spectra(data, fs, 0, self._resp_rows, pulse, ti, c["fmax"],
                         exp_end=0.01 if c["expwin"] else None)
        if (self._frf is None or self._frf_key != key
                or self._frf["Sff"].size != sp["F"].size):
            self._frf = frf_new(sp["F"].size, len(self._resp_rows))
            self._frf_key = key
        if status == "accepted" and self._resp_rows:
            frf_add(self._frf, sp["F"], sp["X"], hit_id=fname)
            self._last_saved["added"] = True
        tiny = np.finfo(float).tiny
        h1_hit = sp["X"] / np.where(np.abs(sp["F"]) > tiny, sp["F"], tiny)[None, :]
        self._last = {"data": data, "fs": fs, "ti": ti, "ffreq": ffreq, "fmag": fmag,
                      "bw": bw, "freqs": sp["freqs"], "h1_hit": h1_hit, "key": key}
        self._redraw()

    # ── 그래프 ──────────────────────────────────────────────────────────────

    def _safe(self, fn, name: str) -> None:
        try:
            fn()
        except Exception as exc:
            if name not in self._plot_warned:
                self._plot_warned.add(name)
                logger.warning("그래프 '%s' 갱신 실패(이후 경고 생략): %s", name, exc)

    def _redraw(self) -> None:
        L = self._last
        if not L or not self._resp_rows:
            return
        k = self._resp_combo.currentIndex()
        if k < 0 or k >= len(self._resp_rows):
            k = 0
        data, fs, ti = L["data"], L["fs"], L["ti"]
        n = data.shape[1]
        fmax = self._run_cfg.get("fmax", 20000.0)

        i0 = max(0, ti - int(0.001 * fs))
        i1 = min(n, ti + int(0.005 * fs))
        t_ms = (np.arange(i0, i1) - ti) / fs * 1e3
        self._safe(lambda: self._c_force.setData(t_ms, data[0, i0:i1]), "force")

        row = self._resp_rows[k]
        spec = self._specs[row]
        t = (np.arange(n) - ti) / fs
        self._safe(lambda: self._c_resp.setData(t, data[row]), "resp")
        self._safe(lambda: self._p_resp.setLabel("left", spec["name"], spec["unit"]), "resp_label")

        ffreq, fmag = L["ffreq"], L["fmag"]
        sel = (ffreq >= 50) & (ffreq <= 500)
        ref = float(np.median(fmag[sel])) if sel.any() else float(np.max(fmag))
        m = ffreq <= fmax
        fdb = 20 * np.log10(np.maximum(fmag[m], 1e-30) / max(ref, 1e-30))
        self._safe(lambda: self._c_fspec.setData(ffreq[m], fdb), "fspec")
        self._safe(lambda: self._l_bw.setValue(L["bw"][10.0]), "bw")

        freqs = L["freqs"]
        self._safe(lambda: self._c_frf_hit.setData(freqs, db20(L["h1_hit"][k])), "frf_hit")
        n_avg = self._frf["n"] if (self._frf is not None and self._frf_key == L["key"]) else 0
        if n_avg > 0:
            H = frf_h1(self._frf)[k]
            coh = frf_coherence(self._frf)[k]
            self._safe(lambda: self._c_frf_avg.setData(freqs, db20(H)), "frf_avg")
            self._safe(lambda: self._c_coh.setData(freqs, coh), "coh")
        else:
            H = L["h1_hit"][k]
            self._safe(lambda: self._c_frf_avg.setData([], []), "frf_avg")
            self._safe(lambda: self._c_coh.setData([], []), "coh")
        peaks = pick_peaks(freqs, H, fmin=200.0, fmax=fmax)
        self._safe(lambda: self._s_peaks.setData([p["f"] for p in peaks], [p["db"] for p in peaks]),
                   "peaks")
        unit = f"{spec['unit']}/N"
        self._safe(lambda: self._p_frf.setTitle(
            f"FRF |H1| — {spec['name']} / 해머 [{unit}]   평균 n={n_avg}"), "frf_title")

        sess, loc, mic = L["key"]
        lines = [f"[{spec['name']} / 해머]  평균 n={n_avg}  (S{sess} · {loc} · {mic})",
                 "   f [Hz]     |H| [dB]   위상 [°]"]
        for p in peaks:
            lines.append(f"  {p['f']:9.2f}   {p['db']:8.1f}   {p['phase_deg']:+7.0f}")
        if not peaks:
            lines.append("  (피크 없음)")
        lines.append("")
        lines.append("※ 위상은 같은 마이크·반경에서 베인별 부호 패턴(h 판별) 비교용")
        self._peaks_text.setPlainText("\n".join(lines))

    # ── 기타 동작 ────────────────────────────────────────────────────────────

    def _on_reject(self) -> None:
        last = self._last_saved
        if not last or self._log is None:
            return
        self._log.set_status(last["file"], "rejected")
        if self._nas_dir is not None:
            nas_uploader.enqueue([self._log.path], self._nas_dir, delete_after=False)
        removed = False
        if (last["added"] and self._frf is not None and self._frf_key == last["key"]
                and self._frf.get("last_id") == last["file"]):
            removed = frf_remove_last(self._frf)
        self._hit_text.appendPlainText(f"          ↳ #{last['hit']:03d} 제외됨 ({last['file']})")
        self._set_status(f"#{last['hit']:03d} 제외 — 로그 상태 rejected"
                         + (", FRF 평균에서 뺌" if removed else ""), "warn")
        self._last_saved = None
        self._reject_btn.setEnabled(False)
        self._redraw()
        self._update_location_status()
        logger.info("타격 제외: %s", last["file"])

    def _on_next_location(self) -> None:
        n = self._loc_combo.count()
        if n:
            self._loc_combo.setCurrentIndex((self._loc_combo.currentIndex() + 1) % n)

    def _update_location_status(self) -> None:
        sess = self._session_spin.value()
        loc = self._loc_combo.currentText().strip() or "unspecified"
        mic = self._mic_pos_combo.currentText().strip() or "-"
        tgt = self._target_spin.value()
        if self._log is None:
            self._loc_status.setText(f"S{sess} · {loc} · {mic} — 목표 {tgt}회")
            return
        acc = self._log.count(sess, loc, mic, "accepted")
        nxt = self._log.next_hit_number(sess, loc, mic)
        done = "  [완료]" if acc >= tgt else ""
        self._loc_status.setText(f"S{sess} · {loc} · {mic} — 채택 {acc}/{tgt}, 다음 #{nxt:03d}{done}")

    def _on_temperature(self, _ts, temps) -> None:
        self._last_temp = [float(x) for x in temps]
        self._last_temp_t = time.monotonic()
        parts = [f"ai{ch} {t:.1f}℃" for ch, t in zip(self._temp_channels, self._last_temp)]
        self._temp_label.setText("온도: " + " · ".join(parts))

    def _on_temp_error(self, msg: str) -> None:
        self._temp_label.setText(f"온도: 오류 ({msg[:40]})")

    def _set_status(self, text: str, level: str = "ok") -> None:
        color = {"ok": "#2e7d32", "busy": "#1565c0", "warn": "#e65100", "error": "#c62828"}.get(level, "")
        self._status.setStyleSheet(f"font-size: 13pt; font-weight: bold; color: {color};")
        self._status.setText(text)

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._stop_acquisition("종료")
        self._save_settings()
        super().closeEvent(event)
