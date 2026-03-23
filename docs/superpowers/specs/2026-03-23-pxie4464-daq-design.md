# PXIe-4464 진동 데이터 수집 시스템 설계 문서

**날짜:** 2026-03-23
**참고 레포:** github.com/yuni2326-oss/Data_aquis
**장비:** NI PXIe-4464 (Sound & Vibration Module)

---

## 1. 개요

NI PXIe-4464를 이용해 4채널 IEPE 가속도계 신호를 동시에 수집하고, 실시간 FFT 분석 및 이상 감지를 수행하는 PyQt5 기반 GUI 애플리케이션.

### 주요 요구사항
- PXIe-4464 4채널 동시 수집 (IEPE 전류 공급 4mA, AC coupling 내장)
- 실시간 시간 파형 + FFT 스펙트럼 시각화
- 이상 감지: LEARNING → NORMAL → WARNING → ALARM 상태 전환 (채널별)
- 샘플레이트 UI 설정 가능 (최대 204.8 kS/s)
- CSV 저장 (버튼 클릭 시 현재 버퍼 스냅샷 저장)
- Mock 모드 (하드웨어 없이 테스트 가능)

---

## 2. 프로젝트 구조

```
pxie4464_daq/
├── device/
│   └── daq.py              # PXIe4464 + MockDAQ 구현 (nidaqmx 기반)
├── acquisition/
│   └── worker.py           # QThread 기반 연속 수집 백그라운드 스레드
├── analysis/
│   ├── fft.py              # Hanning 윈도우 FFT 계산
│   ├── features.py         # 7개 FFT 특징 추출
│   ├── feature_collector.py # 주기적 특징 수집 (4채널 각각)
│   └── anomaly_detector.py # Z-score + IsolationForest (채널별 독립)
├── storage/
│   └── csv_writer.py       # 원시 데이터 + FFT 데이터 CSV 저장
├── ui/
│   ├── main_window.py      # 메인 GUI (제어 패널 + 플롯 배치)
│   ├── waveform_plot.py    # 4채널 실시간 시간 파형
│   ├── fft_plot.py         # 4채널 실시간 FFT 스펙트럼
│   ├── anomaly_plot.py     # 채널별 이상 점수 히스토리
│   └── status_light.py     # 상태 표시등
├── tests/
│   ├── test_daq.py
│   ├── test_worker.py
│   ├── test_fft.py
│   ├── test_features.py
│   ├── test_anomaly_detector.py
│   └── test_csv_writer.py
├── main.py                 # 진입점
└── requirements.txt
```

---

## 3. 아키텍처 및 데이터 흐름

```
PXIe-4464 하드웨어
    │ nidaqmx.Task (4채널 동시, add_ai_accel_chan)
    ▼
device/daq.py (PXIe4464)
    │ read() → numpy array (4, N) shape
    │   (내부적으로 pre-allocated buffer + AnalogMultiChannelReader 사용)
    ▼
acquisition/worker.py (QThread)
    │ data_ready 시그널(object) → numpy array (4, N)
    ▼
    ├──→ ui/waveform_plot.py        (실시간 시간 파형, 4채널)
    │
    ├──→ analysis/fft.py → ui/fft_plot.py  (실시간 FFT 시각화, 4채널)
    │
    └──→ analysis/feature_collector.py
              │ 수집 주기마다 (기본 30초)
              │ features_ready 시그널(object) → shape (4, 7)
              ▼
         analysis/anomaly_detector.py (채널별 4개 독립 인스턴스)
              │ state_changed 시그널(object) → list[State] (4채널 상태)
              ├──→ ui/anomaly_plot.py    (이상 점수 히스토리)
              └──→ ui/status_light.py   (최악 채널 기준 상태)
```

**실시간 FFT 경로:** `AcquisitionWorker.data_ready` → `MainWindow`에서 각 채널 `compute_fft()` 호출 → `FFTPlot.update()` (매 청크마다 갱신)

**CSV 저장 경로:** "CSV 저장" 버튼 클릭 → `MainWindow`가 현재 파형 버퍼 + FFT 결과를 `csv_writer`로 전달하여 스냅샷 저장

---

## 4. 컴포넌트 상세

### 4.1 device/daq.py

**`_DAQBase`** — 추상 베이스 클래스 (context manager 지원)
- `configure(sample_rate, record_length, voltage_range)`
- `read()` → `np.ndarray (4, N)`
- `start()`, `stop()`

**`PXIe4464`** — 실제 하드웨어 드라이버
- `nidaqmx.Task` 생성, `AnalogMultiChannelReader` 사용
- 4채널 (ai0~ai3) IEPE 설정 (`add_ai_accel_chan`):
  ```python
  task.ai_channels.add_ai_accel_chan(
      physical_channel="Dev1/ai0:3",
      sensitivity=100.0,  # mV/g (센서에 따라 조정)
      sensitivity_units=AccelSensitivityUnits.MILLIVOLTS_PER_G,
      current_excit_source=ExcitationSource.INTERNAL,
      current_excit_val=0.004,  # 4 mA (PXIe-4464 지원값: 2/4 mA)
  )
  ```
  - AC coupling은 IEPE 회로 내부에서 자동 처리 (별도 설정 불필요)
- `read()` 구현: pre-allocated buffer `(4, N)` → `samps_read = reader.read_many_sample(buffer, N)` → buffer 반환
  - `read_many_sample()`은 buffer를 in-place 채우고 실제 읽은 샘플 수(int)를 반환함
  - `samps_read < N` 인 경우 (부분 읽기) 해당 청크는 폐기하고 재시도 (데이터 오염 방지)
- 오류 처리: `DaqError` 발생 시 `stop()` 후 re-raise; cleanup 실패 시 로그만 남기고 swallow
- `configure()`에서 샘플 클럭 타이밍: `OnboardClock` (단일 모듈 기준)

**`MockDAQ`** — 테스트용 가상 장치
- 채널별 다른 주파수 사인파 생성:
  - CH0: 100 Hz, CH1: 200 Hz, CH2: 300 Hz, CH3: 400 Hz
- 가우시안 노이즈 추가 (SNR ~40 dB)
- `configure()`, `read()`, `start()`, `stop()` 동일 인터페이스 구현

### 4.2 acquisition/worker.py

**`AcquisitionWorker(QThread)`**
- 시그널: `data_ready = pyqtSignal(object)` (numpy array 전달, object 타입 사용)
- 시그널: `error_occurred = pyqtSignal(str)`
- `run()`: 루프에서 `daq.read()` 호출 → `data_ready` emit
- `stop()`: 루프 종료 플래그 설정 → `daq.stop()` 호출 (실패 시 로그만 남김)

### 4.3 analysis/fft.py

**`compute_fft(data, sample_rate)`**
- 입력: `(N,)` 1D 배열
- Hanning 윈도우 적용
- 단측 스펙트럼 반환 (N/2 bins)
- 정규화: `2.0 / sum(window)` (Hanning 윈도우 적용 시 진폭 보정; `2.0/N` 대신 사용하여 실제 진폭 복원)
- 반환: `(frequencies, magnitudes)` 튜플 (각각 `np.ndarray`)

### 4.4 analysis/features.py

**`extract_features(frequencies, magnitudes)`** → `np.ndarray` (7개 요소):
1. Dominant Frequency (Hz)
2. Dominant Magnitude
3. 2nd Harmonic Magnitude
4. 3rd Harmonic Magnitude
5. THD (Total Harmonic Distortion) = (2nd + 3rd) / dominant_mag
6. Noise Floor RMS (dominant peak ±5 bin 제외)
7. Spectral Centroid (Hz)

### 4.5 analysis/feature_collector.py

**`FeatureCollector(QObject)`**
- 수집 주기: 기본 30초
- 수집 윈도우: 기본 5초 분량의 샘플을 rolling buffer에 누적
- `data_ready` 시그널을 받아 4채널 파형 누적
- 주기마다 각 채널에 `compute_fft()` + `extract_features()` 적용
- 시그널: `features_ready = pyqtSignal(object)` — shape `(4, 7)` numpy 배열
  (`raw_ready` 시그널은 사용하지 않음; CSV 저장은 MainWindow에서 직접 처리)

### 4.6 analysis/anomaly_detector.py

**`ChannelAnomalyDetector`** — 단일 채널용 이상 감지기
- 베이스라인 수집: 20 샘플 → IsolationForest 학습
- 상태 전환 로직:
  - **LEARNING**: 수집 샘플 수 < 20
  - LEARNING → NORMAL: 20번째 샘플 수집 후 IsolationForest fit 완료 시
  - **NORMAL / WARNING / ALARM**: 매 샘플에서 두 지표를 OR 조합으로 판단:

  - IsolationForest: `decision_function()` 사용 (범위: 약 -0.25 ~ +0.15)

| 상태 | Z-score 조건 | IsolationForest decision_function 조건 |
|------|-------------|--------------------------------------|
| NORMAL | < 3.0 | > -0.05 |
| WARNING | 3.0 이상 5.0 미만 | -0.05 이하 -0.15 초과 |
| ALARM | ≥ 5.0 | ≤ -0.15 |

  - 두 지표 중 **더 심각한 쪽** 기준으로 상태 결정 (OR 로직)
  - 임계값(-0.05, -0.15)은 실제 센서 데이터 취득 후 경험적 조정 필요
  - IsolationForest 생성 시 `contamination=0.05` 명시적 지정 (sklearn 기본값 'auto' 대신) → LEARNING 직후 오탐 방지
  - WARNING 상태 진입은 **3회 연속** WARNING 판정 시에만 발동 (단발 스파이크 무시, 홀드오프 정책)

**`AnomalyDetector(QObject)`** — 4채널 통합 관리자
- 내부에 `ChannelAnomalyDetector` 4개 인스턴스 보유
- `features_ready` 시그널을 받아 각 채널 독립 처리
- 시그널: `state_changed = pyqtSignal(object)` — `list[State]` (4채널 상태 리스트)

### 4.7 storage/csv_writer.py

**`save_raw(data, sample_rate, timestamp)`**
- 파일명: `raw_YYYYMMDD_HHMMSS_ch{n}.csv` (채널별)
- 컬럼: `time_s, acceleration_g`

**`save_fft(frequencies, magnitudes, timestamp)`**
- 파일명: `fft_YYYYMMDD_HHMMSS_ch{n}.csv` (채널별)
- 컬럼: `frequency_hz, magnitude`

**저장 트리거:** "CSV 저장" 버튼 클릭 시 `MainWindow`가 현재 버퍼 스냅샷을 전달하여 즉시 저장

### 4.8 UI 컴포넌트

**`MainWindow`** — 제어 패널 + 플롯 배치
- 설정 항목:
  - 장치명 (기본: `Dev1`)
  - 샘플레이트 (기본: 51,200 S/s; 범위: 1,000~204,800)
  - 읽기 청크 크기 (기본: 1,024 샘플; 매 read() 호출당 샘플 수, UI 갱신 주기 결정)
  - 전압 범위 드롭다운: ±1V / ±3.16V / ±10V / ±31.6V (PXIe-4464 실제 지원 범위)
  - Mock 모드 체크박스
- 버튼: 연결, 시작/정지, CSV 저장

**`WaveformPlot`** — 4채널 실시간 시간 파형 (pyqtgraph)

**`FFTPlot`** — 4채널 실시간 FFT 스펙트럼 (pyqtgraph, 매 청크마다 갱신)

**`AnomalyPlot`** — 채널별 이상 점수 히스토리 (50샘플 rolling, 임계선 표시)

**`StatusLight`** — 4채널 중 최악 상태 기준 신호등 (LEARNING/NORMAL/WARNING/ALARM)
- 각 채널 개별 상태도 텍스트로 함께 표시

---

## 5. 설정 파라미터 요약

| 파라미터 | 기본값 | 범위/옵션 |
|---------|--------|----------|
| 샘플레이트 | 51,200 S/s | 1,000 ~ 204,800 |
| 읽기 청크 크기 | 1,024 샘플 | 256 ~ 65,536 |
| IEPE 전류 | 4 mA | 2 / 4 mA (PXIe-4464 하드웨어 지원값) |
| 전압 범위 | ±10 V | ±1V / ±3.16V / ±10V / ±31.6V |
| 베이스라인 샘플 수 | 20 | - |
| 수집 주기 | 30 초 | - |
| 수집 윈도우 | 5 초 | - |
| 샘플 클럭 소스 | OnboardClock | - |

---

## 6. 의존성 (requirements.txt)

```
nidaqmx>=0.9.0
PyQt5>=5.15.0
pyqtgraph>=0.13.0
numpy>=1.24.0
scikit-learn>=1.0
pytest>=7.0.0
```

---

## 7. 참고 레포 대비 변경사항

| 항목 | 참고 레포 (Data_aquis) | 이 프로젝트 |
|------|----------------------|------------|
| 장비 | NI USB-5133 | NI PXIe-4464 |
| 드라이버 | `niscope` | `nidaqmx` |
| 채널 API | `add_ai_voltage_chan` | `add_ai_accel_chan` (IEPE) |
| 채널 수 | 2채널 | 4채널 |
| 센서 | 전압 직접 입력 | IEPE 가속도계 |
| 최대 샘플레이트 | 100 MHz | 204.8 kS/s |
| ADC 분해능 | - | 24-bit |
| PyQt5 시그널 | - | `pyqtSignal(object)` (numpy 배열) |
| 이상 감지 구조 | 단일 감지기 | 채널별 독립 (`ChannelAnomalyDetector` × 4) |
