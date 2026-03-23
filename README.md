# PXIe-4464 진동 데이터 수집 시스템

NI PXIe-4464 Sound & Vibration 모듈을 이용한 4채널 IEPE 가속도계 실시간 데이터 수집 및 이상 감지 애플리케이션입니다.

## 주요 기능

- **4채널 동시 수집**: IEPE 가속도계 신호 (PXIe-4464, nidaqmx)
- **실시간 시각화**: 시간 파형 + FFT 스펙트럼 (pyqtgraph)
- **이상 감지**: Z-score + IsolationForest 기반 LEARNING → NORMAL → WARNING → ALARM
- **CSV 저장**: 버튼 클릭 시 현재 버퍼 스냅샷 저장
- **Mock 모드**: 하드웨어 없이 테스트 가능 (채널별 사인파 생성)

## 참고 장비

| 항목 | 내용 |
|------|------|
| 장비 | NI PXIe-4464 (Sound & Vibration) |
| 드라이버 | nidaqmx |
| 채널 | 4채널 (ai0~ai3) |
| 센서 | IEPE 가속도계 (4mA 전류 공급) |
| 최대 샘플레이트 | 204.8 kS/s |
| ADC 분해능 | 24-bit |

## 프로젝트 구조

```
pxie4464_daq/
├── device/
│   └── daq.py              # PXIe4464 + MockDAQ (nidaqmx 기반)
├── acquisition/
│   └── worker.py           # QThread 연속 수집 스레드
├── analysis/
│   ├── fft.py              # Hanning 윈도우 FFT (진폭 보정)
│   ├── features.py         # 7개 FFT 특징 추출
│   ├── feature_collector.py # 주기적 특징 수집
│   └── anomaly_detector.py # Z-score + IsolationForest 이상 감지
├── storage/
│   └── csv_writer.py       # CSV 저장
├── ui/
│   ├── main_window.py      # 메인 GUI
│   ├── waveform_plot.py    # 4채널 실시간 파형
│   ├── fft_plot.py         # 4채널 실시간 FFT
│   ├── anomaly_plot.py     # 이상 점수 히스토리
│   └── status_light.py     # 상태 표시등
└── main.py                 # 진입점
tests/                      # pytest 단위 테스트 (22 passed)
```

## 설치

```bash
pip install -r pxie4464_daq/requirements.txt
```

**의존성:**
```
nidaqmx>=0.9.0
PyQt5>=5.15.0
pyqtgraph>=0.13.0
numpy>=1.24.0
scikit-learn>=1.0
pytest>=7.0.0
```

> NI-DAQmx 드라이버는 NI 공식 사이트에서 별도 설치 필요: https://www.ni.com/ko-kr/support/downloads/drivers/download.ni-daq-mx.html

## 실행

```bash
python -m pxie4464_daq.main
```

### Mock 모드 (하드웨어 없이 테스트)

1. **Mock 모드** 체크박스 선택 (기본값: ON)
2. **연결** 클릭
3. **▶ 시작** 클릭 → 실시간 파형/FFT 표시

### 실제 하드웨어 사용

1. **Mock 모드** 체크 해제
2. **장치명**에 PXI 시스템 장치명 입력 (기본: `Dev1`)
3. **샘플레이트** 설정 (기본: 51,200 S/s, 최대: 204,800 S/s)
4. **연결** → **▶ 시작**

## 이상 감지 상태

| 상태 | 색상 | 조건 |
|------|------|------|
| LEARNING | 🔵 파란색 | 베이스라인 수집 중 (첫 20 샘플) |
| NORMAL | 🟢 초록색 | Z-score < 3.0 AND IF 점수 > -0.05 |
| WARNING | 🟡 노란색 | Z-score 3.0~5.0 OR IF 점수 -0.05~-0.15 (3회 연속) |
| ALARM | 🔴 빨간색 | Z-score ≥ 5.0 OR IF 점수 ≤ -0.15 (3회 연속) |

## 테스트

```bash
pytest tests/ -v
```

## 설계 문서

- [설계 스펙](docs/superpowers/specs/2026-03-23-pxie4464-daq-design.md)
- [구현 계획](docs/superpowers/plans/2026-03-23-pxie4464-daq.md)

## 참고

- 참고 레포: [yuni2326-oss/Data_aquis](https://github.com/yuni2326-oss/Data_aquis) (NI USB-5133 기반)
- IEPE 전류: PXIe-4464는 **2mA / 4mA만** 지원 (10mA, 20mA 불가)
- FFT 정규화: `2.0 / sum(window)` 사용 (Hanning 윈도우 진폭 보정)
- IsolationForest 임계값(-0.05, -0.15)은 실제 센서 데이터 취득 후 경험적 조정 필요
