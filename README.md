# PXIe-4464 / 4492 진동 데이터 수집 · 이상감지 시스템

NI PXIe Sound & Vibration 모듈을 이용한 **다채널 실시간 진동 데이터 수집, FFT 분석, 이상 감지, 자동 저장** 애플리케이션입니다. 펌프·회전기계의 상시 무인 모니터링을 목표로, 장시간(수일) 연속 운전 안정성에 중점을 두고 있습니다.

## 주요 기능

- **다채널 동시 수집 (최대 12채널)**
  - PXIe-4464: 4채널 IEPE 가속도계 (`PXI1Slot3`)
  - PXIe-4492: 8채널 전압 입력 (`PXI1Slot5`, 선택 사용)
  - 채널별 개별 활성화/비활성화
- **실시간 시각화**: 시간 파형 + FFT 스펙트럼 (pyqtgraph, FFT 0–5 kHz 표시)
- **이상 감지**: Z-score + **정규화 마할라노비스 거리** 기반
  `LEARNING → NORMAL → WARNING → ALARM` (홀드오프 3회)
- **주기적 자동 저장**: 설정 주기(기본 30분)마다 raw 파형 + 채널별 FFT를 CSV로 저장
- **무인 운전 안정화**
  - DAQ 오류 시 **초기 실행조건 그대로 자동 재연결·재시작** (지수 백오프)
  - 멈춘 버퍼 반복 저장 방지(stale 가드), 비차단 오류 알림
  - 회전 파일 로그 + 1시간 heartbeat로 상태 추적
- **Mock 모드**: 하드웨어 없이 테스트 (채널별 사인파 생성)

## 대상 장비

| 항목 | PXIe-4464 | PXIe-4492 |
|------|-----------|-----------|
| 종류 | Sound & Vibration (IEPE) | 전압 입력 |
| 기본 슬롯 | `PXI1Slot3` | `PXI1Slot5` |
| 채널 | 4 (ai0~ai3) | 8 (ai0~ai7) |
| 센서 | IEPE 가속도계 (2/4 mA 공급) | 일반 전압 |
| 최대 샘플레이트 | 204.8 kS/s | 204.8 kS/s |
| ADC | 24-bit | 24-bit |

## 분석 파이프라인

```
DAQ(연속) → AcquisitionWorker(백그라운드 스레드)
   → FeatureCollector(rolling buffer, 주기적 추출)
        ├→ FFT 11개 특징 → AnomalyDetector(채널별 상태)
        └→ raw 윈도우 → DataSaver(주기적 CSV 저장)
```

**FFT 특징 11종** (`features.py`): 지배주파수, 지배진폭, 2차/3차 고조파(H2/H3), THD,
노이즈 플로어, 스펙트럼 중심, 고주파 에너지비(1–3 kHz), RMS, 첨도(kurtosis), 파고율(crest factor)

## 이상 감지 상태

| 상태 | 색상 | 조건 |
|------|------|------|
| LEARNING | 🔵 | 베이스라인 수집 중 (기본 20회 누적) |
| NORMAL | 🟢 | Z-score < 3.0 **AND** norm_dev > −2.0 |
| WARNING | 🟡 | Z-score 3.0~5.0 또는 norm_dev −2.0~−3.0 (3회 연속) |
| ALARM | 🔴 | Z-score ≥ 5.0 또는 norm_dev ≤ −3.0 (3회 연속) |

> **norm_dev**(정규화 마할라노비스 편차)는 특징 간 상관구조를 반영하므로, 단일 특징 Z-score가 놓치는 **다특징 확산 드리프트(펌프 점진적 열화)**도 포착합니다.

## 프로젝트 구조

```
pxie4464_daq/
├── device/daq.py              # PXIe4464 / PXIe4492 / MultiDAQ / MockDAQ
├── acquisition/worker.py      # 백그라운드 연속 수집 (threading.Thread)
├── analysis/
│   ├── fft.py                 # Hanning 윈도우 FFT (진폭 보정)
│   ├── features.py            # 11개 FFT 특징 추출
│   ├── feature_collector.py   # 주기적 특징 수집 + stale 가드
│   └── anomaly_detector.py    # Z-score + 마할라노비스 이상 감지
├── storage/
│   ├── csv_writer.py          # 수동 CSV 저장
│   └── data_saver.py          # 주기적 자동 저장 (+ 저장 로그)
├── ui/
│   ├── main_window.py         # 메인 GUI + 자동 재시작/heartbeat
│   ├── waveform_plot.py       # 실시간 파형
│   ├── fft_plot.py            # 실시간 FFT (0–5 kHz)
│   ├── anomaly_plot.py        # 이상 점수 히스토리
│   └── status_light.py        # 채널별 상태 표시등
└── main.py                    # 진입점 (로깅·예외훅·pyqtgraph 패치)
tests/                         # pytest 단위 테스트 (27 passed)
tools/disable_sleep.ps1        # 화면보호기·절전 비활성화 스크립트
logs/  results/                # 런타임 산출물 (gitignore)
```

## 설치 및 실행

```bash
pip install -r pxie4464_daq/requirements.txt
python -m pxie4464_daq.main      # 또는 python pxie4464_daq/main.py
```

> NI-DAQmx 드라이버는 별도 설치 필요: https://www.ni.com/ko-kr/support/downloads/drivers/download.ni-daq-mx.html

### 사용 순서

1. (실제 장비) **Mock 모드** 해제 → 장치명 확인(`PXI1Slot3` / `PXI1Slot5`)
2. 샘플레이트·수집 주기·수집 시간·학습 누적 횟수·저장 주기·채널 설정
3. **오류 시 자동 재시작** 체크(기본 ON) — 무인 운전 권장
4. **연결** → **▶ 시작**

### 무인 장시간 운전 팁

- `tools/disable_sleep.ps1`을 관리자 권한으로 실행해 화면보호기·절전·USB 선택적 절전·PCIe 링크 전원관리를 끈다. (절전 진입 시 PXI 버스가 끊기며 "argument 5" 오류 유발)
- 로그는 `logs/daq.log`(10 MB × 5 회전)에 기록되며, 저장 성공/실패·디스크 여유·heartbeat·자동 재시작 이력을 확인할 수 있다.

## 안정성 개선 이력 (장시간 운전)

수일 연속 운전 중 발견된 문제들과 해결 내역입니다.

| 문제 | 원인 | 해결 |
|------|------|------|
| 이상감지 100% 실패 (에러 폭주) | Python 3.14 + PyQt5 SIP에서 sklearn `IsolationForest` 내부 `__init__` 비호환 | sklearn 제거 → 순수 numpy **마할라노비스 거리**로 교체 |
| **동일 데이터 31시간 반복 저장** | DAQ 오류 시 **모달 오류창**이 `_stop_acquisition()`을 영구 차단 → 멈춘 버퍼를 계속 저장 | 정지 우선 실행 + **비모달** 알림, **stale 가드**(새 데이터 없으면 저장 건너뜀) |
| DAQ 끊김 후 수집 영구 중단 | 오류 시 멈추고 끝남 | **초기 실행조건 그대로 자동 재연결·재시작**(지수 백오프, 안정 시 카운터 리셋) |
| 플롯 오류로 앱 종료 위험 | pyqtgraph `setData → PlotDataset.__init__` SIP 비호환 + excepthook 2차 실패 | 데이터 경로 우선 공급 + 플롯 갱신 보호(`_safe_plot`), excepthook·pyqtgraph 패치 |
| 장시간 후 무흔적 다운 | C 레벨 크래시/외부 종료 추정 | 회전 파일 로그 + 미처리 예외 기록 + heartbeat (원인 추적) |

> 위 이상감지·플롯 오류는 모두 **Python 3.14 + PyQt5 SIP 디스패치 컨텍스트**에서
> `__init__() should return None` 형태로 나타나는 동일 계열 비호환이며, 각각 우회 적용됨.

## 테스트

```bash
pytest tests/ -v        # 27 passed, 1 skipped (하드웨어 테스트)
```

## 참고

- 참고 레포: [yuni2326-oss/Data_aquis](https://github.com/yuni2326-oss/Data_aquis)
- IEPE 전류: PXIe-4464는 **2mA / 4mA만** 지원
- FFT 정규화: `2.0 / sum(window)` (Hanning 진폭 보정), 1 Hz 분해능 = 1초 윈도우 @ 51.2 kS/s
- 마할라노비스 공분산은 ridge 정칙화 + 의사역행렬(pinv)로 소표본 안정성 확보
