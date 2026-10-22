"""임팩트 해머 시험 저장.

- 타격 1회 = .npz 1개: data(float32, 채널×샘플, 물리단위) + 실험 메타데이터 전부.
  임시 파일에 쓴 뒤 교체(os.replace)해 중간에 끊겨도 깨진 파일이 남지 않는다.
- impact_log.csv: 타격 목록(품질·채택/제외 상태). 작은 파일이라 바뀔 때마다 통째로
  다시 쓴다. 시작 시 기존 로그를 읽어 이어 쓰므로 프로그램을 다시 켜도 번호가 이어진다.

상태(status):
  accepted  품질 ok — 위치별 목표 횟수·FRF 평균에 포함
  flagged   자동 플래그(이중 타격/과부하/프리트리거 부족) — 평균 제외, 파일은 보존
  rejected  사용자가 '직전 타격 제외'로 뺀 것 — 파일은 보존
"""
from __future__ import annotations

import csv
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

LOG_NAME = "impact_log.csv"
LOG_COLUMNS = [
    "file", "timestamp", "session", "location", "mic_position", "hit",
    "peak_force_N", "pulse_width_ms", "bw_5dB_Hz", "bw_10dB_Hz", "bw_20dB_Hz",
    "resp_peak", "resp_unit", "second_peak_N", "double_hit", "overload",
    "temp_C", "quality", "status", "notes",
]


def safe_token(text: str) -> str:
    """파일명에 쓸 수 있게 영숫자·._- 이외 문자를 _로."""
    t = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return t or "x"


def hit_filename(ts: datetime, session: int, location: str, mic: str, hit: int) -> str:
    return (f"{ts:%Y%m%d_%H%M%S}_{ts.microsecond // 1000:03d}"
            f"_S{int(session)}_{safe_token(location)}_{safe_token(mic)}_H{int(hit):03d}.npz")


def _to_array(v):
    """메타데이터 값 → np.load(allow_pickle=False)로 읽히는 배열 (None 금지)."""
    if v is None:
        return np.asarray("")
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return np.asarray([], dtype=np.float64)
        if all(isinstance(x, str) for x in v):
            return np.asarray(list(v), dtype=str)
        return np.asarray([np.nan if x is None else x for x in v], dtype=np.float64)
    if isinstance(v, datetime):
        return np.asarray(v.isoformat(timespec="milliseconds"))
    return np.asarray(v)


def save_hit(folder: Path, filename: str, data: np.ndarray, meta: Dict) -> Path:
    """타격 레코드 저장 → 최종 경로. data는 float32로 저장한다."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    tmp = folder / (filename + ".part")
    payload = {"data": np.asarray(data, dtype=np.float32)}
    for k, v in meta.items():
        payload[k] = _to_array(v)
    with open(tmp, "wb") as fh:
        np.savez(fh, **payload)
    os.replace(tmp, path)
    return path


class ImpactLog:
    """impact_log.csv 관리 (메모리 목록 ↔ 파일)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: List[Dict[str, str]] = []
        self.load()

    # GUI 슬롯에서 불리므로 C 구현인 csv.reader/writer만 쓴다
    # (이 환경에서 슬롯 안 순수 파이썬 클래스 생성이 드물게 실패한 이력 — DictReader/Writer 회피).
    def load(self) -> None:
        self.rows = []
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8", newline="") as fh:
                it = csv.reader(fh)
                header = next(it, None)
                if not header:
                    return
                idx = {name: i for i, name in enumerate(header)}
                for rec in it:
                    if not rec:
                        continue
                    self.rows.append({c: (rec[idx[c]] if c in idx and idx[c] < len(rec) else "")
                                      for c in LOG_COLUMNS})
        except Exception as exc:
            logger.warning("impact_log 읽기 실패(새로 시작): %s", exc)
            self.rows = []

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(LOG_COLUMNS)
            for r in self.rows:
                w.writerow([r.get(c, "") for c in LOG_COLUMNS])
        # NAS 업로더가 원본을 복사 중이면 교체가 막힐 수 있다 → 잠깐 재시도
        for _ in range(10):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.05)
        os.replace(tmp, self.path)

    def append(self, row: Dict) -> None:
        self.rows.append({c: _fmt(row.get(c, "")) for c in LOG_COLUMNS})
        self.write()

    def set_status(self, file: str, status: str) -> bool:
        for r in reversed(self.rows):
            if r["file"] == file:
                r["status"] = status
                self.write()
                return True
        return False

    def _match(self, session: int, location: str, mic: str):
        s = str(int(session))
        return [r for r in self.rows
                if r["session"] == s and r["location"] == location and r["mic_position"] == mic]

    def next_hit_number(self, session: int, location: str, mic: str) -> int:
        nums = []
        for r in self._match(session, location, mic):
            try:
                nums.append(int(r["hit"]))
            except ValueError:
                pass
        return (max(nums) + 1) if nums else 1

    def count(self, session: int, location: str, mic: str, status: Optional[str] = None) -> int:
        rows = self._match(session, location, mic)
        if status is not None:
            rows = [r for r in rows if r["status"] == status]
        return len(rows)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        if v != v:            # NaN
            return ""
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return " ".join(_fmt(x) for x in v)
    return str(v)
