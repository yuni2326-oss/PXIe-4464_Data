from __future__ import annotations
import logging
import queue as _queue
import shutil
import threading
import time
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# 네트워크(NAS) 복사는 지연·끊김이 있을 수 있어 절대 메인 스레드에서 하지 않는다.
# Python 3.14 + PyQt5 SIP 컨텍스트에서 threading 객체 생성이 실패하므로,
# 모듈 임포트 시점(SIP 밖)에 전용 워커 스레드와 큐를 미리 생성한다.
# (worker.py / 기존 패턴과 동일)
_UPLOAD_Q: "_queue.Queue" = _queue.Queue()
_MAX_ATTEMPTS = 3        # 전송 재시도 횟수 (일시적 네트워크 블립 대비)
_RETRY_DELAY = 3.0       # 재시도 간 대기(초)


def _upload_worker() -> None:
    while True:
        job = _UPLOAD_Q.get()
        if job is None:
            return
        src, dest_dir, delete_after = job
        try:
            _do_upload(Path(src), Path(dest_dir), delete_after)
        except Exception as exc:  # 워커는 어떤 경우에도 죽지 않는다
            try:
                logger.error("[NAS전송] 처리 중 예외 (무시): %s", exc)
            except Exception:
                pass


def _do_upload(src: Path, dest_dir: Path, delete_after: bool) -> None:
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            shutil.copy2(src, dest)  # 메타데이터 보존
            if delete_after:
                try:
                    src.unlink()
                except Exception as exc:
                    logger.warning("[NAS전송] 전송 후 로컬 삭제 실패: %s (%s)", src.name, exc)
            logger.info("[NAS전송] 완료: %s → %s", src.name, dest_dir)
            return
        except Exception as exc:
            logger.warning("[NAS전송] 실패(%d/%d) %s → %s: %s",
                           attempt, _MAX_ATTEMPTS, src.name, dest_dir, exc)
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_RETRY_DELAY)
    # 최종 실패 — 로컬 파일은 보존(삭제하지 않음)하여 데이터 유실 방지
    logger.error("[NAS전송] 최종 실패, 로컬 보존: %s", src.name)


# 모듈 임포트(앱 시작, SIP 컨텍스트 밖) 시점에 워커 스레드 기동
threading.Thread(target=_upload_worker, daemon=True, name="NASUploader").start()


def enqueue(paths: Iterable[Path], dest_dir: str | Path, delete_after: bool = False) -> None:
    """로컬 파일들을 NAS(dest_dir)로 백그라운드 복사하도록 큐에 넣는다.

    호출은 즉시 반환한다(네트워크 대기 없음). 실제 복사는 워커 스레드에서 수행.
    """
    dest = str(dest_dir)
    for p in paths:
        _UPLOAD_Q.put((str(p), dest, delete_after))


def pending_count() -> int:
    """전송 대기 중인 파일 수 (heartbeat/상태 로깅용)."""
    return _UPLOAD_Q.qsize()
