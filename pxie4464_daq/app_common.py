"""두 실행 프로그램(상시 모니터링 main.py, 임팩트 해머 시험 impact_main.py)이
공유하는 초기화: 회전 파일 로그, 미처리 예외 기록, pyqtgraph Python 3.14 패치.
"""
import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def setup_logging(log_name: str = "daq.log") -> None:
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    # 10 MB × 5개 = 최대 50 MB 로그 보관
    fh = RotatingFileHandler(
        LOG_DIR / log_name,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(sh)
    root.addHandler(fh)


def install_exception_hook() -> None:
    """메인 스레드의 미처리 예외를 로그 파일에 기록한다."""
    _hook_log = logging.getLogger("uncaught")

    def _hook(exc_type, exc_value, exc_tb):
        # Python 3.14의 traceback.format_exception이 SIP 컨텍스트에서
        # f.line=None으로 깨지는 경우가 있어, 포맷 실패 시 최소 정보로 폴백한다.
        try:
            tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        except Exception:
            tb = f"{getattr(exc_type, '__name__', exc_type)}: {exc_value} " \
                 f"(트레이스백 포맷 실패 — Python 3.14 호환성)"
        try:
            _hook_log.critical("미처리 예외:\n%s", tb)
        except Exception:
            pass
        try:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        except Exception:
            pass

    sys.excepthook = _hook


def patch_pyqtgraph() -> None:
    """Python 3.14 + PyQt5 SIP 비호환 워크어라운드.

    PyQt5 SIP이 C++ 가상 메서드(mouseMoveEvent/leaveEvent)를 Python으로
    디스패치하는 컨텍스트에서 HoverEvent() 생성 시 TypeError 발생.
    원인: Python 3.14의 강화된 __init__ 반환값 검사 + SIP 디스패치 상호작용.

    sendHoverEvents는 마우스 커서 시각 효과 전용이므로
    TypeError를 무시해도 데이터 수집·이상 감지에 영향 없음.
    """
    _log = logging.getLogger(__name__)
    # ① HoverEvent (마우스 커서 시각 효과)
    try:
        from pyqtgraph.GraphicsScene.GraphicsScene import GraphicsScene
        _orig = GraphicsScene.sendHoverEvents

        def _safe(self, ev, exitOnly=False):
            try:
                _orig(self, ev, exitOnly=exitOnly)
            except TypeError:
                pass

        GraphicsScene.sendHoverEvents = _safe
        _log.debug("pyqtgraph HoverEvent 패치 적용 완료")
    except Exception as e:
        _log.warning("pyqtgraph HoverEvent 패치 실패 (무시): %s", e)

    # ② setData → PlotDataset.__init__ (그래프 데이터 갱신)
    #    Python 3.14+SIP에서 "__init__() should return None" TypeError 발생.
    #    실패 시 해당 프레임 그리기만 건너뛴다 (데이터 수집·저장에는 무관).
    try:
        from pyqtgraph.graphicsItems.PlotDataItem import PlotDataItem
        _orig_setdata = PlotDataItem.setData

        def _safe_setdata(self, *args, **kwargs):
            try:
                return _orig_setdata(self, *args, **kwargs)
            except TypeError:
                return None

        PlotDataItem.setData = _safe_setdata
        _log.debug("pyqtgraph setData 패치 적용 완료")
    except Exception as e:
        _log.warning("pyqtgraph setData 패치 실패 (무시): %s", e)
