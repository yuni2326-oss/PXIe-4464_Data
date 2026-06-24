import sys
import os
import logging
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 직접 실행 시 (python main.py) 패키지 루트를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PyQt5.QtWidgets import QApplication
from pxie4464_daq.ui.main_window import MainWindow

_LOG_DIR = Path(__file__).parent.parent / "logs"


def _setup_logging() -> None:
    _LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    # 10 MB × 5개 = 최대 50 MB 로그 보관
    fh = RotatingFileHandler(
        _LOG_DIR / "daq.log",
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


def _install_exception_hook() -> None:
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


def _patch_pyqtgraph_hover() -> None:
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


def main():
    _setup_logging()
    _install_exception_hook()
    _log = logging.getLogger(__name__)
    _log.info("=" * 60)
    _log.info("DAQ 프로그램 시작")
    _log.info("=" * 60)
    try:
        _patch_pyqtgraph_hover()
        app = QApplication(sys.argv)
        window = MainWindow()
        window.resize(1400, 800)
        window.show()
        ret = app.exec_()
        _log.info("DAQ 프로그램 정상 종료 (exit code=%d)", ret)
        sys.exit(ret)
    except Exception:
        _log.critical("DAQ 프로그램 비정상 종료:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
