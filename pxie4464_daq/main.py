import sys
import os
import logging

# 직접 실행 시 (python main.py) 패키지 루트를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PyQt5.QtWidgets import QApplication
from pxie4464_daq.ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


def _patch_pyqtgraph_hover() -> None:
    """Python 3.14 + PyQt5 SIP 비호환 워크어라운드.

    PyQt5 SIP이 C++ 가상 메서드(mouseMoveEvent/leaveEvent)를 Python으로
    디스패치하는 컨텍스트에서 HoverEvent() 생성 시 TypeError 발생.
    원인: Python 3.14의 강화된 __init__ 반환값 검사 + SIP 디스패치 상호작용.

    sendHoverEvents는 마우스 커서 시각 효과 전용이므로
    TypeError를 무시해도 데이터 수집·이상 감지에 영향 없음.
    """
    try:
        from pyqtgraph.GraphicsScene.GraphicsScene import GraphicsScene
        _orig = GraphicsScene.sendHoverEvents

        def _safe(self, ev, exitOnly=False):
            try:
                _orig(self, ev, exitOnly=exitOnly)
            except TypeError:
                pass

        GraphicsScene.sendHoverEvents = _safe
        logging.getLogger(__name__).debug("pyqtgraph HoverEvent 패치 적용 완료")
    except Exception as e:
        logging.getLogger(__name__).warning("pyqtgraph 패치 실패 (무시): %s", e)


def main():
    _patch_pyqtgraph_hover()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1400, 800)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
