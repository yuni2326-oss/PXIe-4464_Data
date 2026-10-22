import sys
import os
import logging
import traceback

# 직접 실행 시 (python main.py) 패키지 루트를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PyQt5.QtWidgets import QApplication
from pxie4464_daq.app_common import setup_logging, install_exception_hook, patch_pyqtgraph
from pxie4464_daq.ui.main_window import MainWindow


def main():
    setup_logging("daq.log")
    install_exception_hook()
    _log = logging.getLogger(__name__)
    autostart = "--autostart" in sys.argv
    _log.info("=" * 60)
    _log.info("DAQ 프로그램 시작 (autostart=%s)", autostart)
    _log.info("=" * 60)
    try:
        from PyQt5.QtCore import QTimer
        patch_pyqtgraph()
        app = QApplication(sys.argv)
        window = MainWindow()
        window.resize(1500, 950)      # 최대화 해제 시 복원 크기
        window.showMaximized()        # 시작 시 전체 화면 → 긴 메뉴가 잘리지 않음
        if autostart:
            # 이벤트 루프 시작 후 저장된 초기 실행조건으로 자동 재개
            QTimer.singleShot(1000, window.autostart)
        ret = app.exec_()
        _log.info("DAQ 프로그램 종료 (exit code=%d)", ret)
        sys.exit(ret)
    except Exception:
        _log.critical("DAQ 프로그램 비정상 종료:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
