"""임팩트 해머 시험 프로그램 진입점.

상시 모니터링(main.py)과 별개로 실행한다. 두 프로그램이 같은 DAQ 장치를 동시에
쓸 수 없으므로 상시 모니터링을 먼저 종료한 뒤 실행한다.

    python -m pxie4464_daq.impact_main      (또는 run_impact.bat)
"""
import logging
import os
import sys
import traceback

# 직접 실행 시 (python impact_main.py) 패키지 루트를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PyQt5.QtWidgets import QApplication
from pxie4464_daq.app_common import setup_logging, install_exception_hook, patch_pyqtgraph
from pxie4464_daq.ui.impact_window import ImpactWindow


def main():
    setup_logging("impact.log")
    install_exception_hook()
    _log = logging.getLogger(__name__)
    _log.info("=" * 60)
    _log.info("임팩트 해머 시험 프로그램 시작")
    _log.info("=" * 60)
    try:
        patch_pyqtgraph()
        app = QApplication(sys.argv)
        window = ImpactWindow()
        window.resize(1600, 980)
        window.showMaximized()
        ret = app.exec_()
        _log.info("임팩트 해머 시험 프로그램 종료 (exit code=%d)", ret)
        sys.exit(ret)
    except Exception:
        _log.critical("임팩트 프로그램 비정상 종료:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
