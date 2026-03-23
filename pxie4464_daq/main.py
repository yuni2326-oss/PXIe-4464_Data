import sys
import logging
from PyQt5.QtWidgets import QApplication
from pxie4464_daq.ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1400, 800)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
