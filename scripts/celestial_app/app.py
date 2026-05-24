
#!/usr/bin/env python3
"""
Celestial Sphere Visualization - Refactored Version
"""
import logging
import sys
from PyQt6 import QtWidgets
from ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1400, 800)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

