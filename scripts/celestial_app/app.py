#!/usr/bin/env python3
"""
Celestial Sphere Visualization - Refactored Version
"""
import sys
from PyQt6 import QtWidgets
from ui.main_window import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1400, 800)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()