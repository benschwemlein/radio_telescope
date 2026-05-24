
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

_DARK_STYLESHEET = """
QWidget {
    background-color: #12121a;
    color: #e8e8f0;
    font-size: 12px;
}
QMainWindow, QDialog {
    background-color: #0e0e16;
}
QLabel {
    color: #c8c8d8;
    background-color: transparent;
}
QLineEdit {
    background-color: #1e1e2e;
    color: #e8e8f0;
    border: 1px solid #3a3a52;
    border-radius: 3px;
    padding: 3px 5px;
    selection-background-color: #4a4a70;
}
QLineEdit:focus {
    border: 1px solid #6a6a9a;
}
QPushButton {
    background-color: #2a2a3e;
    color: #d0d0e8;
    border: 1px solid #4a4a66;
    border-radius: 4px;
    padding: 4px 12px;
}
QPushButton:hover {
    background-color: #3a3a52;
    border-color: #7a7aaa;
}
QPushButton:pressed {
    background-color: #1e1e30;
}
QPushButton:default {
    border-color: #7070b0;
}
QTabWidget::pane {
    border: 1px solid #3a3a52;
    background-color: #0e0e16;
}
QTabBar::tab {
    background-color: #1e1e2e;
    color: #9090b0;
    border: 1px solid #3a3a52;
    padding: 5px 14px;
    border-bottom: none;
}
QTabBar::tab:selected {
    background-color: #2a2a40;
    color: #e0e0f8;
    border-bottom: 1px solid #2a2a40;
}
QTabBar::tab:hover {
    background-color: #252538;
    color: #c0c0e0;
}
QMenuBar {
    background-color: #0e0e16;
    color: #c8c8d8;
}
QMenuBar::item:selected {
    background-color: #2a2a40;
}
QMenu {
    background-color: #1a1a28;
    color: #c8c8d8;
    border: 1px solid #3a3a52;
}
QMenu::item:selected {
    background-color: #3a3a58;
}
QMenu::separator {
    height: 1px;
    background: #3a3a52;
    margin: 3px 8px;
}
QGroupBox {
    color: #9090b0;
    border: 1px solid #3a3a52;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 6px;
    font-size: 11px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QTableWidget {
    background-color: #14141e;
    color: #d0d0e0;
    gridline-color: #2e2e42;
    border: 1px solid #3a3a52;
    alternate-background-color: #1a1a28;
}
QTableWidget::item:selected {
    background-color: #2e2e50;
    color: #f0f0ff;
}
QHeaderView::section {
    background-color: #1e1e30;
    color: #9090b8;
    border: 1px solid #2e2e44;
    padding: 4px 6px;
}
QScrollBar:vertical {
    background: #14141e;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #3a3a52;
    border-radius: 4px;
    min-height: 20px;
}
QDoubleSpinBox, QSpinBox {
    background-color: #1e1e2e;
    color: #e8e8f0;
    border: 1px solid #3a3a52;
    border-radius: 3px;
    padding: 3px 5px;
}
QDateTimeEdit {
    background-color: #1e1e2e;
    color: #e8e8f0;
    border: 1px solid #3a3a52;
    border-radius: 3px;
    padding: 3px 5px;
}
QDateTimeEdit::drop-down {
    background-color: #2a2a3e;
    border-left: 1px solid #3a3a52;
}
QTextEdit {
    background-color: #1e1e2e;
    color: #e8e8f0;
    border: 1px solid #3a3a52;
    border-radius: 3px;
}
QComboBox {
    background-color: #1e1e2e;
    color: #e8e8f0;
    border: 1px solid #3a3a52;
    border-radius: 3px;
    padding: 3px 5px;
}
QMessageBox {
    background-color: #1a1a28;
}
"""

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(_DARK_STYLESHEET)
    w = MainWindow()
    w.resize(1400, 800)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

