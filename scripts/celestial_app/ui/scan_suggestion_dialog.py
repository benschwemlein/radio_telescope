"""
Scan Suggestion Dialog — shows optimal drift-scan opportunities and lets the
user send one directly into the New Scan dialog.
"""
from __future__ import annotations

from PyQt6 import QtWidgets, QtCore, QtGui
from radio_telescope.scan_planner import ScanSuggestion


class ScanSuggestionDialog(QtWidgets.QDialog):
    """
    Displays ranked drift-scan opportunities computed by ScanPlanner.
    Selecting a suggestion and clicking "Use This Scan" populates the
    standard ScanEntryDialog with the recommended parameters.
    """

    # Emitted when the user picks a suggestion to act on
    scan_accepted = QtCore.pyqtSignal(dict)

    def __init__(self, suggestions: list[ScanSuggestion], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Suggested Milky Way Drift Scans")
        self.setMinimumWidth(720)
        self.setMinimumHeight(480)
        self.suggestions = suggestions
        self._build_ui()
        if suggestions:
            self.table.selectRow(0)
            self._on_selection_changed()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        # --- Explanation banner ---
        info = QtWidgets.QLabel(
            "The table below lists the best times to perform a drift scan across "
            "the Milky Way in the next 24 hours.\n"
            "Point your dish at the suggested Alt/Az and record from Start to End — "
            "Earth's rotation will sweep the beam from empty sky, through the galactic "
            "plane (hydrogen peak), and back to empty sky."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 6px; background: #1a2030; border-radius: 4px;")
        root.addWidget(info)
        root.addSpacing(8)

        # --- Results table ---
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Region", "Alt", "Az", "Start", "MW Peak", "End",
            "MW crossing", "Brightness",
        ])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)

        self._populate_table()
        self.table.resizeColumnsToContents()
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        root.addWidget(self.table)

        # --- Detail panel ---
        self.detail_box = QtWidgets.QGroupBox("Scan detail")
        detail_layout = QtWidgets.QVBoxLayout(self.detail_box)

        self.detail_label = QtWidgets.QLabel("Select a row above.")
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        detail_layout.addWidget(self.detail_label)
        root.addWidget(self.detail_box)
        root.addSpacing(4)

        # --- Buttons ---
        btn_row = QtWidgets.QHBoxLayout()

        self.use_btn = QtWidgets.QPushButton("Use This Scan →")
        self.use_btn.setDefault(True)
        self.use_btn.setEnabled(bool(self.suggestions))
        self.use_btn.clicked.connect(self._on_use)
        btn_row.addWidget(self.use_btn)

        btn_row.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)

        root.addLayout(btn_row)

    def _populate_table(self):
        self.table.setRowCount(len(self.suggestions))
        for row, s in enumerate(self.suggestions):
            def cell(text: str, align=QtCore.Qt.AlignmentFlag.AlignCenter) -> QtWidgets.QTableWidgetItem:
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(align)
                return item

            time_fmt = "%H:%M"
            self.table.setItem(row, 0, cell(s.galactic_region,
                                            QtCore.Qt.AlignmentFlag.AlignLeft))
            self.table.setItem(row, 1, cell(f"{s.altitude_deg:.0f}°"))
            self.table.setItem(row, 2, cell(f"{s.azimuth_deg:.0f}°  "
                                             f"({'S' if s.azimuth_deg == 180 else 'N'})"))
            self.table.setItem(row, 3, cell(s.start_time.strftime(time_fmt)))
            self.table.setItem(row, 4, cell(s.peak_time.strftime(time_fmt)))
            self.table.setItem(row, 5, cell(s.end_time.strftime(time_fmt)))
            self.table.setItem(row, 6, cell(f"{s.crossing_duration_min:.0f} min"))

            # Colour-coded brightness bar
            pct = int(s.brightness * 100)
            bright_item = cell(f"{pct}%")
            # Green → orange → red depending on brightness (higher = better)
            r = max(0, int(255 * (1 - s.brightness)))
            g = max(0, int(255 * s.brightness))
            bright_item.setBackground(QtGui.QColor(r // 2, g // 2, 0))
            self.table.setItem(row, 7, bright_item)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_selection_changed(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.detail_label.setText("Select a row above.")
            self.use_btn.setEnabled(False)
            return

        self.use_btn.setEnabled(True)
        s = self.suggestions[rows[0].row()]
        date_fmt = "%Y-%m-%d %H:%M"

        tz_label = ""
        try:
            tz_label = f"  ({s.peak_time.tzname()})"
        except Exception:
            pass

        self.detail_label.setText(
            f"<b>Region:</b> {s.galactic_region}  "
            f"(galactic longitude l = {s.galactic_longitude_deg:.1f}°, "
            f"brightness {s.brightness:.0%})<br>"
            f"<b>Pointing:</b> Altitude {s.altitude_deg:.1f}°  "
            f"Azimuth {s.azimuth_deg:.0f}° "
            f"({'due South' if s.azimuth_deg == 180 else 'due North'})<br>"
            f"<b>Declination swept:</b> {s.dec_deg:+.1f}°<br><br>"
            f"<b>Start recording:</b> {s.start_time.strftime(date_fmt)}{tz_label}<br>"
            f"<b>Galactic plane in beam:</b> {s.peak_time.strftime(date_fmt)}<br>"
            f"<b>Stop recording:</b> {s.end_time.strftime(date_fmt)}<br><br>"
            f"<b>MW crossing time:</b> {s.crossing_duration_min:.0f} min  |  "
            f"<b>Baseline each side:</b> {s.baseline_min:.0f} min  |  "
            f"<b>Total recording:</b> {s.total_duration_min:.0f} min<br><br>"
            f"<i>The hydrogen line signal should peak near the 'MW Peak' time.  "
            f"Compare the baseline regions (before start / after end) to the "
            f"peak region to see the 1420.4 MHz emission.</i>"
        )

    def _on_use(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        s = self.suggestions[rows[0].row()]
        self.scan_accepted.emit({
            'name': f"MW drift — {s.galactic_region}",
            'altitude': s.altitude_deg,
            'azimuth': s.azimuth_deg,
            'duration_seconds': s.total_duration_min * 60,
            'resolution': 5.0,   # default beam width
            'start_time': s.start_time.replace(tzinfo=None),  # naive for DB
            'notes': (
                f"Suggested drift scan across {s.galactic_region} "
                f"(l={s.galactic_longitude_deg:.1f}°).  "
                f"Galactic plane peaks in beam around "
                f"{s.peak_time.strftime('%H:%M')}.  "
                f"Brightness: {s.brightness:.0%}."
            ),
        })
        self.accept()
