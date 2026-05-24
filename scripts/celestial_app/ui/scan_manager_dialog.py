"""
Scan Manager Dialog — list, edit, and delete saved radio telescope scans.
"""
from __future__ import annotations

from datetime import timezone
from PyQt6 import QtWidgets, QtCore

from ui.scan_dialog import ScanEntryDialog


class ScanManagerDialog(QtWidgets.QDialog):
    """
    Shows all saved scans in a table.  The user can edit or delete any row.

    Signals
    -------
    scan_edited(scan_id, new_data)
        Emitted when the user confirms an edit.  ``scan_id`` is the DB primary
        key; ``new_data`` is the dict returned by ``ScanEntryDialog.get_scan_data()``.
    scan_deleted(scan_id)
        Emitted when the user confirms deletion of a scan.
    """

    scan_edited = QtCore.pyqtSignal(int, dict)
    scan_deleted = QtCore.pyqtSignal(int)

    # Column indices
    _COL_NAME = 0
    _COL_ALT  = 1
    _COL_AZ   = 2
    _COL_DUR  = 3
    _COL_START = 4
    _COL_NOTES = 5

    def __init__(self, scan_items: list, local_tz=None, parent=None):
        """
        Parameters
        ----------
        scan_items:
            Reference to ``MainWindow.scan_items`` — a list of dicts with keys
            ``id``, ``data``, ``path``, ``mesh``.  The dialog reads this list
            directly; callers must keep it up-to-date when signals are handled.
        local_tz:
            Optional tzinfo (e.g. ``ZoneInfo("America/New_York")``).  When
            provided, UTC times stored in ``data`` are converted to local time
            for display and for pre-filling the edit dialog.
        """
        super().__init__(parent)
        self.setWindowTitle("Manage Scans")
        self.setMinimumWidth(700)
        self.setMinimumHeight(380)
        self._scan_items = scan_items
        self._local_tz = local_tz
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Alt", "Az", "Duration", "Start Time", "Notes"]
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)

        self._populate_table()
        self.table.resizeColumnsToContents()
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        root.addWidget(self.table)

        # Button row
        btn_row = QtWidgets.QHBoxLayout()

        self.edit_btn = QtWidgets.QPushButton("Edit Scan…")
        self.edit_btn.clicked.connect(self._on_edit)
        btn_row.addWidget(self.edit_btn)

        self.delete_btn = QtWidgets.QPushButton("Delete Scan")
        self.delete_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self.delete_btn)

        btn_row.addStretch()

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)

        root.addLayout(btn_row)

        # Initialise button state
        self._on_selection_changed()

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    def _populate_table(self):
        self.table.setRowCount(len(self._scan_items))
        for row, item in enumerate(self._scan_items):
            self._write_row(row, item)

    def _write_row(self, row: int, item: dict):
        """Fill one table row from a scan_items entry."""
        d = item["data"]

        def cell(text: str, align=QtCore.Qt.AlignmentFlag.AlignCenter):
            w = QtWidgets.QTableWidgetItem(text)
            w.setTextAlignment(align)
            # Tag every cell in this row with the scan id for easy retrieval
            w.setData(QtCore.Qt.ItemDataRole.UserRole, item["id"])
            return w

        # Duration: convert seconds to h m s
        dur = int(d.get("duration_seconds", 0))
        h, rem = divmod(dur, 3600)
        m, s = divmod(rem, 60)
        dur_str = f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

        start_utc = d.get("start_time")
        if start_utc and self._local_tz:
            try:
                start_local = start_utc.replace(tzinfo=timezone.utc).astimezone(self._local_tz)
                start_str = start_local.strftime("%Y-%m-%d %H:%M")
            except Exception:
                start_str = start_utc.strftime("%Y-%m-%d %H:%M")
        elif start_utc:
            start_str = start_utc.strftime("%Y-%m-%d %H:%M") + " UTC"
        else:
            start_str = "—"

        self.table.setItem(
            row, self._COL_NAME,
            cell(d.get("name", ""), QtCore.Qt.AlignmentFlag.AlignLeft)
        )
        self.table.setItem(row, self._COL_ALT,  cell(f"{d.get('altitude', 0):.1f}°"))
        self.table.setItem(row, self._COL_AZ,   cell(f"{d.get('azimuth', 0):.1f}°"))
        self.table.setItem(row, self._COL_DUR,  cell(dur_str))
        self.table.setItem(row, self._COL_START, cell(start_str))
        self.table.setItem(
            row, self._COL_NOTES,
            cell(d.get("notes", ""), QtCore.Qt.AlignmentFlag.AlignLeft)
        )

    def _selected_row_and_item(self):
        """Return (row_index, scan_item) for the currently selected row, or (None, None)."""
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None, None
        row = rows[0].row()
        # Recover the scan id from the cell tag
        scan_id = self.table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
        # Find matching entry in scan_items (in case the list was reordered)
        for item in self._scan_items:
            if item["id"] == scan_id:
                return row, item
        return None, None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_selection_changed(self):
        has = bool(self.table.selectionModel().selectedRows())
        self.edit_btn.setEnabled(has)
        self.delete_btn.setEnabled(has)

    def _on_edit(self):
        row, item = self._selected_row_and_item()
        if item is None:
            return

        dialog = ScanEntryDialog(self)

        # DB stores UTC; convert to local so the user sees a recognisable time
        display_data = dict(item["data"])
        start_utc = display_data.get("start_time")
        if start_utc and self._local_tz:
            try:
                display_data["start_time"] = (
                    start_utc.replace(tzinfo=timezone.utc)
                    .astimezone(self._local_tz)
                    .replace(tzinfo=None)
                )
            except Exception:
                pass
        dialog.prefill(display_data)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_data = dialog.get_scan_data()
            # new_data['start_time'] is local naive — MainWindow (_apply_scan_edit)
            # will convert it to UTC before storing.
            self.scan_edited.emit(item["id"], new_data)
            # Refresh our table row (item['data'] is updated by MainWindow)
            self._write_row(row, item)
            self.table.resizeColumnsToContents()

    def _on_delete(self):
        row, item = self._selected_row_and_item()
        if item is None:
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete Scan",
            f"Permanently delete scan '{item['data'].get('name', '')}'?",
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        # Notify MainWindow — it will remove from DB, scene, and scan_items
        self.scan_deleted.emit(item["id"])
        # Remove from our table (scan_items is now one shorter)
        self.table.removeRow(row)
        self.table.resizeColumnsToContents()
