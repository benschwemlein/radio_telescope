"""
Radio Telescope Scan Entry Dialog
"""
from PyQt6 import QtWidgets, QtCore
from datetime import datetime


class ScanEntryDialog(QtWidgets.QDialog):
    """Dialog for entering radio telescope scan parameters."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Radio Telescope Scan")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Form layout for inputs
        form_layout = QtWidgets.QFormLayout()
        
        # Scan name
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("e.g., Hydrogen Line Survey")
        form_layout.addRow("Scan Name:", self.name_edit)
        
        # Altitude (elevation angle)
        self.altitude_spin = QtWidgets.QDoubleSpinBox()
        self.altitude_spin.setRange(0.0, 90.0)
        self.altitude_spin.setValue(45.0)
        self.altitude_spin.setSuffix("°")
        self.altitude_spin.setDecimals(2)
        form_layout.addRow("Altitude (Elevation):", self.altitude_spin)
        
        # Azimuth
        self.azimuth_spin = QtWidgets.QDoubleSpinBox()
        self.azimuth_spin.setRange(0.0, 360.0)
        self.azimuth_spin.setValue(180.0)
        self.azimuth_spin.setSuffix("°")
        self.azimuth_spin.setDecimals(2)
        self.azimuth_spin.setWrapping(True)  # Wrap around at 360
        form_layout.addRow("Azimuth (Direction):", self.azimuth_spin)
        
        # Duration (hours, minutes, seconds)
        duration_layout = QtWidgets.QHBoxLayout()
        
        self.hours_spin = QtWidgets.QSpinBox()
        self.hours_spin.setRange(0, 48)
        self.hours_spin.setValue(2)
        self.hours_spin.setSuffix(" h")
        duration_layout.addWidget(self.hours_spin)
        
        self.minutes_spin = QtWidgets.QSpinBox()
        self.minutes_spin.setRange(0, 59)
        self.minutes_spin.setValue(0)
        self.minutes_spin.setSuffix(" m")
        duration_layout.addWidget(self.minutes_spin)
        
        self.seconds_spin = QtWidgets.QSpinBox()
        self.seconds_spin.setRange(0, 59)
        self.seconds_spin.setValue(0)
        self.seconds_spin.setSuffix(" s")
        duration_layout.addWidget(self.seconds_spin)
        
        duration_layout.addStretch()
        
        form_layout.addRow("Duration:", duration_layout)
        
        # Resolution (beam width)
        self.resolution_spin = QtWidgets.QDoubleSpinBox()
        self.resolution_spin.setRange(0.1, 90.0)
        self.resolution_spin.setValue(1.0)
        self.resolution_spin.setSuffix("°")
        self.resolution_spin.setDecimals(2)
        form_layout.addRow("Beam Width:", self.resolution_spin)
        
        # Start time
        start_time_layout = QtWidgets.QHBoxLayout()
        
        self.start_datetime = QtWidgets.QDateTimeEdit()
        self.start_datetime.setDateTime(QtCore.QDateTime.currentDateTime())
        self.start_datetime.setCalendarPopup(True)
        self.start_datetime.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        start_time_layout.addWidget(self.start_datetime)
        
        self.now_button = QtWidgets.QPushButton("Use Now")
        self.now_button.clicked.connect(self._set_now)
        start_time_layout.addWidget(self.now_button)
        
        form_layout.addRow("Start Time:", start_time_layout)
        
        # Notes (optional)
        self.notes_edit = QtWidgets.QTextEdit()
        self.notes_edit.setMaximumHeight(60)
        self.notes_edit.setPlaceholderText("Optional notes about this scan...")
        form_layout.addRow("Notes:", self.notes_edit)
        
        layout.addLayout(form_layout)
        
        # Add some spacing
        layout.addSpacing(10)
        
        # Info label
        info_label = QtWidgets.QLabel(
            "The telescope will point at the specified altitude and azimuth.\n"
            "Earth's rotation will scan across the sky for the specified duration."
        )
        info_label.setStyleSheet("color: #808080; font-size: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addSpacing(10)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        self.save_button = QtWidgets.QPushButton("Save Scan")
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self.accept)
        button_layout.addWidget(self.save_button)
        
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def _set_now(self):
        """Set start time to current time."""
        self.start_datetime.setDateTime(QtCore.QDateTime.currentDateTime())
    
    def get_scan_data(self):
        """
        Get the scan parameters from the dialog.
        
        Returns:
            Dictionary with scan parameters
        """
        # Convert duration to seconds
        duration_seconds = (
            self.hours_spin.value() * 3600 +
            self.minutes_spin.value() * 60 +
            self.seconds_spin.value()
        )
        
        # Convert QDateTime to Python datetime
        qdt = self.start_datetime.dateTime()
        start_time = datetime(
            qdt.date().year(),
            qdt.date().month(),
            qdt.date().day(),
            qdt.time().hour(),
            qdt.time().minute(),
            qdt.time().second()
        )
        
        return {
            'name': self.name_edit.text().strip() or "Unnamed Scan",
            'altitude': self.altitude_spin.value(),
            'azimuth': self.azimuth_spin.value(),
            'duration_seconds': duration_seconds,
            'resolution': self.resolution_spin.value(),
            'start_time': start_time,
            'notes': self.notes_edit.toPlainText().strip()
        }
    
    def validate_input(self):
        """
        Validate the input values.
        
        Returns:
            (bool, str): (is_valid, error_message)
        """
        # Check duration
        duration_seconds = (
            self.hours_spin.value() * 3600 +
            self.minutes_spin.value() * 60 +
            self.seconds_spin.value()
        )
        
        if duration_seconds <= 0:
            return False, "Duration must be greater than zero."
        
        if duration_seconds > 48 * 3600:  # 48 hours max
            return False, "Duration cannot exceed 48 hours."
        
        # Check name
        if not self.name_edit.text().strip():
            # Auto-generate name if empty
            self.name_edit.setText(f"Scan {datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        return True, ""
    
    def accept(self):
        """Override accept to validate before closing."""
        is_valid, error_msg = self.validate_input()
        
        if not is_valid:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Input",
                error_msg
            )
            return
        
        super().accept()
