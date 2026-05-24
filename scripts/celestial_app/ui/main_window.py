import sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import numpy as np
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import Qt

from visualization.gl_widgets import FixedGLViewWidget
from visualization.scene_builder import SceneBuilder
from astronomy.time_utils import julian_day, gmst_degrees
from astronomy.coordinates import (
    clamp_lat_lon,
    equatorial_to_local_enu_matrix,
    unit_vector_enu_to_alt_az,
    alt_az_from_ra_dec,
    make_horizon_ring_ecef,
    ra_dec_to_unit_vector_equatorial,
    latlon_to_ecef
)
from astronomy.celestial_objects import sun_ra_dec_degrees, galactic_center_unit_eq
from geometry.mesh_generation import make_disk_mesh, make_uv_sphere
from geometry.transformations import normalize_vector, rotz_deg
import pyqtgraph.opengl as gl

# Import view modes
from .globe_view import GlobeView
from .star_chart_view import StarChartView
from .scan_suggestion_dialog import ScanSuggestionDialog

# Radio Telescope imports
from database import ScanDatabase
from ui.scan_dialog import ScanEntryDialog
from radio_telescope import ScanPath
from radio_telescope.scan_planner import suggest_scans

APP_TZ = ZoneInfo("America/New_York")
EARTH_ROT_SIGN = 1.0
SUN_ANG_RADIUS_DEG = 0.265

# Debug toggle - set to True to enable debug output
DEBUG_ENABLED = False

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Celestial Sphere - 3D & Star Chart")
        self.scene_builder = SceneBuilder(radius=1.0, earth_radius=0.36)
        self.radius = self.scene_builder.radius
        self.earth_radius = self.scene_builder.earth_radius
        self.lat = 39.9612
        self.lon = -82.9988
        self.dt_local = datetime.now(APP_TZ)
        self.dt_utc = self.dt_local.astimezone(timezone.utc)

        # Radio Telescope: Initialize database and scan storage
        self.scan_db = ScanDatabase()
        self.scan_paths = []
        self.scan_mesh_items = []
        
        # Create central widget for QMainWindow
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Build UI on central widget
        self._build_ui(central_widget)
        self._build_scene()

        # Radio Telescope: Create menu bar
        self._create_menu_bar()

        # Radio Telescope: Load saved scans
        self._load_scans()
        
        # Initialize view mode
        self._init_view_mode()
        
        # Start with initial update
        self.update_all_views()
    
    def _build_ui(self, parent):
        """Build Qt UI components - controls on left, views on right"""
        layout = QtWidgets.QHBoxLayout(parent)
        
        # Left panel with controls
        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, 0)
        
        self.lat_edit = QtWidgets.QLineEdit(str(self.lat))
        self.lon_edit = QtWidgets.QLineEdit(str(self.lon))
        self.time_edit = QtWidgets.QLineEdit(self.dt_local.strftime("%Y-%m-%d %H:%M:%S"))
        
        left.addWidget(QtWidgets.QLabel("Latitude"))
        left.addWidget(self.lat_edit)
        left.addWidget(QtWidgets.QLabel("Longitude"))
        left.addWidget(self.lon_edit)
        left.addWidget(QtWidgets.QLabel("Date and time (America/New_York)"))
        left.addWidget(self.time_edit)
        
        btn_row = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.now_btn = QtWidgets.QPushButton("Now")
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.now_btn)
        left.addLayout(btn_row)
        
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        left.addWidget(self.info)
        left.addStretch(1)
        
        # Right side: Tabbed view for 3D and 2D
        self.tabs = QtWidgets.QTabWidget()
        
        # 3D Globe View tab
        self.globe_widget = QtWidgets.QWidget()
        globe_layout = QtWidgets.QVBoxLayout(self.globe_widget)
        self.view = FixedGLViewWidget()
        self.view.setBackgroundColor((10, 10, 14))
        self.view.set_fixed_distance(2.6)
        globe_layout.addWidget(self.view)
        self.tabs.addTab(self.globe_widget, "3D Globe View")
        
        # 2D Star Chart tab
        self.star_chart = StarChartView(radius=self.radius)
        self.tabs.addTab(self.star_chart, "2D Star Chart (Printable)")
        
        layout.addWidget(self.tabs, 1)  # Takes up remaining space
        
        # Connect signals
        self.apply_btn.clicked.connect(self.on_apply)
        self.now_btn.clicked.connect(self.on_now)
        self.tabs.currentChanged.connect(self.on_tab_changed)
    
    def _build_scene(self):
        """Build 3D scene objects"""
        # Celestial sphere
        self.sky_item = self.scene_builder.build_celestial_sphere()
        self.view.addItem(self.sky_item)
        
        # Earth
        self.earth_item = self.scene_builder.build_earth()
        self.view.addItem(self.earth_item)
        
        # Location marker
        self.loc_marker = self.scene_builder.build_location_marker()
        self.view.addItem(self.loc_marker)
        
        # Horizon ring
        hz0 = make_horizon_ring_ecef(self.radius, self.lat, self.lon, n=600)
        self.horizon_item = self.scene_builder.build_horizon_ring(hz0)
        self.view.addItem(self.horizon_item)
        
        # Celestial equator
        self.eq_item = self.scene_builder.build_celestial_equator()
        self.view.addItem(self.eq_item)
        
        # Milky Way
        self.mw_item, self.mw_pts_eq, self.mw_cols = self.scene_builder.build_milky_way()
        self.view.addItem(self.mw_item)
        
        # Sun
        self.sun_disk_item, self.sun_dot = self.scene_builder.build_sun()
        self.view.addItem(self.sun_disk_item)
        self.view.addItem(self.sun_dot)
        
        # Earth axis
        self.earth_axis_item = self.scene_builder.build_earth_axis()
        self.view.addItem(self.earth_axis_item)
        
        # Galactic center
        self.gc_dot = self.scene_builder.build_galactic_center_dot()
        self.view.addItem(self.gc_dot)
        
        # Compass directions (N, S, E, W)
        self._build_compass_markers()
    
    def _build_compass_markers(self):
        """Create compass direction markers (N, S, E, W) on horizon"""
        self.compass_markers = {}
        
        for direction in ['N', 'S', 'E', 'W']:
            marker_size = 0.02
            verts, faces, _ = make_uv_sphere(marker_size, n_lon=16, n_lat=8)
            md = gl.MeshData(vertexes=verts, faces=faces)
            marker = gl.GLMeshItem(meshdata=md, smooth=True, drawEdges=False, shader="shaded")
            
            colors = {
                'N': (1.0, 0.0, 0.0, 1.0),  # Red
                'S': (0.0, 0.5, 1.0, 1.0),  # Blue
                'E': (1.0, 1.0, 0.0, 1.0),  # Yellow
                'W': (1.0, 1.0, 1.0, 1.0),  # White
            }
            marker.setColor(colors[direction])
            marker.setGLOptions("translucent")
            marker.setDepthValue(40000)
            marker.setVisible(False)
            self.view.addItem(marker)
            self.compass_markers[direction] = marker
    
    def _init_view_mode(self):
        """Initialize view mode controller"""
        # Create scene items dictionary for view mode
        self.scene_items = {
            'sky': self.sky_item,
            'earth': self.earth_item,
            'loc_marker': self.loc_marker,
            'horizon': self.horizon_item,
            'eq': self.eq_item,
            'mw': self.mw_item,
            'sun_disk': self.sun_disk_item,
            'sun_dot': self.sun_dot,
            'earth_axis': self.earth_axis_item,
            'gc_dot': self.gc_dot,
            'compass_markers': self.compass_markers
        }
        
        # Initialize globe view controller
        self.globe_view = GlobeView(
            self.view, 
            self.scene_items, 
            self.radius, 
            self.earth_radius
        )
    
    def on_now(self):
        """Set time to current"""
        self.dt_local = datetime.now(APP_TZ)
        self.time_edit.setText(self.dt_local.strftime("%Y-%m-%d %H:%M:%S"))
        self.on_apply()
    
    def on_apply(self):
        """Apply user inputs"""
        errors = []
        try:
            lat = float(self.lat_edit.text().strip())
            lon = float(self.lon_edit.text().strip())
            self.lat, self.lon = clamp_lat_lon(lat, lon)
        except ValueError:
            errors.append("lat/lon must be numbers")

        try:
            dt = datetime.strptime(self.time_edit.text().strip(), "%Y-%m-%d %H:%M:%S")
            self.dt_local = dt.replace(tzinfo=APP_TZ)
        except ValueError:
            errors.append("time must be YYYY-MM-DD HH:MM:SS (reset to now)")
            self.dt_local = datetime.now(APP_TZ)
            self.time_edit.setText(self.dt_local.strftime("%Y-%m-%d %H:%M:%S"))

        self.dt_utc = self.dt_local.astimezone(timezone.utc)

        if errors:
            self.info.setText("Input error: " + "; ".join(errors))
            return

        self.update_all_views()
    
    def on_tab_changed(self, index):
        """Handle tab changes"""
        # Refresh the current view when switching tabs
        self.update_all_views()
    
    def update_all_views(self):
        """Update both 3D globe and 2D star chart"""
        dt_utc_naive = self.dt_utc.replace(tzinfo=None)
        jd = julian_day(dt_utc_naive)
        gmst = gmst_degrees(dt_utc_naive)
        lst = (gmst + self.lon) % 360.0
        
        # Update 3D globe view
        Rearth, p_view = self.globe_view.update_scene(
            self.lat, self.lon, gmst, EARTH_ROT_SIGN
        )
        
        # Update celestial objects in 3D view
        self._update_celestial_objects(dt_utc_naive, lst)
        
        # Pass scan paths to star chart
        self.star_chart.set_scan_paths(self.scan_paths)
        
        # Update 2D star chart
        self.star_chart.update_chart(
            self.lat, self.lon, lst, self.dt_local, dt_utc_naive
        )
        
        # Update info display
        sun_ra, sun_dec = sun_ra_dec_degrees(dt_utc_naive)
        sun_eq = ra_dec_to_unit_vector_equatorial(sun_ra, sun_dec)
        M = equatorial_to_local_enu_matrix(self.lat, lst)
        sun_local = (M @ sun_eq.reshape(3, 1)).ravel().astype(np.float32)
        sun_local = normalize_vector(sun_local)
        alt_vec, az_vec = unit_vector_enu_to_alt_az(sun_local)
        
        self.info.setText(
            f"Lat {self.lat:.4f}  Lon {self.lon:.4f}\n"
            f"Local {self.dt_local.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"UTC {self.dt_utc.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Sun alt {alt_vec:.2f}° az {az_vec:.2f}°"
        )
        
        # Debug output
        if DEBUG_ENABLED:
            from debug.debug_output import print_celestial_debug, print_gc_visibility
            alt_formula, az_formula = alt_az_from_ra_dec(self.lat, lst, sun_ra, sun_dec)
            gc_eq = galactic_center_unit_eq()
            print_celestial_debug(
                self.dt_local, self.dt_utc, self.lat, self.lon, jd, gmst, lst,
                sun_ra, sun_dec, sun_eq, sun_local, alt_vec, az_vec, 
                alt_formula, az_formula, M, 
                latlon_to_ecef(self.lat, self.lon, self.earth_radius), 
                p_view, gc_eq, self.mw_pts_eq, self.radius, 
                EARTH_ROT_SIGN, APP_TZ
            )
            print_gc_visibility(self.lat, self.lon, self.dt_local, APP_TZ)
    
    def _update_celestial_objects(self, dt_utc_naive, lst):
        """Update celestial objects (Milky Way, Sun, Galactic Center)"""
        # Milky Way (already in equatorial coords, no rotation needed)
        self.mw_item.setData(pos=self.mw_pts_eq, color=self.mw_cols)
        
        # Sun position and disk
        sun_ra, sun_dec = sun_ra_dec_degrees(dt_utc_naive)
        sun_eq = ra_dec_to_unit_vector_equatorial(sun_ra, sun_dec)
        sun_pos = (self.radius * sun_eq).astype(np.float32)
        
        self.sun_dot.setData(pos=sun_pos.reshape(1, 3))
        
        # Realistic sun disk size
        disk_radius = self.radius * np.sin(np.deg2rad(SUN_ANG_RADIUS_DEG))
        self.sun_disk_item.setMeshData(meshdata=make_disk_mesh(
            center=sun_pos,
            normal=sun_eq.astype(np.float32),
            radius=disk_radius,
            segments=56
        ))
        
        # Galactic center
        gc_eq = galactic_center_unit_eq()
        gc_pos = (self.radius * gc_eq).astype(np.float32)
        self.gc_dot.setData(pos=gc_pos.reshape(1, 3))
    
    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # Radio Telescope menu
        radio_menu = menubar.addMenu("Radio Telescope")
        
        suggest_action = QtGui.QAction("Suggest Optimal Scan...", self)
        suggest_action.triggered.connect(self._on_suggest_scan)
        radio_menu.addAction(suggest_action)

        radio_menu.addSeparator()

        new_scan_action = QtGui.QAction("New Scan...", self)
        new_scan_action.triggered.connect(self._on_new_scan)
        radio_menu.addAction(new_scan_action)
    
    def _on_suggest_scan(self):
        """Compute and display optimal drift-scan suggestions."""
        dt_utc_naive = self.dt_utc.replace(tzinfo=None)

        suggestions = suggest_scans(
            lat_deg=self.lat,
            lon_deg=self.lon,
            dt_utc_naive=dt_utc_naive,
            local_tz=APP_TZ,
            min_alt_deg=20.0,
            lookahead_hours=24.0,
            beam_width_deg=5.0,
        )

        if not suggestions:
            QtWidgets.QMessageBox.information(
                self,
                "No Suggestions",
                "No observable Milky Way crossings found in the next 24 hours\n"
                "from your current location and time.\n\n"
                "Try adjusting the date or lowering the minimum altitude."
            )
            return

        dlg = ScanSuggestionDialog(suggestions, parent=self)
        dlg.scan_accepted.connect(self._on_suggestion_accepted)
        dlg.exec()

    def _on_suggestion_accepted(self, scan_data: dict):
        """Open the scan entry dialog pre-filled with the suggested values."""
        dialog = ScanEntryDialog(self)
        dialog.prefill(scan_data)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            confirmed = dialog.get_scan_data()
            self.scan_db.add_scan(
                name=confirmed['name'],
                altitude=confirmed['altitude'],
                azimuth=confirmed['azimuth'],
                duration_seconds=confirmed['duration_seconds'],
                resolution=confirmed['resolution'],
                start_time=confirmed['start_time'],
                notes=confirmed['notes'],
            )
            self._add_scan_visualization(confirmed)
            self.update_all_views()
            QtWidgets.QMessageBox.information(
                self,
                "Scan Saved",
                f"Scan '{confirmed['name']}' has been saved."
            )

    def _on_new_scan(self):
        """Handle new scan menu action."""
        dialog = ScanEntryDialog(self)
        
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            scan_data = dialog.get_scan_data()
            
            # Save to database
            scan_id = self.scan_db.add_scan(
                name=scan_data['name'],
                altitude=scan_data['altitude'],
                azimuth=scan_data['azimuth'],
                duration_seconds=scan_data['duration_seconds'],
                resolution=scan_data['resolution'],
                start_time=scan_data['start_time'],
                notes=scan_data['notes']
            )
            
            # Create scan path and visualize
            self._add_scan_visualization(scan_data)
            
            # Refresh views
            self.update_all_views()
            
            QtWidgets.QMessageBox.information(
                self,
                "Scan Saved",
                f"Scan '{scan_data['name']}' has been saved."
            )
    
    def _load_scans(self):
        """Load all scans from database and create visualizations."""
        scans = self.scan_db.get_all_scans()
        
        for scan_data in scans:
            self._add_scan_visualization(scan_data)
    
    def _add_scan_visualization(self, scan_data):
        """
        Create and add scan path visualization.
        
        Args:
            scan_data: Dictionary with scan parameters
        """
        # Create ScanPath object
        scan_path = ScanPath(
            altitude=scan_data['altitude'],
            azimuth=scan_data['azimuth'],
            duration_seconds=scan_data['duration_seconds'],
            resolution=scan_data['resolution'],
            start_time=scan_data['start_time'],
            latitude=self.lat,
            longitude=self.lon
        )
        
        self.scan_paths.append(scan_path)
        
        # Create 3D mesh for globe view
        vertices, faces = scan_path.get_band_mesh_3d(radius=self.radius, segments=16)
        
        if len(vertices) > 0 and len(faces) > 0:
            md = gl.MeshData(vertexes=vertices, faces=faces)
            mesh_item = gl.GLMeshItem(
                meshdata=md,
                smooth=True,
                drawEdges=False,
                shader="shaded",
                glOptions="translucent"
            )
            # Red semi-transparent
            mesh_item.setColor((1.0, 0.45, 0.45, 0.35))
            
            self.view.addItem(mesh_item)
            self.scan_mesh_items.append(mesh_item)