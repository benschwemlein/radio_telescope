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
from .scan_manager_dialog import ScanManagerDialog

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
        # Each entry: {'id': int, 'data': dict, 'path': ScanPath, 'mesh': GLMeshItem|None}
        self.scan_items: list = []
        
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
        self.view.scene_clicked.connect(self._on_globe_clicked)
    
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
    
    # ------------------------------------------------------------------
    # Timezone helpers
    # All scan times are stored in the DB as naive UTC.  These helpers
    # convert to/from APP_TZ (America/New_York) for dialog display.
    # ------------------------------------------------------------------

    def _utc_to_local(self, dt: datetime) -> datetime:
        """Naive UTC → naive APP_TZ local (for displaying to the user)."""
        return dt.replace(tzinfo=timezone.utc).astimezone(APP_TZ).replace(tzinfo=None)

    def _local_to_utc(self, dt: datetime) -> datetime:
        """Naive APP_TZ local → naive UTC (for storing in DB / ScanPath)."""
        return dt.replace(tzinfo=APP_TZ).astimezone(timezone.utc).replace(tzinfo=None)

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
        self.star_chart.set_scan_paths([item['path'] for item in self.scan_items])
        
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

        manage_action = QtGui.QAction("Manage Scans...", self)
        manage_action.triggered.connect(self._on_manage_scans)
        radio_menu.addAction(manage_action)
    
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

        # scan_data['start_time'] is UTC naive (from ScanSuggestionDialog).
        # Convert to local for a user-friendly display in the dialog.
        display_data = dict(scan_data)
        if display_data.get('start_time'):
            display_data['start_time'] = self._utc_to_local(display_data['start_time'])
        dialog.prefill(display_data)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            confirmed = dialog.get_scan_data()
            # Dialog returns naive local time → convert back to UTC for DB / ScanPath
            if confirmed.get('start_time'):
                confirmed['start_time'] = self._local_to_utc(confirmed['start_time'])
            scan_id = self.scan_db.add_scan(
                name=confirmed['name'],
                altitude=confirmed['altitude'],
                azimuth=confirmed['azimuth'],
                duration_seconds=confirmed['duration_seconds'],
                resolution=confirmed['resolution'],
                start_time=confirmed['start_time'],
                notes=confirmed['notes'],
            )
            self._add_scan_visualization(confirmed, scan_id)
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

            # Dialog returns naive local time (America/New_York).
            # Convert to UTC before storing in DB and passing to ScanPath.
            if scan_data.get('start_time'):
                scan_data['start_time'] = self._local_to_utc(scan_data['start_time'])

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
            self._add_scan_visualization(scan_data, scan_id)
            
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
            self._add_scan_visualization(scan_data, scan_data['id'])
    
    def _add_scan_visualization(self, scan_data: dict, scan_id: int):
        """
        Create a ScanPath, build a 3D mesh, and register everything in
        ``self.scan_items`` so the entry can later be edited or deleted.

        Args:
            scan_data: Dictionary with scan parameters (altitude, azimuth, …)
            scan_id:   Database primary key for this scan
        """
        scan_path = ScanPath(
            altitude=scan_data['altitude'],
            azimuth=scan_data['azimuth'],
            duration_seconds=scan_data['duration_seconds'],
            resolution=scan_data['resolution'],
            start_time=scan_data['start_time'],
            latitude=self.lat,
            longitude=self.lon
        )

        entry: dict = {
            'id':       scan_id,
            'data':     scan_data,
            'path':     scan_path,
            'mesh':     None,
            'centroid': None,   # 3-D world-space centroid used for click picking
        }

        # Build 3D mesh for the globe view
        vertices, faces = scan_path.get_band_mesh_3d(radius=self.radius, segments=16)
        if len(vertices) > 0 and len(faces) > 0:
            entry['centroid'] = vertices.mean(axis=0).astype(np.float32)

            md = gl.MeshData(vertexes=vertices, faces=faces)
            mesh_item = gl.GLMeshItem(
                meshdata=md,
                smooth=True,
                drawEdges=False,
                shader="shaded",
                glOptions="translucent"
            )
            mesh_item.setColor((1.0, 0.45, 0.45, 0.35))  # salmon, semi-transparent
            self.view.addItem(mesh_item)
            entry['mesh'] = mesh_item

        self.scan_items.append(entry)

    # ------------------------------------------------------------------
    # Scan management handlers
    # ------------------------------------------------------------------

    def _on_manage_scans(self):
        """Open the Manage Scans dialog."""
        if not self.scan_items:
            QtWidgets.QMessageBox.information(
                self,
                "No Scans",
                "There are no saved scans yet.\n"
                "Use Radio Telescope → New Scan… to add one."
            )
            return

        dlg = ScanManagerDialog(self.scan_items, local_tz=APP_TZ, parent=self)
        dlg.scan_edited.connect(self._on_scan_edited)
        dlg.scan_deleted.connect(self._on_scan_deleted)
        dlg.exec()

    def _on_scan_edited(self, scan_id: int, new_data: dict):
        """Slot connected to ScanManagerDialog.scan_edited."""
        self._apply_scan_edit(scan_id, new_data)

    def _apply_scan_edit(self, scan_id: int, new_data: dict):
        """
        Persist edits to the DB, rebuild the 3D mesh, and update the entry
        in place so any open ScanManagerDialog table row stays valid.

        ``new_data['start_time']`` is expected to be a naive local (APP_TZ)
        datetime as returned by ScanEntryDialog.get_scan_data().  It is
        converted to UTC here before being stored.
        """
        entry = next((it for it in self.scan_items if it['id'] == scan_id), None)
        if entry is None:
            return

        # Work with a copy so we don't mutate the caller's dict
        data_utc = dict(new_data)
        if data_utc.get('start_time'):
            data_utc['start_time'] = self._local_to_utc(data_utc['start_time'])

        # Persist to DB (UTC)
        self.scan_db.update_scan(
            scan_id,
            name=data_utc['name'],
            altitude=data_utc['altitude'],
            azimuth=data_utc['azimuth'],
            duration_seconds=data_utc['duration_seconds'],
            resolution=data_utc['resolution'],
            start_time=data_utc['start_time'],
            notes=data_utc['notes'],
        )

        # Remove old mesh from scene
        if entry['mesh'] is not None:
            self.view.removeItem(entry['mesh'])

        # Rebuild ScanPath and mesh with UTC start_time
        scan_path = ScanPath(
            altitude=data_utc['altitude'],
            azimuth=data_utc['azimuth'],
            duration_seconds=data_utc['duration_seconds'],
            resolution=data_utc['resolution'],
            start_time=data_utc['start_time'],
            latitude=self.lat,
            longitude=self.lon
        )

        new_mesh = None
        new_centroid = None
        vertices, faces = scan_path.get_band_mesh_3d(radius=self.radius, segments=16)
        if len(vertices) > 0 and len(faces) > 0:
            new_centroid = vertices.mean(axis=0).astype(np.float32)
            md = gl.MeshData(vertexes=vertices, faces=faces)
            new_mesh = gl.GLMeshItem(
                meshdata=md,
                smooth=True,
                drawEdges=False,
                shader="shaded",
                glOptions="translucent"
            )
            new_mesh.setColor((1.0, 0.45, 0.45, 0.35))
            self.view.addItem(new_mesh)

        # Update the entry in place (the dialog still holds a reference to it).
        # Store UTC so any subsequent prefill can convert correctly.
        entry['data']     = data_utc
        entry['path']     = scan_path
        entry['mesh']     = new_mesh
        entry['centroid'] = new_centroid

        self.update_all_views()

    def _on_scan_deleted(self, scan_id: int):
        """
        Handle a deletion confirmed in ScanManagerDialog.
        Removes the scan from the DB, the 3D scene, and ``scan_items``.
        """
        self._apply_scan_delete(scan_id)

    # ------------------------------------------------------------------
    # Shared edit / delete logic (used by both the manager dialog and the
    # 3-D globe click handler)
    # ------------------------------------------------------------------

    def _apply_scan_delete(self, scan_id: int):
        """Remove a scan from the DB, the 3D scene, and scan_items."""
        entry = next((it for it in self.scan_items if it['id'] == scan_id), None)
        if entry is None:
            return
        self.scan_db.delete_scan(scan_id)
        if entry['mesh'] is not None:
            self.view.removeItem(entry['mesh'])
        self.scan_items.remove(entry)
        self.update_all_views()

    def _open_edit_dialog_for(self, scan_id: int):
        """Open ScanEntryDialog pre-filled for the given scan_id.  On accept,
        persist the changes and refresh the scene."""
        entry = next((it for it in self.scan_items if it['id'] == scan_id), None)
        if entry is None:
            return
        dialog = ScanEntryDialog(self)

        # DB stores UTC; convert to local so the user sees a recognisable time
        display_data = dict(entry['data'])
        if display_data.get('start_time'):
            display_data['start_time'] = self._utc_to_local(display_data['start_time'])
        dialog.prefill(display_data)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_data = dialog.get_scan_data()
            # Dialog returns local time → _apply_scan_edit converts to UTC
            self._apply_scan_edit(scan_id, new_data)

    # ------------------------------------------------------------------
    # 3-D globe click-to-pick
    # ------------------------------------------------------------------

    def _on_globe_clicked(self, screen_x: int, screen_y: int):
        """
        Called when the user clicks (no drag) anywhere in the 3-D view.
        Projects every scan band's centroid to screen space and opens a
        context menu for the nearest one within 50 px.
        """
        if not self.scan_items:
            return

        # We need to be on the 3D tab; ignore clicks on the star chart.
        if self.tabs.currentIndex() != 0:
            return

        from PyQt6.QtGui import QVector4D

        try:
            P = self.view.projectionMatrix()
            V = self.view.viewMatrix()
        except Exception:
            return          # pyqtgraph version doesn't expose these — bail out

        PV = P * V
        w = max(self.view.width(), 1)
        h = max(self.view.height(), 1)

        THRESHOLD_SQ = 50 * 50      # 50-pixel click radius

        best_entry = None
        best_dist_sq = THRESHOLD_SQ

        for item in self.scan_items:
            centroid = item.get('centroid')
            if centroid is None:
                continue
            cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])
            clip = PV.map(QVector4D(cx, cy, cz, 1.0))
            if abs(clip.w()) < 1e-9:
                continue
            ndc_x = clip.x() / clip.w()
            ndc_y = clip.y() / clip.w()
            # NDC → pixel (OpenGL Y is bottom-up, Qt Y is top-down)
            sx = (ndc_x + 1.0) / 2.0 * w
            sy = (1.0 - ndc_y) / 2.0 * h
            dx = sx - screen_x
            dy = sy - screen_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_entry = item

        if best_entry is None:
            return

        # Build a small context menu at the click position
        from PyQt6.QtCore import QPoint
        scan_name = best_entry['data'].get('name', 'Scan')

        menu = QtWidgets.QMenu(self)

        title_action = menu.addAction(f"📡  {scan_name}")
        title_action.setEnabled(False)          # non-clickable label
        menu.addSeparator()

        edit_action   = menu.addAction("Edit Scan…")
        delete_action = menu.addAction("Delete Scan")

        global_pos = self.view.mapToGlobal(QPoint(screen_x, screen_y))
        chosen = menu.exec(global_pos)

        if chosen == edit_action:
            self._open_edit_dialog_for(best_entry['id'])
        elif chosen == delete_action:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Delete Scan",
                f"Permanently delete scan '{scan_name}'?",
                QtWidgets.QMessageBox.StandardButton.Yes |
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                self._apply_scan_delete(best_entry['id'])