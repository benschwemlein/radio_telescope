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
    latlon_to_ecef,
    make_horizon_ring_ecef,
    eq_to_gal_matrix_j2000,
    ecef_basis_at
)
from astronomy.celestial_objects import sun_ra_dec_degrees, galactic_center_unit_eq
from astronomy.galactic import build_milky_way_band_equatorial
from geometry.transformations import rotz_deg
from geometry.mesh_generation import make_ring, make_disk_mesh
import pyqtgraph.opengl as gl

APP_TZ = ZoneInfo("America/New_York")
EARTH_ROT_SIGN = 1.0
SUN_ANG_RADIUS_DEG = 0.265

# Debug toggle - set to True to enable debug output
DEBUG_ENABLED = False

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Celestial Sphere")
        self.scene_builder = SceneBuilder(radius=1.0, earth_radius=0.36)
        self.radius = self.scene_builder.radius
        self.earth_radius = self.scene_builder.earth_radius
        self.lat = 39.9612
        self.lon = -82.9988
        self.dt_local = datetime.now(APP_TZ)
        self.dt_utc = self.dt_local.astimezone(timezone.utc)
        self.ground_fov = 90.0
        self._build_ui()
        self._build_scene()
        self.update_scene()
    
    def _build_ui(self):
        """Build Qt UI components"""
        layout = QtWidgets.QHBoxLayout(self)
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
        
        # Add view mode buttons
        view_row = QtWidgets.QHBoxLayout()
        self.sphere_view_btn = QtWidgets.QPushButton("Sphere View")
        self.ground_view_btn = QtWidgets.QPushButton("Ground View")
        view_row.addWidget(self.sphere_view_btn)
        view_row.addWidget(self.ground_view_btn)
        left.addLayout(view_row)
        
        # Add zoom slider for ground view
        zoom_label = QtWidgets.QLabel("Ground View Zoom (FOV)")
        left.addWidget(zoom_label)
        
        zoom_layout = QtWidgets.QHBoxLayout()
        self.zoom_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(120)
        self.zoom_slider.setValue(90)
        self.zoom_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.zoom_slider.setTickInterval(10)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        
        self.zoom_value_label = QtWidgets.QLabel(f"{self.zoom_slider.value()}°")
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_value_label)
        left.addLayout(zoom_layout)
        
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        left.addWidget(self.info)
        left.addStretch(1)
        
        self.view = FixedGLViewWidget()
        self.view.setBackgroundColor((10, 10, 14))
        self.view.set_fixed_distance(2.6)
        layout.addWidget(self.view, 1)
        
        self.apply_btn.clicked.connect(self.on_apply)
        self.now_btn.clicked.connect(self.on_now)
        self.sphere_view_btn.clicked.connect(self.on_sphere_view)
        self.ground_view_btn.clicked.connect(self.on_ground_view)
        
        self.current_view_mode = "sphere"
    
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
        self.sun_item, self.sun_dot = self.scene_builder.build_sun()
        self.view.addItem(self.sun_item)
        self.view.addItem(self.sun_dot)
        
        # Earth axis
        self.earth_axis_item = self.scene_builder.build_earth_axis()
        self.view.addItem(self.earth_axis_item)
        
        # Galactic center
        self.gc_dot = self.scene_builder.build_galactic_center_dot()
        self.view.addItem(self.gc_dot)
        
        # Ground plane
        self._build_ground_plane()
        
        # Compass directions (N, S, E, W)
        self._build_compass_markers()
    
    def _build_ground_plane(self):
        """Create a large green ground plane for ground view"""
        size = 5.0
        y_offset = -0.001
        
        verts = np.array([
            [-size, -size, y_offset],
            [ size, -size, y_offset],
            [ size,  size, y_offset],
            [-size,  size, y_offset],
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)
        
        ground_md = gl.MeshData(vertexes=verts, faces=faces)
        self.ground_plane = gl.GLMeshItem(
            meshdata=ground_md,
            smooth=False,
            drawEdges=False,
            shader="shaded"
        )
        self.ground_plane.setColor((0.2, 0.5, 0.2, 1.0))
        self.ground_plane.setGLOptions("opaque")
        self.ground_plane.setVisible(False)
        self.view.addItem(self.ground_plane)
    
    def _build_compass_markers(self):
        """Create compass direction markers (N, S, E, W) on horizon"""
        self.compass_markers = {}
        
        for direction in ['N', 'S', 'E', 'W']:
            marker_size = 0.015
            from geometry.mesh_generation import make_uv_sphere
            verts, faces, _ = make_uv_sphere(marker_size, n_lon=12, n_lat=6)
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
            marker.setVisible(False)
            self.view.addItem(marker)
            self.compass_markers[direction] = marker
    
    def _update_compass_positions(self, p_view, Rearth):
        """Update compass marker positions on horizon"""
        east, north, up = ecef_basis_at(self.lat, self.lon)
        east_view = (Rearth @ east).astype(np.float32)
        north_view = (Rearth @ north).astype(np.float32)
        up_view = (Rearth @ up).astype(np.float32)
        
        horizon_dist = 0.8
        
        positions = {
            'N': p_view + horizon_dist * north_view,
            'S': p_view - horizon_dist * north_view,
            'E': p_view + horizon_dist * east_view,
            'W': p_view - horizon_dist * east_view,
        }
        
        for direction, pos in positions.items():
            self.compass_markers[direction].resetTransform()
            self.compass_markers[direction].translate(float(pos[0]), float(pos[1]), float(pos[2]))
    
    def _set_object_visibility(self, sphere_mode=True):
        """Set visibility of objects based on view mode"""
        if sphere_mode:
            # Sphere view
            self.sky_item.setVisible(True)
            self.earth_item.setVisible(True)
            self.loc_marker.setVisible(True)
            self.earth_axis_item.setVisible(True)
            self.ground_plane.setVisible(False)
            for marker in self.compass_markers.values():
                marker.setVisible(False)
            self.sun_item.setVisible(True)
            self.sun_dot.setVisible(True)
        else:
            # Ground view
            self.sky_item.setVisible(True)
            self.earth_item.setVisible(False)
            self.loc_marker.setVisible(False)
            self.earth_axis_item.setVisible(False)
            self.ground_plane.setVisible(True)
            for marker in self.compass_markers.values():
                marker.setVisible(True)
            self.sun_item.setVisible(True)
            self.sun_dot.setVisible(False)
    
    def on_now(self):
        """Set time to current"""
        self.dt_local = datetime.now(APP_TZ)
        self.time_edit.setText(self.dt_local.strftime("%Y-%m-%d %H:%M:%S"))
        self.on_apply()
    
    def on_apply(self):
        """Apply user inputs"""
        try:
            lat = float(self.lat_edit.text().strip())
            lon = float(self.lon_edit.text().strip())
            lat, lon = clamp_lat_lon(lat, lon)
            self.lat = lat
            self.lon = lon
        except Exception:
            pass
        try:
            dt = datetime.strptime(self.time_edit.text().strip(), "%Y-%m-%d %H:%M:%S")
            self.dt_local = dt.replace(tzinfo=APP_TZ)
        except Exception:
            self.dt_local = datetime.now(APP_TZ)
        self.dt_utc = self.dt_local.astimezone(timezone.utc)
        self.update_scene()
    
    def on_zoom_changed(self, value):
        """Handle zoom slider changes"""
        self.ground_fov = float(value)
        self.zoom_value_label.setText(f"{value}°")
        if self.current_view_mode == "ground":
            self.on_ground_view()
    
    def on_sphere_view(self):
        """Set camera to sphere view (external view of celestial sphere)"""
        self.current_view_mode = "sphere"
        self._set_object_visibility(sphere_mode=True)
        
        self.view.opts['center'] = QtGui.QVector3D(0, 0, 0)
        self.view.opts['elevation'] = 30
        self.view.opts['azimuth'] = 45
        self.view.opts['distance'] = 2.6
        self.view.opts['fov'] = 60
        self.view.update()
    
    def on_ground_view(self):
        """Set camera to ground view (observer on Earth looking south at horizon)"""
        self.current_view_mode = "ground"
        self._set_object_visibility(sphere_mode=False)
        
        p_ecef = latlon_to_ecef(self.lat, self.lon, self.earth_radius)
        dt_utc_naive = self.dt_utc.replace(tzinfo=None)
        gmst = gmst_degrees(dt_utc_naive)
        Rearth = rotz_deg(EARTH_ROT_SIGN * gmst)
        p_view = (Rearth @ p_ecef).astype(np.float32)
        
        east, north, up = ecef_basis_at(self.lat, self.lon)
        east_view = (Rearth @ east).astype(np.float32)
        north_view = (Rearth @ north).astype(np.float32)
        up_view = (Rearth @ up).astype(np.float32)
        
        # Update compass positions
        self._update_compass_positions(p_view, Rearth)
        
        # Position ground plane
        self.ground_plane.resetTransform()
        self.ground_plane.translate(float(p_view[0]), float(p_view[1]), float(p_view[2]))
        
        cam_pos = p_view
        look_dir = -north_view
        look_dir = look_dir / (np.linalg.norm(look_dir) + 1e-12)
        
        self.view.opts['center'] = QtGui.QVector3D(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        self.view.opts['distance'] = 0.01
        self.view.opts['fov'] = self.ground_fov
        
        azimuth = np.rad2deg(np.arctan2(look_dir[1], look_dir[0]))
        self.view.opts['azimuth'] = float(azimuth)
        self.view.opts['elevation'] = 0.0
        self.view.update()
    
    def update_scene(self):
        """Update scene for current time and location"""
        dt_utc_naive = self.dt_utc.replace(tzinfo=None)
        jd = julian_day(dt_utc_naive)
        gmst = gmst_degrees(dt_utc_naive)
        lst = (gmst + self.lon) % 360.0
        
        # EARTH FRAME
        Rearth = rotz_deg(EARTH_ROT_SIGN * gmst)
        
        self.earth_item.resetTransform()
        self.earth_item.rotate(EARTH_ROT_SIGN * gmst, 0, 0, 1)
        
        p_ecef = latlon_to_ecef(self.lat, self.lon, self.earth_radius)
        p_view = (Rearth @ p_ecef).astype(np.float32)
        self.loc_marker.resetTransform()
        self.loc_marker.translate(float(p_view[0]), float(p_view[1]), float(p_view[2]))
        
        horizon_ecef = make_horizon_ring_ecef(self.radius, self.lat, self.lon, n=600)
        horizon_view = (Rearth @ horizon_ecef.T).T.astype(np.float32)
        self.horizon_item.setData(pos=horizon_view)
        
        axis_pts = np.array([[0.0, 0.0, -self.radius],
                             [0.0, 0.0,  self.radius]], dtype=np.float32)
        self.earth_axis_item.setData(pos=axis_pts)
        
        # CELESTIAL FRAME
        eq = make_ring(self.radius, 600, "xy")
        self.eq_item.setData(pos=eq)
        
        self.mw_item.setData(pos=self.mw_pts_eq, color=self.mw_cols)
        
        # Sun
        sun_ra, sun_dec = sun_ra_dec_degrees(dt_utc_naive)
        from astronomy.coordinates import ra_dec_to_unit_vector_equatorial
        sun_eq = ra_dec_to_unit_vector_equatorial(sun_ra, sun_dec)
        sun_pos = (self.radius * sun_eq).astype(np.float32)
        
        self.sun_dot.setData(pos=sun_pos.reshape(1, 3))
        
        # Realistic sun disk size
        disk_radius = self.radius * np.sin(np.deg2rad(SUN_ANG_RADIUS_DEG))
        self.sun_item.setMeshData(meshdata=make_disk_mesh(
            center=sun_pos,
            normal=sun_eq.astype(np.float32),
            radius=disk_radius,
            segments=56
        ))
        
        # Calculate sun position
        M = equatorial_to_local_enu_matrix(self.lat, lst)
        sun_local = (M @ sun_eq.reshape(3, 1)).ravel().astype(np.float32)
        sun_local = sun_local / (np.linalg.norm(sun_local) + 1e-12)
        alt_vec, az_vec = unit_vector_enu_to_alt_az(sun_local)
        alt_formula, az_formula = alt_az_from_ra_dec(self.lat, lst, sun_ra, sun_dec)
        
        self.info.setText(
            f"Lat {self.lat:.4f}  Lon {self.lon:.4f}\n"
            f"Local {self.dt_local.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"UTC {self.dt_utc.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Sun alt {alt_vec:.2f}° az {az_vec:.2f}°"
        )
        
        # Galactic center
        gc_eq = galactic_center_unit_eq()
        gc_pos = (self.radius * gc_eq).astype(np.float32)
        self.gc_dot.setData(pos=gc_pos.reshape(1, 3))
        
        # Update ground view elements if active
        if self.current_view_mode == "ground":
            self._update_compass_positions(p_view, Rearth)
            self.ground_plane.resetTransform()
            self.ground_plane.translate(float(p_view[0]), float(p_view[1]), float(p_view[2]))
        
        # Debug output
        if DEBUG_ENABLED:
            from debug.debug_output import print_celestial_debug, print_gc_visibility
            print_celestial_debug(
                self.dt_local, self.dt_utc, self.lat, self.lon, jd, gmst, lst,
                sun_ra, sun_dec, sun_eq, sun_local, alt_vec, az_vec, 
                alt_formula, az_formula, M, p_ecef, p_view, gc_eq, 
                self.mw_pts_eq, self.radius, EARTH_ROT_SIGN, APP_TZ
            )
            print_gc_visibility(self.lat, self.lon, self.dt_local, APP_TZ)