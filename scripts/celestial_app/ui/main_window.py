import sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import numpy as np
from PyQt6 import QtWidgets, QtGui

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

APP_TZ = ZoneInfo("America/New_York")
DEBUG_VERBOSE = True
EARTH_ROT_SIGN = 1.0
SUN_ANG_RADIUS_DEG = 0.265

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
    
    def _set_object_visibility(self, sphere_mode=True):
        """Set visibility of objects based on view mode"""
        if sphere_mode:
            # Sphere view: show everything
            self.sky_item.setVisible(True)
            self.earth_item.setVisible(True)
            self.loc_marker.setVisible(True)
            self.earth_axis_item.setVisible(True)
        else:
            # Ground view: hide Earth and location marker, show only sky
            self.sky_item.setVisible(True)
            self.earth_item.setVisible(False)
            self.loc_marker.setVisible(False)
            self.earth_axis_item.setVisible(False)
    
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
    
    def on_sphere_view(self):
        """Set camera to sphere view (external view of celestial sphere)"""
        # Show all objects
        self._set_object_visibility(sphere_mode=True)
        
        # Reset camera to default external view
        self.view.opts['center'] = QtGui.QVector3D(0, 0, 0)
        self.view.opts['elevation'] = 30  # Look down at 30 degrees
        self.view.opts['azimuth'] = 45    # Rotate 45 degrees
        self.view.opts['distance'] = 2.6
        self.view.update()
    
    def on_ground_view(self):
        """Set camera to ground view (observer on Earth looking south at horizon)"""
        # Hide Earth and location marker for ground view
        self._set_object_visibility(sphere_mode=False)
        
        # Calculate observer position on Earth surface
        p_ecef = latlon_to_ecef(self.lat, self.lon, self.earth_radius)
        dt_utc_naive = self.dt_utc.replace(tzinfo=None)
        gmst = gmst_degrees(dt_utc_naive)
        Rearth = rotz_deg(EARTH_ROT_SIGN * gmst)
        p_view = (Rearth @ p_ecef).astype(np.float32)
        
        # Get local coordinate basis (east, north, up)
        east, north, up = ecef_basis_at(self.lat, self.lon)
        
        # Transform basis vectors to current view frame
        east_view = (Rearth @ east).astype(np.float32)
        north_view = (Rearth @ north).astype(np.float32)
        up_view = (Rearth @ up).astype(np.float32)
        
        # Camera position: at the surface (origin, since we're at the "center" of our local view)
        cam_pos = p_view
        
        # Look direction: south along horizon (negative north)
        # Normalize to ensure we're looking exactly at the horizon
        look_dir = -north_view
        look_dir = look_dir / (np.linalg.norm(look_dir) + 1e-12)
        
        # Target point: far in the distance toward south
        target = cam_pos + 10.0 * look_dir
        
        # Set camera - position it at observer location
        self.view.opts['center'] = QtGui.QVector3D(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        self.view.opts['distance'] = 0.01  # Very close to "being" at that point
        
        # Calculate azimuth and elevation for camera orientation
        # We want to look horizontally (elevation ≈ 0) toward south
        azimuth = np.rad2deg(np.arctan2(look_dir[1], look_dir[0]))
        
        self.view.opts['azimuth'] = float(azimuth)
        self.view.opts['elevation'] = 0.0  # Look straight at horizon
        self.view.update()
    
    def update_scene(self):
        """Update scene for current time and location"""
        dt_utc_naive = self.dt_utc.replace(tzinfo=None)
        jd = julian_day(dt_utc_naive)
        gmst = gmst_degrees(dt_utc_naive)
        lst = (gmst + self.lon) % 360.0
        
        # EARTH FRAME: Rotates with Earth (by GMST)
        Rearth = rotz_deg(EARTH_ROT_SIGN * gmst)
        
        # Rotate Earth mesh to align texture with GMST
        self.earth_item.resetTransform()
        self.earth_item.rotate(EARTH_ROT_SIGN * gmst, 0, 0, 1)
        
        # Location marker on Earth surface
        p_ecef = latlon_to_ecef(self.lat, self.lon, self.earth_radius)
        p_view = (Rearth @ p_ecef).astype(np.float32)
        self.loc_marker.resetTransform()
        self.loc_marker.translate(float(p_view[0]), float(p_view[1]), float(p_view[2]))
        
        # Horizon ring (Earth-fixed, rotates with Earth)
        horizon_ecef = make_horizon_ring_ecef(self.radius, self.lat, self.lon, n=600)
        horizon_view = (Rearth @ horizon_ecef.T).T.astype(np.float32)
        self.horizon_item.setData(pos=horizon_view)
        
        # Earth's rotation axis (fixed along Z-axis in inertial frame)
        axis_pts = np.array([[0.0, 0.0, -self.radius],
                             [0.0, 0.0,  self.radius]], dtype=np.float32)
        self.earth_axis_item.setData(pos=axis_pts)
        
        # CELESTIAL FRAME: Fixed in inertial space
        # Celestial equator
        eq = make_ring(self.radius, 600, "xy")
        self.eq_item.setData(pos=eq)
        
        # Milky Way (already in equatorial XYZ)
        self.mw_item.setData(pos=self.mw_pts_eq, color=self.mw_cols)
        
        # Sun in equatorial coordinates
        sun_ra, sun_dec = sun_ra_dec_degrees(dt_utc_naive)
        from astronomy.coordinates import ra_dec_to_unit_vector_equatorial
        sun_eq = ra_dec_to_unit_vector_equatorial(sun_ra, sun_dec)
        sun_pos = (self.radius * sun_eq).astype(np.float32)
        
        self.sun_dot.setData(pos=sun_pos.reshape(1, 3))
        disk_radius = self.radius * np.sin(np.deg2rad(SUN_ANG_RADIUS_DEG))
        self.sun_item.setMeshData(meshdata=make_disk_mesh(
            center=sun_pos,
            normal=sun_eq.astype(np.float32),
            radius=disk_radius,
            segments=56
        ))
        
        # For alt/az calculation
        M = equatorial_to_local_enu_matrix(self.lat, lst)
        sun_local = (M @ sun_eq.reshape(3, 1)).ravel().astype(np.float32)
        sun_local = sun_local / (np.linalg.norm(sun_local) + 1e-12)
        alt_vec, az_vec = unit_vector_enu_to_alt_az(sun_local)
        
        self.info.setText(
            f"Lat {self.lat:.4f}  Lon {self.lon:.4f}\n"
            f"Local {self.dt_local.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"UTC {self.dt_utc.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Sun alt {alt_vec:.2f} deg  az {az_vec:.2f} deg"
        )
        
        # Galactic center
        gc_eq = galactic_center_unit_eq()
        gc_pos = (self.radius * gc_eq).astype(np.float32)
        self.gc_dot.setData(pos=gc_pos.reshape(1, 3))
        
        if DEBUG_VERBOSE:
            self._print_debug(jd, gmst, lst, sun_ra, sun_dec, sun_eq, 
                            sun_local, alt_vec, az_vec, M, p_ecef, p_view, gc_eq)
    
    def _print_debug(self, jd, gmst, lst, sun_ra, sun_dec, sun_eq, 
                    sun_local, alt_vec, az_vec, M, p_ecef, p_view, gc_eq):
        """Print debug information"""
        alt_formula, az_formula = alt_az_from_ra_dec(self.lat, lst, sun_ra, sun_dec)
        ha = (lst - sun_ra) % 360.0
        
        E2G = eq_to_gal_matrix_j2000()
        g = (E2G @ sun_eq.reshape(3, 1)).ravel()
        g = g / (np.linalg.norm(g) + 1e-12)
        b_gal = np.rad2deg(np.arcsin(np.clip(g[2], -1.0, 1.0)))
        l_gal = np.rad2deg(np.arctan2(g[1], g[0])) % 360.0
        
        # GC sanity checks
        gc_gal = (E2G @ gc_eq.reshape(3, 1)).ravel()
        gc_gal = gc_gal / (np.linalg.norm(gc_gal) + 1e-12)
        gc_l = np.rad2deg(np.arctan2(gc_gal[1], gc_gal[0])) % 360.0
        gc_b = np.rad2deg(np.arcsin(np.clip(gc_gal[2], -1.0, 1.0)))
        print(f"  GC in galactic coords: l={gc_l:.4f} deg, b={gc_b:.4f} deg (should be ~0, ~0)")
        
        scp = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        ncp = np.array([0.0, 0.0,  1.0], dtype=np.float32)
        
        def angular_sep_deg(u, v):
            u = u / (np.linalg.norm(u) + 1e-12)
            v = v / (np.linalg.norm(v) + 1e-12)
            return float(np.rad2deg(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))
        
        gc_sep_scp = angular_sep_deg(gc_eq, scp)
        gc_sep_ncp = angular_sep_deg(gc_eq, ncp)
        gc_local = (M @ gc_eq.reshape(3, 1)).ravel().astype(np.float32)
        gc_local = gc_local / (np.linalg.norm(gc_local) + 1e-12)
        gc_alt, gc_az = unit_vector_enu_to_alt_az(gc_local)
        
        mw_dirs_eq = self.mw_pts_eq / (self.radius + 1e-12)
        dots = np.clip(mw_dirs_eq @ (gc_eq / (np.linalg.norm(gc_eq) + 1e-12)), -1.0, 1.0)
        min_sep = float(np.rad2deg(np.arccos(float(np.max(dots)))))
        
        print("\nDEBUG")
        print(f"Local time {self.dt_local.strftime('%Y-%m-%d %H:%M:%S')} {APP_TZ.key}")
        print(f"UTC time   {self.dt_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Lat deg   {self.lat: .6f}")
        print(f"Lon deg   {self.lon: .6f}")
        print(f"JD        {jd: .12f}")
        print(f"GMST deg  {gmst: .6f}")
        print(f"LST deg   {lst: .6f}")
        print(f"Sun RA deg  {sun_ra: .6f}")
        print(f"Sun Dec deg {sun_dec: .6f}")
        print(f"Hour angle deg {ha: .6f}")
        print(f"Alt deg from vector  {alt_vec: .6f}")
        print(f"Az deg from vector   {az_vec: .6f}")
        print(f"Alt deg from formula {alt_formula: .6f}")
        print(f"Az deg from formula  {az_formula: .6f}")
        print(f"Sun galactic l deg   {l_gal: .6f}")
        print(f"Sun galactic b deg   {b_gal: .6f}")
        print("\nGalactic Center sanity checks")
        print(f"  GC local alt {gc_alt: .6f} deg  az {gc_az: .6f} deg")
        print(f"  GC angle to south celestial pole {gc_sep_scp: .6f} deg (expected about 61.064)")
        print(f"  GC angle to north celestial pole {gc_sep_ncp: .6f} deg (expected about 118.936)")
        print(f"  Min angular sep between GC and Milky Way points {min_sep: .6f} deg (should be small)")
        print("END DEBUG\n")