"""
Ground View Mode - Observer on Earth looking at sky
"""
import numpy as np
from PyQt6 import QtGui
from geometry.transformations import rotz_deg
from astronomy.coordinates import latlon_to_ecef, ecef_basis_at


class GroundView:
    """Handles ground view mode - observer standing on Earth looking at sky"""
    
    def __init__(self, view_widget, scene_items, radius=1.0, earth_radius=0.36):
        """
        Initialize ground view
        
        Args:
            view_widget: The FixedGLViewWidget instance
            scene_items: Dict containing all scene items (sky, earth, markers, etc.)
            radius: Celestial sphere radius
            earth_radius: Earth sphere radius
        """
        self.view = view_widget
        self.items = scene_items
        self.radius = radius
        self.earth_radius = earth_radius
        
        # Ground view settings
        self.fov = 90.0  # Field of view in degrees
        self.camera_distance = 0.001  # Very close to observer position
    
    def activate(self, lat, lon, gmst, earth_rot_sign=1.0):
        """
        Activate ground view mode
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            gmst: Greenwich Mean Sidereal Time in degrees
            earth_rot_sign: Sign of Earth rotation (+1 or -1)
        """
        self._set_visibility(True)
        Rearth, p_view = self._calculate_observer_position(lat, lon, gmst, earth_rot_sign)
        self._update_ground_plane(lat, lon, p_view, Rearth)
        self._update_compass_markers(lat, lon, Rearth)
        self._set_camera(lat, lon, p_view, Rearth)
    
    def _set_visibility(self, visible: bool):
        """Set visibility of objects in ground view"""
        # Show ground view objects
        self.items['ground_plane'].setVisible(visible)
        for marker in self.items['compass_markers'].values():
            marker.setVisible(visible)
        self.items['sky'].setVisible(visible)
        self.items['sun_disk'].setVisible(visible)
        
        # Hide globe view objects
        self.items['earth'].setVisible(False)
        self.items['loc_marker'].setVisible(False)
        self.items['earth_axis'].setVisible(False)
        self.items['sun_dot'].setVisible(False)
    
    def _calculate_observer_position(self, lat, lon, gmst, earth_rot_sign):
        """Calculate observer position in view coordinates"""
        p_ecef = latlon_to_ecef(lat, lon, self.earth_radius)
        Rearth = rotz_deg(earth_rot_sign * gmst)
        p_view = (Rearth @ p_ecef).astype(np.float32)
        return Rearth, p_view
    
    def _update_ground_plane(self, lat, lon, p_view, Rearth):
        """Update ground plane position and orientation"""
        east, north, up = ecef_basis_at(lat, lon)
        up_view = (Rearth @ up).astype(np.float32)
        
        # Position ground plane at observer location
        self.items['ground_plane'].resetTransform()
        self.items['ground_plane'].translate(
            float(p_view[0]), 
            float(p_view[1]), 
            float(p_view[2])
        )
        
        # Rotate to align with local horizontal
        z_axis = np.array([0, 0, 1], dtype=np.float32)
        rotation_axis = np.cross(z_axis, up_view)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, up_view), -1.0, 1.0))
            angle_deg = np.rad2deg(angle)
            self.items['ground_plane'].rotate(
                angle_deg, 
                rotation_axis[0], 
                rotation_axis[1], 
                rotation_axis[2]
            )
    
    def _update_compass_markers(self, lat, lon, Rearth):
        """Update compass marker positions on horizon"""
        east, north, up = ecef_basis_at(lat, lon)
        east_view = (Rearth @ east).astype(np.float32)
        north_view = (Rearth @ north).astype(np.float32)
        
        # Place markers on the celestial sphere at horizon level
        horizon_dist = self.radius * 0.99
        
        positions = {
            'N': horizon_dist * north_view,
            'S': -horizon_dist * north_view,
            'E': horizon_dist * east_view,
            'W': -horizon_dist * east_view,
        }
        
        for direction, pos in positions.items():
            marker = self.items['compass_markers'][direction]
            marker.resetTransform()
            marker.translate(float(pos[0]), float(pos[1]), float(pos[2]))
    
    def _set_camera(self, lat, lon, p_view, Rearth):
        """Set camera position and orientation for ground view"""
        east, north, up = ecef_basis_at(lat, lon)
        north_view = (Rearth @ north).astype(np.float32)
        
        # Camera position: at observer location on Earth
        self.view.opts['center'] = QtGui.QVector3D(
            float(p_view[0]), 
            float(p_view[1]), 
            float(p_view[2])
        )
        self.view.opts['distance'] = self.camera_distance
        self.view.opts['fov'] = self.fov
        
        # Calculate camera orientation to look south
        south_dir = -north_view
        
        # Project south direction onto XY plane for azimuth
        azimuth = np.rad2deg(np.arctan2(south_dir[1], south_dir[0]))
        
        # Elevation should be 0 to look at horizon
        elevation = 0.0
        
        self.view.opts['azimuth'] = float(azimuth)
        self.view.opts['elevation'] = float(elevation)
        self.view.update()
    
    def set_fov(self, fov_degrees):
        """Set field of view for ground view"""
        self.fov = float(fov_degrees)
        # Update camera if currently active
        if self.items['ground_plane'].visible():
            self.view.opts['fov'] = self.fov
            self.view.update()
    
    def update_for_time_change(self, lat, lon, gmst, earth_rot_sign=1.0):
        """Update ground view when time changes (recalculate positions)"""
        Rearth, p_view = self._calculate_observer_position(lat, lon, gmst, earth_rot_sign)
        self._update_ground_plane(lat, lon, p_view, Rearth)
        self._update_compass_markers(lat, lon, Rearth)
        self._set_camera(lat, lon, p_view, Rearth)
