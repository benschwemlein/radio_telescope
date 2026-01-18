
"""
Ground View Mode - Simple ground and sky only
"""
import numpy as np
from PyQt6 import QtGui
from geometry.transformations import rotz_deg
from astronomy.coordinates import latlon_to_ecef, ecef_basis_at


class GroundView:
    """Simple ground view - just green ground and blue sky"""
    
    def __init__(self, view_widget, scene_items, radius=1.0, earth_radius=0.36):
        self.view = view_widget
        self.items = scene_items
        self.radius = radius
        self.earth_radius = earth_radius
        self.fov = 90.0
    
    def activate(self, lat, lon, gmst, earth_rot_sign=1.0):
        """Switch to ground view - show ONLY ground and sky"""
        
        # SHOW ONLY: green ground and blue sky
        self.items['ground_plane'].setVisible(True)
        self.items['sky'].setVisible(True)
        
        # HIDE EVERYTHING ELSE
        self.items['sun_disk'].setVisible(False)
        self.items['sun_dot'].setVisible(False)
        self.items['mw'].setVisible(False)
        self.items['horizon'].setVisible(False)
        self.items['earth'].setVisible(False)
        self.items['loc_marker'].setVisible(False)
        self.items['earth_axis'].setVisible(False)
        self.items['eq'].setVisible(False)
        self.items['gc_dot'].setVisible(False)
        
        for marker in self.items['compass_markers'].values():
            marker.setVisible(False)
        
        # Calculate observer position
        p_ecef = latlon_to_ecef(lat, lon, self.earth_radius)
        Rearth = rotz_deg(earth_rot_sign * gmst)
        p_view = (Rearth @ p_ecef).astype(np.float32)
        
        # Get local orientation
        east, north, up = ecef_basis_at(lat, lon)
        north_view = (Rearth @ north).astype(np.float32)
        up_view = (Rearth @ up).astype(np.float32)
        
        # Position ground plane at observer location
        self.items['ground_plane'].resetTransform()
        self.items['ground_plane'].translate(
            float(p_view[0]), 
            float(p_view[1]), 
            float(p_view[2])
        )
        
        # Rotate ground to be horizontal
        z_axis = np.array([0, 0, 1], dtype=np.float32)
        rotation_axis = np.cross(z_axis, up_view)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, up_view), -1.0, 1.0))
            angle_deg = np.rad2deg(angle)
            self.items['ground_plane'].rotate(
                angle_deg, 
                float(rotation_axis[0]), 
                float(rotation_axis[1]), 
                float(rotation_axis[2])
            )
        
        # Set camera at observer position
        self.view.opts['center'] = QtGui.QVector3D(
            float(p_view[0]), 
            float(p_view[1]), 
            float(p_view[2])
        )
        self.view.opts['distance'] = 0.001
        self.view.opts['fov'] = self.fov
        
        # Look south at horizon
        south_dir = -north_view
        azimuth = np.rad2deg(np.arctan2(south_dir[1], south_dir[0]))
        elevation = 0.0  # Look straight at horizon
        
        self.view.opts['azimuth'] = float(azimuth)
        self.view.opts['elevation'] = float(elevation)
        self.view.update()
    
    def set_fov(self, fov_degrees):
        """Adjust zoom"""
        self.fov = float(fov_degrees)
        self.view.opts['fov'] = self.fov
        self.view.update()
    
    def update_for_time_change(self, lat, lon, gmst, earth_rot_sign=1.0):
        """Update when time changes"""
        self.activate(lat, lon, gmst, earth_rot_sign)

