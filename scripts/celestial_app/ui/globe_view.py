"""
Globe View Mode - External view of celestial sphere with Earth
"""
import numpy as np
from PyQt6 import QtGui
from geometry.transformations import rotz_deg
from astronomy.coordinates import latlon_to_ecef, make_horizon_ring_ecef
from geometry.mesh_generation import make_ring


class GlobeView:
    """Handles globe/sphere view mode - external view of celestial sphere"""
    
    def __init__(self, view_widget, scene_items, radius=1.0, earth_radius=0.36):
        """
        Initialize globe view
        
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
        
        # Default camera settings for globe view
        self.default_elevation = 30
        self.default_azimuth = 45
        self.default_distance = 2.6
        self.default_fov = 60
    
    def activate(self):
        """Activate globe view mode"""
        self._set_visibility(True)
        self._set_camera()
    
    def _set_visibility(self, visible: bool):
        """Set visibility of objects in globe view"""
        # Show all globe view objects
        self.items['sky'].setVisible(True)
        self.items['earth'].setVisible(True)
        self.items['loc_marker'].setVisible(True)
        self.items['earth_axis'].setVisible(True)
        self.items['sun_dot'].setVisible(True)
        self.items['sun_disk'].setVisible(True)
        self.items['mw'].setVisible(True)
        self.items['eq'].setVisible(True)
        self.items['gc_dot'].setVisible(True)
        self.items['horizon'].setVisible(True)
        
        # Hide compass markers (not needed in globe view)
        for marker in self.items['compass_markers'].values():
            marker.setVisible(False)
    
    def _set_camera(self):
        """Set camera position and orientation for globe view"""
        self.view.opts['center'] = QtGui.QVector3D(0, 0, 0)
        self.view.opts['elevation'] = self.default_elevation
        self.view.opts['azimuth'] = self.default_azimuth
        self.view.opts['distance'] = self.default_distance
        self.view.opts['fov'] = self.default_fov
        self.view.update()
    
    def update_scene(self, lat, lon, gmst, earth_rot_sign=1.0):
        """
        Update scene elements for current time and location
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            gmst: Greenwich Mean Sidereal Time in degrees
            earth_rot_sign: Sign of Earth rotation (+1 or -1)
        """
        # Rotate Earth
        Rearth = rotz_deg(earth_rot_sign * gmst)
        
        self.items['earth'].resetTransform()
        self.items['earth'].rotate(earth_rot_sign * gmst, 0, 0, 1)
        
        # Update location marker position
        p_ecef = latlon_to_ecef(lat, lon, self.earth_radius)
        p_view = (Rearth @ p_ecef).astype(np.float32)
        self.items['loc_marker'].resetTransform()
        self.items['loc_marker'].translate(float(p_view[0]), float(p_view[1]), float(p_view[2]))
        
        # Update horizon ring
        horizon_ecef = make_horizon_ring_ecef(self.radius, lat, lon, n=600)
        horizon_view = (Rearth @ horizon_ecef.T).T.astype(np.float32)
        self.items['horizon'].setData(pos=horizon_view)
        
        # Update Earth axis
        axis_pts = np.array([
            [0.0, 0.0, -self.radius],
            [0.0, 0.0,  self.radius]
        ], dtype=np.float32)
        self.items['earth_axis'].setData(pos=axis_pts)
        
        # Update celestial equator
        eq = make_ring(self.radius, 600, "xy")
        self.items['eq'].setData(pos=eq)
        
        return Rearth, p_view