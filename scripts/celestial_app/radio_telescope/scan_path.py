"""
Radio Telescope Scan Path Calculator

Calculates the celestial path traced by a fixed radio telescope
as Earth rotates during the scan duration.
"""
import numpy as np
from datetime import datetime, timedelta
from astronomy.coordinates import (
    equatorial_to_local_enu_matrix,
    unit_vector_enu_to_alt_az,
    ra_dec_to_unit_vector_equatorial
)
from astronomy.time_utils import gmst_degrees
from geometry.transformations import normalize_vector


class ScanPath:
    """Represents a radio telescope scan path."""
    
    def __init__(self, altitude: float, azimuth: float, duration_seconds: float,
                 resolution: float, start_time: datetime, latitude: float, 
                 longitude: float):
        """
        Initialize scan path.
        
        Args:
            altitude: Telescope altitude/elevation in degrees (0-90)
            azimuth: Telescope azimuth in degrees (0-360)
            duration_seconds: Scan duration in seconds
            resolution: Telescope beam width in degrees
            start_time: Scan start time
            latitude: Observer latitude in degrees
            longitude: Observer longitude in degrees
        """
        self.altitude = altitude
        self.azimuth = azimuth
        self.duration_seconds = duration_seconds
        self.resolution = resolution
        self.start_time = start_time
        self.latitude = latitude
        self.longitude = longitude
        
        # Calculate the scan path
        self._calculate_path()
    
    def _calculate_path(self):
        """Calculate the celestial coordinates of the scan path."""
        # Sample the scan at regular intervals (every 60 seconds)
        sample_interval = 60  # seconds
        num_samples = max(2, int(self.duration_seconds / sample_interval) + 1)
        
        # Generate time samples
        time_samples = []
        for i in range(num_samples):
            t = self.start_time + timedelta(seconds=i * sample_interval)
            time_samples.append(t)
        
        # For each time sample, calculate the equatorial coordinates
        # of the telescope's pointing direction
        self.equatorial_points = []  # List of (RA, Dec) in degrees
        self.equatorial_vectors = []  # List of unit vectors
        
        for t in time_samples:
            # Get LST at this time
            t_naive = t.replace(tzinfo=None) if t.tzinfo else t
            gmst = gmst_degrees(t_naive)
            lst = (gmst + self.longitude) % 360.0
            
            # Get transformation matrix from equatorial to local at this time
            M = equatorial_to_local_enu_matrix(self.latitude, lst)
            M_inv = np.linalg.inv(M)
            
            # Convert telescope's fixed alt/az to local ENU vector
            alt_rad = np.deg2rad(self.altitude)
            az_rad = np.deg2rad(self.azimuth)
            
            # ENU vector from alt/az
            E = np.cos(alt_rad) * np.sin(az_rad)
            N = np.cos(alt_rad) * np.cos(az_rad)
            U = np.sin(alt_rad)
            local_vec = np.array([E, N, U], dtype=np.float32)
            
            # Transform to equatorial coordinates
            eq_vec = (M_inv @ local_vec).astype(np.float32)
            eq_vec = normalize_vector(eq_vec)
            
            # Convert to RA/Dec
            ra_rad = np.arctan2(eq_vec[1], eq_vec[0])
            ra_deg = np.rad2deg(ra_rad) % 360.0
            
            dec_rad = np.arcsin(np.clip(eq_vec[2], -1.0, 1.0))
            dec_deg = np.rad2deg(dec_rad)
            
            self.equatorial_points.append((ra_deg, dec_deg))
            self.equatorial_vectors.append(eq_vec)
    
    def get_path_vertices_3d(self, radius: float = 1.0):
        """
        Get 3D vertices for drawing the scan path on a sphere.
        
        Args:
            radius: Sphere radius
        
        Returns:
            numpy array of shape (N, 3) with 3D coordinates
        """
        vertices = []
        for eq_vec in self.equatorial_vectors:
            vertex = eq_vec * radius
            vertices.append(vertex)
        
        return np.array(vertices, dtype=np.float32)
    
    def get_band_mesh_3d(self, radius: float = 1.0, segments: int = 20):
        """
        Get 3D mesh vertices for the scan band (with width = resolution).
        
        Creates a tube/band around the scan path with width determined
        by the telescope's beam width.
        
        Args:
            radius: Sphere radius
            segments: Number of segments around the band circumference
        
        Returns:
            (vertices, faces) tuple for mesh rendering
        """
        if len(self.equatorial_vectors) < 2:
            return np.array([]), np.array([])
        
        # Half beam width
        half_width = self.resolution / 2.0
        
        vertices_list = []
        
        # For each point on the path, create a circle perpendicular to the path
        for i, eq_vec in enumerate(self.equatorial_vectors):
            # Find a perpendicular vector
            # Use a simple approach: cross product with a reference vector
            if abs(eq_vec[2]) < 0.9:
                ref = np.array([0, 0, 1], dtype=np.float32)
            else:
                ref = np.array([1, 0, 0], dtype=np.float32)
            
            perp1 = np.cross(eq_vec, ref)
            perp1 = normalize_vector(perp1)
            
            perp2 = np.cross(eq_vec, perp1)
            perp2 = normalize_vector(perp2)
            
            # Create circle of points around this path point
            for j in range(segments):
                angle = 2 * np.pi * j / segments
                offset = (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                offset = offset * np.deg2rad(half_width)
                
                # Rotate the equatorial vector slightly
                point = eq_vec + offset
                point = normalize_vector(point) * radius
                
                vertices_list.append(point)
        
        vertices = np.array(vertices_list, dtype=np.float32)
        
        # Create faces (triangles) connecting the rings
        faces_list = []
        num_rings = len(self.equatorial_vectors)
        
        for i in range(num_rings - 1):
            for j in range(segments):
                j_next = (j + 1) % segments
                
                # Current ring indices
                v0 = i * segments + j
                v1 = i * segments + j_next
                
                # Next ring indices
                v2 = (i + 1) * segments + j
                v3 = (i + 1) * segments + j_next
                
                # Two triangles to form a quad
                faces_list.append([v0, v1, v2])
                faces_list.append([v1, v3, v2])
        
        faces = np.array(faces_list, dtype=np.uint32)
        
        return vertices, faces
    
    def is_visible_at_time(self, current_time: datetime, latitude: float, 
                          longitude: float) -> tuple:
        """
        Check if any part of the scan is visible at a given time and location.
        
        Args:
            current_time: Current time to check
            latitude: Observer latitude
            longitude: Observer longitude
        
        Returns:
            (is_visible, visible_points) - boolean and list of (alt, az) pairs
                for points above the horizon
        """
        current_naive = current_time.replace(tzinfo=None) if current_time.tzinfo else current_time
        gmst = gmst_degrees(current_naive)
        lst = (gmst + longitude) % 360.0
        
        M = equatorial_to_local_enu_matrix(latitude, lst)
        
        visible_points = []
        
        for eq_vec in self.equatorial_vectors:
            # Transform to local coordinates
            local_vec = (M @ eq_vec).astype(np.float32)
            local_vec = normalize_vector(local_vec)
            
            # Get alt/az
            alt, az = unit_vector_enu_to_alt_az(local_vec)
            
            if alt > 0:  # Above horizon
                visible_points.append((alt, az))
        
        return len(visible_points) > 0, visible_points
    
    def get_visible_band_for_chart(self, current_time: datetime, latitude: float,
                                   longitude: float):
        """
        Get scan band segments that are visible above the horizon.
        
        Args:
            current_time: Current observation time
            latitude: Observer latitude
            longitude: Observer longitude
        
        Returns:
            List of (zenith_dist, azimuth) pairs for visible segments
        """
        is_visible, visible_points = self.is_visible_at_time(
            current_time, latitude, longitude
        )
        
        if not is_visible:
            return []
        
        # Convert alt/az to zenith distance and azimuth for polar plot
        chart_points = []
        for alt, az in visible_points:
            zenith_dist = 90 - alt
            chart_points.append((zenith_dist, az))
        
        return chart_points
