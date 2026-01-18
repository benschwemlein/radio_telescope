"""
Milky Way Renderer - Utility class for accurate Milky Way band structure
Uses real all-sky Milky Way imagery and transforms it to observer's view
Can be used by both 3D globe view and 2D star chart
"""
import numpy as np
from typing import Tuple
from PIL import Image

class MilkyWayRenderer:
    """
    Renders the Milky Way using real all-sky imagery in galactic coordinates.
    Transforms the image to equatorial coordinates for accurate sky display.
    """
    
    def __init__(self, radius: float = 1.0, image_path: str = None):
        """
        Initialize Milky Way renderer
        
        Args:
            radius: Radius of the celestial sphere
            image_path: Path to all-sky Milky Way image in galactic coordinates
        """
        self.radius = radius
        self.milky_way_image = None
        
        if image_path:
            self.load_milky_way_image(image_path)
    
    def load_milky_way_image(self, image_path: str):
        """Load the Milky Way background image"""
        try:
            self.milky_way_image = Image.open(image_path).convert('L')  # Grayscale
            print(f"Loaded Milky Way image: {self.milky_way_image.size}")
        except Exception as e:
            print(f"Could not load Milky Way image: {e}")
            self.milky_way_image = None
    
    def sample_milky_way_at_galactic_coords(self, l_deg: float, b_deg: float) -> float:
        """
        Sample brightness from the Milky Way image at galactic coordinates.
        
        The image is a Mollweide projection in galactic coordinates:
        - Horizontal center = galactic center (l=0°)
        - Left edge = l=180° (anticenter side, wrapping from l=180° to l=360°=0°)
        - Right edge = l=180° (anticenter, other side)
        - Vertical center = galactic plane (b=0°)
        - Top = north galactic pole (b=+90°)
        - Bottom = south galactic pole (b=-90°)
        
        Args:
            l_deg: Galactic longitude in degrees (0-360)
            b_deg: Galactic latitude in degrees (-90 to +90)
            
        Returns:
            Brightness value (0.0 to 1.0)
        """
        if self.milky_way_image is None:
            return 0.0
        
        width, height = self.milky_way_image.size
        
        # Normalize galactic longitude: 0-360 degrees
        # The image center (x = width/2) corresponds to l=0° (galactic center)
        # We need to shift so that l=0 is at center
        # Map: l=0 -> 0.5, l=180 -> 0.0 or 1.0, l=360 -> 0.5
        
        # Shift longitude so center of image is l=0
        l_shifted = (l_deg + 180) % 360  # Now l=180 is at position 0, l=0 is at position 180
        l_norm = l_shifted / 360.0  # 0 to 1
        
        # Latitude: -90 to +90 maps to bottom to top of image
        # For Mollweide projection, need to handle the curved edges
        # Simple approximation: linear mapping
        b_norm = (b_deg + 90) / 180.0  # 0 (bottom, -90°) to 1 (top, +90°)
        
        # Convert to pixel coordinates
        x = int(l_norm * width) % width
        y = int((1.0 - b_norm) * height)  # Flip y axis (image y=0 is top)
        y = max(0, min(height - 1, y))
        
        # Sample pixel value
        pixel_value = self.milky_way_image.getpixel((x, y))
        
        # Normalize to 0-1
        return pixel_value / 255.0
    
    def generate_milky_way_texture_points(self, 
                                          density: int = 5000,
                                          min_brightness: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate points from the Milky Way image in equatorial coordinates.
        Samples the image and converts bright pixels to 3D points.
        
        Args:
            density: Number of points to sample
            min_brightness: Minimum brightness threshold (0-1)
            
        Returns:
            points: (N, 3) array of points in equatorial coordinates
            brightness: (N,) array of brightness values
        """
        if self.milky_way_image is None:
            # Fallback to empty arrays
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
        
        points_list = []
        brightness_list = []
        
        # Sample galactic coordinates
        rng = np.random.default_rng(42)
        
        for _ in range(density):
            # Random galactic longitude and latitude
            l_deg = rng.uniform(0, 360)
            b_deg = rng.uniform(-30, 30)  # Focus on band (±30° from plane)
            
            # Sample brightness from image
            brightness = self.sample_milky_way_at_galactic_coords(l_deg, b_deg)
            
            # Only include if bright enough
            if brightness < min_brightness:
                continue
            
            # Convert galactic (l, b) to Cartesian in galactic frame
            l_rad = np.deg2rad(l_deg)
            b_rad = np.deg2rad(b_deg)
            
            x_gal = np.cos(b_rad) * np.cos(l_rad)
            y_gal = np.cos(b_rad) * np.sin(l_rad)
            z_gal = np.sin(b_rad)
            
            gal_vec = np.array([x_gal, y_gal, z_gal], dtype=np.float32)
            
            # Convert to equatorial coordinates
            eq_vec = self._galactic_to_equatorial(gal_vec)
            eq_vec = self.radius * eq_vec / (np.linalg.norm(eq_vec) + 1e-12)
            
            points_list.append(eq_vec)
            brightness_list.append(brightness)
        
        if points_list:
            return (np.array(points_list, dtype=np.float32),
                    np.array(brightness_list, dtype=np.float32))
        else:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
    
    def generate_milky_way_polygon_mesh(self, l_samples: int = 360, b_samples: int = 20) -> Tuple:
        """
        Generate a mesh of the Milky Way band for polygon rendering.
        Creates strips along galactic longitude with varying width.
        
        Args:
            l_samples: Number of samples along galactic longitude
            b_samples: Number of samples across galactic latitude (band width)
            
        Returns:
            Tuple of (vertices_equatorial, brightness_values, indices)
        """
        if self.milky_way_image is None:
            return None, None, None
        
        vertices = []
        brightness = []
        
        # Sample along galactic longitude
        l_values = np.linspace(0, 360, l_samples, endpoint=False)
        # Sample across galactic latitude (±30° covers the visible band)
        b_values = np.linspace(-30, 30, b_samples)
        
        for l_deg in l_values:
            for b_deg in b_values:
                # Sample brightness
                bright = self.sample_milky_way_at_galactic_coords(l_deg, b_deg)
                
                # Skip very dim areas
                if bright < 0.05:
                    bright = 0.0
                
                # Convert to Cartesian galactic
                l_rad = np.deg2rad(l_deg)
                b_rad = np.deg2rad(b_deg)
                
                x_gal = np.cos(b_rad) * np.cos(l_rad)
                y_gal = np.cos(b_rad) * np.sin(l_rad)
                z_gal = np.sin(b_rad)
                
                gal_vec = np.array([x_gal, y_gal, z_gal], dtype=np.float32)
                
                # Convert to equatorial
                eq_vec = self._galactic_to_equatorial(gal_vec)
                eq_vec = self.radius * eq_vec
                
                vertices.append(eq_vec)
                brightness.append(bright)
        
        vertices = np.array(vertices, dtype=np.float32)
        brightness = np.array(brightness, dtype=np.float32)
        
        # Generate triangle indices for mesh
        indices = []
        for i in range(l_samples - 1):
            for j in range(b_samples - 1):
                # Two triangles per quad
                idx1 = i * b_samples + j
                idx2 = i * b_samples + (j + 1)
                idx3 = (i + 1) * b_samples + j
                idx4 = (i + 1) * b_samples + (j + 1)
                
                indices.extend([idx1, idx2, idx3])
                indices.extend([idx2, idx4, idx3])
        
        indices = np.array(indices, dtype=np.int32)
        
        return vertices, brightness, indices
    
    def _galactic_to_equatorial(self, gal_vec: np.ndarray) -> np.ndarray:
        """
        Convert from galactic coordinates to equatorial (J2000) coordinates.
        Uses IAU standard transformation matrix.
        
        Galactic coordinate system (IAU 1958):
        - North galactic pole: RA = 192.859°, Dec = 27.128° (J2000)
        - Galactic center: RA = 266.405°, Dec = -28.936° (J2000)
        
        Args:
            gal_vec: Unit vector in galactic coordinates (x, y, z)
            
        Returns:
            Unit vector in equatorial J2000 coordinates
        """
        # CORRECT rotation matrix from galactic to equatorial (J2000)
        # This is the transpose of the equatorial-to-galactic matrix
        R = np.array([
            [-0.0548755604,  0.4941094279, -0.8676661490],
            [-0.8734370902, -0.4448296300, -0.1980763734],
            [-0.4838350155,  0.7469822445,  0.4559837762]
        ], dtype=np.float32)
        
        eq_vec = R @ gal_vec
        return eq_vec / (np.linalg.norm(eq_vec) + 1e-12)
    
    def _equatorial_to_galactic(self, eq_vec: np.ndarray) -> np.ndarray:
        """
        Convert from equatorial (J2000) to galactic coordinates.
        Uses IAU standard transformation matrix.
        
        Args:
            eq_vec: Unit vector in equatorial J2000 coordinates
            
        Returns:
            Unit vector in galactic coordinates
        """
        # CORRECT rotation matrix from equatorial to galactic
        R = np.array([
            [-0.0548755604, -0.8734370902, -0.4838350155],
            [ 0.4941094279, -0.4448296300,  0.7469822445],
            [-0.8676661490, -0.1980763734,  0.4559837762]
        ], dtype=np.float32)
        
        gal_vec = R @ eq_vec
        return gal_vec / (np.linalg.norm(gal_vec) + 1e-12)
    
    def _get_width_factor(self, galactic_longitude_deg: float) -> float:
        """
        Get the width factor for the Milky Way at a given galactic longitude.
        The band is wider near the galactic center and anticenter.
        
        Args:
            galactic_longitude_deg: Galactic longitude in degrees (0-360)
            
        Returns:
            Width multiplier (0.5 to 1.5)
        """
        # Normalize to 0-180 range (symmetric)
        l = galactic_longitude_deg % 180
        
        # Wider at l=0° (galactic center) and somewhat wider at l=180° (anticenter)
        if l < 30 or l > 150:
            # Near center or anticenter
            return 1.3
        elif 60 < l < 120:
            # Thinnest region
            return 0.7
        else:
            # Transition regions
            return 1.0
    
    def _get_brightness_profile(self, galactic_longitude_deg: float) -> float:
        """
        Get brightness profile along the Milky Way.
        Brightest near galactic center (Sagittarius, l~0°),
        dimmer toward anticenter (Auriga, l~180°)
        
        Args:
            galactic_longitude_deg: Galactic longitude in degrees
            
        Returns:
            Brightness value (0.2 to 1.0)
        """
        l = galactic_longitude_deg
        
        # Galactic center region (Sagittarius-Scorpius): l = 350-10° (wraps around 0°)
        # Very bright core
        if l < 30 or l > 330:
            return 1.0
        
        # Cygnus region (Summer Milky Way): l = 60-90°
        # Also quite bright (we're looking along the Perseus arm)
        elif 60 <= l <= 90:
            return 0.85
        
        # Anticenter region (Auriga-Gemini): l = 170-190°
        # Dimmest - looking toward outer edge of galaxy
        elif 170 <= l <= 190:
            return 0.3
        
        # Transition regions
        elif 30 <= l <= 60:
            # Fading from center toward Cygnus
            return 0.9
        elif 90 <= l <= 170:
            # Fading from Cygnus toward anticenter
            t = (l - 90) / 80.0
            return 0.85 - 0.55 * t
        else:  # 190 < l < 330
            # Rising from anticenter back toward center
            t = (l - 190) / 140.0
            return 0.3 + 0.7 * t
    
    def _get_dust_obscuration(self, galactic_longitude_deg: float) -> float:
        """
        Model dark dust lanes that obscure parts of the Milky Way.
        Most prominent near the galactic center.
        
        Args:
            galactic_longitude_deg: Galactic longitude in degrees
            
        Returns:
            Transmission factor (0.0 = fully obscured, 1.0 = clear)
        """
        l = galactic_longitude_deg
        
        # Great Rift (Cygnus region): major dark lane at l~60-90°
        if 70 <= l <= 85:
            # Strong obscuration
            return 0.4
        
        # Dark lanes near galactic center
        elif l < 15 or l > 345:
            return 0.6
        
        # Otherwise relatively clear
        else:
            return 1.0
    
    def _galactic_to_equatorial(self, gal_vec: np.ndarray) -> np.ndarray:
        """
        Convert from galactic coordinates to equatorial (J2000) coordinates.
        
        The galactic coordinate system is centered on the Sun with:
        - Origin: Sun
        - l=0°, b=0°: Direction toward galactic center (Sagittarius)
        - l=90°, b=0°: Direction of galactic rotation (Cygnus)
        - b=+90°: North galactic pole (Coma Berenices)
        
        Args:
            gal_vec: Unit vector in galactic coordinates (x, y, z)
            
        Returns:
            Unit vector in equatorial J2000 coordinates
        """
        # Rotation matrix from galactic to equatorial (J2000)
        # Based on IAU definitions
        # Galactic north pole: RA = 192.859°, Dec = 27.128° (J2000)
        # Galactic center: RA = 266.405°, Dec = -28.936° (J2000)
        
        # This is the inverse of the eq_to_gal matrix
        R = np.array([
            [-0.054875560416, -0.873437090234, -0.483835015548],
            [ 0.494109427875, -0.444829629960,  0.746982248696],
            [-0.867666149019, -0.198076373431,  0.455983776175]
        ], dtype=np.float32)
        
        eq_vec = R @ gal_vec
        return eq_vec / (np.linalg.norm(eq_vec) + 1e-12)
    
    def generate_solid_band_edges(self, density: int = 360) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate upper and lower edges of the Milky Way band for solid polygon rendering.
        
        Args:
            density: Number of samples around the galactic plane (more = smoother)
            
        Returns:
            upper_edge: (N, 3) array of points along upper edge in equatorial coords
            lower_edge: (N, 3) array of points along lower edge in equatorial coords
            widths: (N,) array of band width at each longitude (for variable width)
        """
        # Sample along galactic longitude
        l_samples = np.linspace(0, 360, density, endpoint=False)
        
        upper_edge = []
        lower_edge = []
        widths = []
        
        for l_deg in l_samples:
            l_rad = np.deg2rad(l_deg)
            
            # Variable width based on location
            width_factor = self._get_width_factor(l_deg)
            band_width = 20.0 * width_factor  # degrees
            widths.append(band_width)
            
            # Upper edge (positive galactic latitude)
            b_upper = band_width / 2.0
            b_upper_rad = np.deg2rad(b_upper)
            x_u = np.cos(b_upper_rad) * np.cos(l_rad)
            y_u = np.cos(b_upper_rad) * np.sin(l_rad)
            z_u = np.sin(b_upper_rad)
            gal_upper = np.array([x_u, y_u, z_u], dtype=np.float32)
            
            # Lower edge (negative galactic latitude)
            b_lower = -band_width / 2.0
            b_lower_rad = np.deg2rad(b_lower)
            x_l = np.cos(b_lower_rad) * np.cos(l_rad)
            y_l = np.cos(b_lower_rad) * np.sin(l_rad)
            z_l = np.sin(b_lower_rad)
            gal_lower = np.array([x_l, y_l, z_l], dtype=np.float32)
            
            # Convert to equatorial
            eq_upper = self._galactic_to_equatorial(gal_upper) * self.radius
            eq_lower = self._galactic_to_equatorial(gal_lower) * self.radius
            
            upper_edge.append(eq_upper)
            lower_edge.append(eq_lower)
        
        return (np.array(upper_edge, dtype=np.float32),
                np.array(lower_edge, dtype=np.float32),
                np.array(widths, dtype=np.float32))
    
    def get_milky_way_description(self) -> str:
        """Get a description of the Milky Way structure"""
        return """
        Milky Way Structure:
        
        The Milky Way appears as a luminous band across the sky because Earth is
        located within the galactic disk. We see the combined light of billions
        of stars along our line of sight through the disk.
        
        Key regions:
        - Galactic Center (Sagittarius): Brightest and widest, l=0°
        - Cygnus-Scutum (Summer): Very bright, l=60-90°
        - Perseus Arm: Moderate brightness, l=120-150°
        - Anticenter (Auriga): Dimmest and narrowest, l=180°
        
        Dark lanes are caused by interstellar dust that absorbs starlight,
        most prominent in the Great Rift near Cygnus.
        """