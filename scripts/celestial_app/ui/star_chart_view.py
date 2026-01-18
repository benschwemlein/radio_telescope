"""
Star Chart View - Flat 2D projection of the sky as seen from observer's location
Similar to a planisphere or star finder that can be printed
"""
import numpy as np
from geometry.transformations import normalize_vector
from PyQt6 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from ui.theme import ColorTheme
from astronomy.coordinates import (
    equatorial_to_local_enu_matrix,
    unit_vector_enu_to_alt_az,
    ra_dec_to_unit_vector_equatorial,
)
from astronomy.celestial_objects import sun_ra_dec_degrees, galactic_center_unit_eq
from visualization.milky_way_renderer import MilkyWayRenderer

# Bright stars catalog (name, RA hours, Dec degrees, magnitude)
BRIGHT_STARS = [
    # Winter constellations - Orion area
    ("Sirius", 6.752, -16.716, -1.46),      # Canis Major - brightest star
    ("Betelgeuse", 5.919, 7.407, 0.50),     # Orion - red supergiant
    ("Rigel", 5.242, -8.202, 0.13),         # Orion - blue supergiant
    ("Bellatrix", 5.418, 6.350, 1.64),      # Orion
    ("Alnilam", 5.603, -1.202, 1.69),       # Orion belt center
    ("Alnitak", 5.679, -1.943, 1.77),       # Orion belt left
    ("Mintaka", 5.533, -0.299, 2.23),       # Orion belt right
    ("Saiph", 5.796, -9.669, 2.06),         # Orion
    
    ("Procyon", 7.655, 5.225, 0.38),        # Canis Minor
    ("Adhara", 6.977, -28.972, 1.50),       # Canis Major
    
    ("Aldebaran", 4.599, 16.509, 0.85),     # Taurus - red giant
    ("Elnath", 5.438, 28.608, 1.65),        # Taurus
    
    ("Capella", 5.278, 45.998, 0.08),       # Auriga
    
    ("Pollux", 7.755, 28.026, 1.14),        # Gemini
    ("Castor", 7.577, 31.888, 1.58),        # Gemini
    
    # Spring constellations
    ("Regulus", 10.139, 11.967, 1.35),      # Leo
    ("Denebola", 11.818, 14.572, 2.14),     # Leo
    ("Algieba", 10.333, 19.841, 2.08),      # Leo
    
    ("Arcturus", 14.261, 19.182, -0.05),    # Boötes - orange giant
    
    ("Spica", 13.420, -11.161, 0.97),       # Virgo
    
    # Summer constellations
    ("Vega", 18.615, 38.783, 0.03),         # Lyra - summer triangle
    ("Altair", 19.846, 8.868, 0.77),        # Aquila - summer triangle
    ("Deneb", 20.690, 45.280, 1.25),        # Cygnus - summer triangle
    
    ("Antares", 16.490, -26.432, 0.96),     # Scorpius - red supergiant
    ("Shaula", 17.560, -37.104, 1.63),      # Scorpius
    
    ("Rasalhague", 17.582, 12.560, 2.08),   # Ophiuchus
    
    # Fall constellations
    ("Fomalhaut", 22.961, -29.622, 1.16),   # Piscis Austrinus
    
    ("Altair", 19.846, 8.868, 0.77),        # Aquila
    
    ("Markab", 23.079, 15.205, 2.49),       # Pegasus
    ("Scheat", 23.063, 28.083, 2.42),       # Pegasus
    ("Algenib", 0.220, 15.184, 2.83),       # Pegasus
    ("Alpheratz", 0.140, 29.091, 2.06),     # Andromeda/Pegasus
    
    ("Mirach", 1.162, 35.621, 2.06),        # Andromeda
    ("Almach", 2.065, 42.330, 2.26),        # Andromeda
    
    # Circumpolar stars (visible year-round from mid-northern latitudes)
    ("Polaris", 2.530, 89.264, 1.98),       # Ursa Minor - North Star
    
    # Big Dipper / Ursa Major
    ("Dubhe", 11.062, 61.751, 1.79),        # Big Dipper
    ("Merak", 11.031, 56.383, 2.37),        # Big Dipper
    ("Phecda", 11.897, 53.695, 2.44),       # Big Dipper
    ("Megrez", 12.257, 57.033, 3.31),       # Big Dipper
    ("Alioth", 12.900, 55.960, 1.77),       # Big Dipper
    ("Mizar", 13.398, 54.925, 2.27),        # Big Dipper
    ("Alkaid", 13.792, 49.313, 1.86),       # Big Dipper
    
    # Cassiopeia
    ("Schedar", 0.675, 56.537, 2.23),       # Cassiopeia
    ("Caph", 0.153, 59.150, 2.27),          # Cassiopeia
    ("Gamma Cas", 0.945, 60.717, 2.47),     # Cassiopeia
    ("Ruchbah", 1.430, 60.235, 2.68),       # Cassiopeia
    ("Segin", 1.901, 63.670, 3.38),         # Cassiopeia
    
    # Little Dipper / Ursa Minor
    ("Kochab", 14.845, 74.155, 2.08),       # Ursa Minor
    ("Pherkad", 15.345, 71.834, 3.05),      # Ursa Minor
]

# Constellation lines (RA in hours, Dec in degrees for star pairs)
CONSTELLATION_LINES = {
    "Orion": [
        # Belt (right to left looking at it)
        (5.679, -1.943, 5.603, -1.202),    # Alnitak to Alnilam
        (5.603, -1.202, 5.533, -0.299),    # Alnilam to Mintaka
        # Shoulders to belt
        (5.919, 7.407, 5.418, 6.350),      # Betelgeuse to Bellatrix
        (5.418, 6.350, 5.533, -0.299),     # Bellatrix to Mintaka
        (5.919, 7.407, 5.603, -1.202),     # Betelgeuse to Alnilam  
        # Legs from belt
        (5.679, -1.943, 5.796, -9.669),    # Alnitak to Saiph
        (5.533, -0.299, 5.242, -8.202),    # Mintaka to Rigel
        (5.796, -9.669, 5.242, -8.202),    # Saiph to Rigel
    ],
    "Big Dipper": [
        # Handle
        (13.792, 49.313, 13.398, 54.925),  # Alkaid to Mizar
        (13.398, 54.925, 12.900, 55.960),  # Mizar to Alioth
        (12.900, 55.960, 12.257, 57.033),  # Alioth to Megrez
        # Bowl
        (12.257, 57.033, 11.897, 53.695),  # Megrez to Phecda
        (11.897, 53.695, 11.031, 56.383),  # Phecda to Merak
        (11.031, 56.383, 11.062, 61.751),  # Merak to Dubhe
        (11.062, 61.751, 12.257, 57.033),  # Dubhe to Megrez (close the bowl)
    ],
    "Cassiopeia": [
        # W shape (left to right)
        (0.153, 59.150, 0.675, 56.537),    # Caph to Schedar
        (0.675, 56.537, 0.945, 60.717),    # Schedar to Gamma Cas
        (0.945, 60.717, 1.430, 60.235),    # Gamma Cas to Ruchbah
        (1.430, 60.235, 1.901, 63.670),    # Ruchbah to Segin
    ],
    "Leo": [
        # Sickle (head)
        (10.139, 11.967, 10.333, 19.841),  # Regulus to Algieba
        (10.333, 19.841, 10.278, 23.417),  # Algieba to head
        # Body
        (10.139, 11.967, 11.237, 20.524),  # Regulus to back
        (11.237, 20.524, 11.818, 14.572),  # Back to Denebola (tail)
    ],
    "Gemini": [
        # The twins
        (7.577, 31.888, 7.755, 28.026),    # Castor to Pollux
        (7.577, 31.888, 6.629, 25.131),    # Castor down
        (7.755, 28.026, 6.248, 22.514),    # Pollux down
    ],
    "Taurus": [
        # V shape of Hyades cluster with Aldebaran
        (4.599, 16.509, 4.476, 15.871),    # Aldebaran to nearby
        (4.599, 16.509, 4.329, 15.627),    # Aldebaran to V
        # To Elnath
        (4.599, 16.509, 5.438, 28.608),    # Aldebaran to Elnath (horn)
    ],
    "Cygnus": [
        # Northern Cross
        (20.690, 45.280, 20.370, 40.257),  # Deneb to center
        (20.370, 40.257, 19.512, 27.960),  # Center to Albireo (head)
        # Cross bar
        (19.930, 35.083, 20.370, 40.257),  # Left wing to center
        (20.370, 40.257, 21.227, 30.227),  # Center to right wing
    ],
    "Lyra": [
        # Small parallelogram with Vega
        (18.615, 38.783, 18.982, 32.689),  # Vega to Sheliak
        (18.982, 32.689, 18.746, 33.363),  # Sheliak to corner
        (18.746, 33.363, 18.835, 36.898),  # Corner to Sulafat
        (18.835, 36.898, 18.615, 38.783),  # Sulafat back to Vega
    ],
    "Scorpius": [
        # Head and body
        (16.490, -26.432, 16.836, -28.216), # Antares to body
        (16.836, -28.216, 17.560, -37.104), # Body to Shaula (tail)
        # Claws
        (16.090, -19.461, 16.490, -26.432), # Claw to Antares
    ],
    "Pegasus": [
        # Great Square
        (23.079, 15.205, 23.063, 28.083),  # Markab to Scheat
        (23.063, 28.083, 0.220, 15.184),   # Scheat to Algenib  
        (0.220, 15.184, 23.079, 15.205),   # Algenib to Markab
        (23.063, 28.083, 0.140, 29.091),   # Scheat to Alpheratz
    ],
    "Andromeda": [
        # Line of stars from Pegasus
        (0.140, 29.091, 1.162, 35.621),    # Alpheratz to Mirach
        (1.162, 35.621, 2.065, 42.330),    # Mirach to Almach
    ],
}


class StarChartView(QtWidgets.QWidget):
    """
    Flat 2D star chart view - overhead projection of the sky
    Shows what you would see looking up, suitable for printing
    """
    
    def __init__(self, radius=1.0):
        super().__init__()
        self.radius = radius
        
        # Initialize Milky Way renderer with background image
        self.milky_way_renderer = MilkyWayRenderer(radius=radius)
        # Load the Milky Way background image from app directory
        try:
            self.milky_way_renderer.load_milky_way_image('gal_background.jpg')
        except Exception as e:
            print(f"Could not load Milky Way background image: {e}")
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 10), facecolor='#0a0a0e')
        self.canvas = FigureCanvasQTAgg(self.figure)
        
        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Add toolbar
        toolbar = QtWidgets.QHBoxLayout()
        self.print_btn = QtWidgets.QPushButton("Print/Save Chart")
        self.invert_btn = QtWidgets.QPushButton("Invert Colors (for printing)")
        toolbar.addWidget(self.print_btn)
        toolbar.addWidget(self.invert_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        layout.addWidget(self.canvas)
        
        # Connect buttons
        self.print_btn.clicked.connect(self.save_chart)
        self.invert_btn.clicked.connect(self.toggle_colors)
        
        # Color scheme using theme manager
        self.inverted = False
        self._apply_theme()
        
        # Initialize the axes
        self.ax = self.figure.add_subplot(111, projection='polar')
        self._setup_axes()
    
    def _apply_theme(self):
        """Apply color theme based on inverted state"""
        theme = ColorTheme.get_theme(dark_mode=not self.inverted)
        self.bg_color = theme['bg_color']
        self.fg_color = theme['fg_color']
        self.grid_color = theme['grid_color']
    
    def _setup_axes(self):
        """Setup polar axes for sky projection"""
        self.ax.clear()
        
        # Set up polar plot (azimuth is theta, altitude affects radius)
        # Radius goes from 0 (zenith) to 90 (horizon)
        self.ax.set_ylim(0, 90)
        self.ax.set_theta_zero_location('N')  # North at top
        self.ax.set_theta_direction(1)  # Counter-clockwise (to match overhead view)
        
        # Grid styling
        self.ax.set_facecolor(self.bg_color)
        self.ax.grid(True, color=self.grid_color, linestyle=':', alpha=0.5)
        
        # Altitude circles every 30 degrees
        self.ax.set_yticks([0, 30, 60, 90])
        self.ax.set_yticklabels(['90°', '60°', '30°', '0°'], color=self.fg_color, fontsize=10)
        
        # Azimuth ticks (labels handled by _add_compass_rose)
        self.ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        self.ax.set_xticklabels([])  # Empty labels - custom compass rose adds them
        
        self.ax.spines['polar'].set_color(self.fg_color)
        self.ax.tick_params(colors=self.fg_color)
    
    def update_chart(self, lat, lon, lst, dt_local, dt_utc):
        """
        Update the star chart for given time and location
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees  
            lst: Local Sidereal Time in degrees
            dt_local: Local datetime
            dt_utc: UTC datetime (naive)
        """
        self._setup_axes()
        
        # Get transformation matrix
        M = equatorial_to_local_enu_matrix(lat, lst)
        
        # Plot dimmer background stars first
        self._plot_dimmer_stars(M)
        
        # Plot constellation lines
        self._plot_constellation_lines(M)
        
        # Plot bright stars
        self._plot_bright_stars(M)
        
        # Plot Milky Way
        self._plot_milky_way(M)
        
        # Plot celestial equator
        self._plot_celestial_equator(M)
        
        # Plot Sun
        self._plot_sun(M, dt_utc)
        
        # Plot Galactic Center
        self._plot_galactic_center(M)
        
        # Add compass rose at horizon
        self._add_compass_rose()
        
        # Add title with location and time info
        title = (f"Sky View from Lat {lat:.2f}°, Lon {lon:.2f}°\n"
                f"{dt_local.strftime('%Y-%m-%d %H:%M:%S')} Local")
        self.ax.set_title(title, color=self.fg_color, fontsize=12, pad=20)
        
        self.canvas.draw()
    
    def _plot_milky_way(self, M):
        """Plot the Milky Way as a background texture image"""
        if self.milky_way_renderer.milky_way_image is None:
            return  # No image loaded
        
        # Create a grid covering the entire sky chart
        # We'll sample the Milky Way image at each grid point
        n_az = 360  # Azimuth resolution
        n_alt = 90  # Altitude resolution (from horizon to zenith)
        
        # Create meshgrid for the chart (polar coordinates)
        az_grid = np.linspace(0, 2*np.pi, n_az)
        alt_grid = np.linspace(0, 90, n_alt)  # Zenith distance from 0 (zenith) to 90 (horizon)
        
        AZ, ALT = np.meshgrid(az_grid, alt_grid)
        
        # Create brightness grid by sampling the Milky Way image
        brightness_grid = np.zeros_like(AZ)
        
        for i in range(n_alt):
            for j in range(n_az):
                # Convert from chart coordinates to sky coordinates
                az_deg = np.rad2deg(az_grid[j])
                zenith_dist = alt_grid[i]
                altitude = 90 - zenith_dist
                
                if altitude < 0:
                    continue  # Below horizon
                
                # Convert alt/az to local ENU unit vector
                alt_rad = np.deg2rad(altitude)
                az_rad = az_grid[j]
                
                # ENU unit vector from alt/az
                E = np.cos(alt_rad) * np.sin(az_rad)
                N = np.cos(alt_rad) * np.cos(az_rad)
                U = np.sin(alt_rad)
                local_vec = np.array([E, N, U], dtype=np.float32)
                
                # Transform to equatorial coordinates (inverse of M)
                M_inv = np.linalg.inv(M)
                eq_vec = (M_inv @ local_vec).astype(np.float32)
                eq_vec = normalize_vector(eq_vec)
                
                # Convert equatorial to galactic coordinates
                gal_vec = self.milky_way_renderer._equatorial_to_galactic(eq_vec)
                
                # Convert galactic Cartesian to galactic (l, b)
                l_rad = np.arctan2(gal_vec[1], gal_vec[0])
                l_deg = np.rad2deg(l_rad) % 360
                
                b_rad = np.arcsin(np.clip(gal_vec[2], -1, 1))
                b_deg = np.rad2deg(b_rad)
                
                # Sample brightness from Milky Way image
                brightness = self.milky_way_renderer.sample_milky_way_at_galactic_coords(l_deg, b_deg)
                brightness_grid[i, j] = brightness
        
        # Plot as a colored mesh
        # Use a colormap that looks like the Milky Way
        from matplotlib.colors import LinearSegmentedColormap
        
        # Create a custom colormap: black -> dark blue -> light blue/white
        colors = [(0, 0, 0), (0.05, 0.05, 0.15), (0.3, 0.35, 0.5), (0.7, 0.75, 0.85), (0.95, 0.97, 1.0)]
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('milkyway', colors, N=n_bins)
        
        # Plot with transparency based on brightness
        self.ax.pcolormesh(AZ, ALT, brightness_grid, 
                          cmap=cmap, 
                          shading='gouraud',
                          alpha=0.6,
                          zorder=0.5,
                          vmin=0, vmax=1)
    
    def _plot_bright_stars(self, M):
        """Plot bright stars"""
        for star_name, ra_hrs, dec_deg, mag in BRIGHT_STARS:
            # Convert RA from hours to degrees
            ra_deg = ra_hrs * 15.0
            
            # Get unit vector in equatorial coords
            star_eq = ra_dec_to_unit_vector_equatorial(ra_deg, dec_deg)
            
            # Transform to local coords
            star_local = (M @ star_eq).astype(np.float32)
            star_local = normalize_vector(star_local)
            
            # Convert to alt/az
            alt, az = unit_vector_enu_to_alt_az(star_local)
            
            if alt > 0:  # Only plot if above horizon
                zenith_dist = 90 - alt
                az_rad = np.deg2rad(az)
                
                # Size based on magnitude (brighter = larger)
                # Magnitude scale: negative is brighter
                size = 100 * (2.0 ** (-mag))  # Exponential scaling
                size = np.clip(size, 20, 400)  # Limit size range
                
                # Plot star
                self.ax.scatter([az_rad], [zenith_dist], s=size, c='white',
                              marker='*', edgecolors='yellow', linewidths=0.5,
                              zorder=5, alpha=0.9)
                
                # Label bright stars (magnitude < 1.5)
                if mag < 1.5 and zenith_dist < 85:  # Don't label if too close to horizon
                    self.ax.text(az_rad, zenith_dist - 3, star_name,
                               ha='center', va='top', fontsize=6,
                               color=self.fg_color, alpha=0.7, zorder=6)
    
    def _plot_dimmer_stars(self, M):
        """Plot a grid of dimmer stars to fill out the sky"""
        # Generate a grid of stars across the sky (simplified star field)
        # RA: 0 to 24 hours, Dec: -60 to +90 degrees (visible from northern hemisphere)
        ra_hrs_grid = np.arange(0, 24, 0.5)  # Every 0.5 hours
        dec_grid = np.arange(-60, 90, 10)    # Every 10 degrees
        
        for ra_hrs in ra_hrs_grid:
            for dec_deg in dec_grid:
                # Skip if too close to actual bright stars
                skip = False
                for _, star_ra, star_dec, _ in BRIGHT_STARS:
                    if abs(star_ra - ra_hrs) < 0.3 and abs(star_dec - dec_deg) < 5:
                        skip = True
                        break
                
                if skip:
                    continue
                
                # Convert RA from hours to degrees
                ra_deg = ra_hrs * 15.0
                
                # Get unit vector in equatorial coords
                star_eq = ra_dec_to_unit_vector_equatorial(ra_deg, dec_deg)
                
                # Transform to local coords
                star_local = (M @ star_eq).astype(np.float32)
                star_local = normalize_vector(star_local)
                
                # Convert to alt/az
                alt, az = unit_vector_enu_to_alt_az(star_local)
                
                if alt > 5:  # Only plot if well above horizon
                    zenith_dist = 90 - alt
                    az_rad = np.deg2rad(az)
                    
                    # Small size for dim stars
                    self.ax.scatter([az_rad], [zenith_dist], s=3, c='white',
                                  marker='.', alpha=0.4, zorder=3)
    
    def _plot_constellation_lines(self, M):
        """Plot constellation lines"""
        for const_name, lines in CONSTELLATION_LINES.items():
            for ra1_hrs, dec1, ra2_hrs, dec2 in lines:
                # Convert RA from hours to degrees
                ra1_deg = ra1_hrs * 15.0
                ra2_deg = ra2_hrs * 15.0
                
                # Get unit vectors
                star1_eq = ra_dec_to_unit_vector_equatorial(ra1_deg, dec1)
                star2_eq = ra_dec_to_unit_vector_equatorial(ra2_deg, dec2)
                
                # Transform to local coords
                star1_local = (M @ star1_eq).astype(np.float32)
                star2_local = (M @ star2_eq).astype(np.float32)
                
                star1_local = normalize_vector(star1_local)
                star2_local = normalize_vector(star2_local)
                
                # Convert to alt/az
                alt1, az1 = unit_vector_enu_to_alt_az(star1_local)
                alt2, az2 = unit_vector_enu_to_alt_az(star2_local)
                
                # Only plot if both stars are above horizon
                if alt1 > 0 and alt2 > 0:
                    zenith1 = 90 - alt1
                    zenith2 = 90 - alt2
                    az1_rad = np.deg2rad(az1)
                    az2_rad = np.deg2rad(az2)
                    
                    # Draw line
                    self.ax.plot([az1_rad, az2_rad], [zenith1, zenith2],
                               color='cyan', linewidth=0.8, alpha=0.4, zorder=2)
    
    
    def _plot_celestial_equator(self, M):
        """Plot the celestial equator"""
        # Generate points along celestial equator
        ra_points = np.linspace(0, 360, 360, endpoint=False)
        dec = 0.0  # Equator
        
        alts = []
        azs = []
        
        for ra in ra_points:
            eq_vec = ra_dec_to_unit_vector_equatorial(ra, dec)
            local_vec = (M @ eq_vec).astype(np.float32)
            local_vec = normalize_vector(local_vec)
            alt, az = unit_vector_enu_to_alt_az(local_vec)
            
            if alt > 0:  # Only plot if above horizon
                alts.append(90 - alt)
                azs.append(np.deg2rad(az))
        
        if alts:
            self.ax.plot(azs + [azs[0]], alts + [alts[0]], 
                        color='#9090a0', linewidth=1, linestyle='--', 
                        alpha=0.5, label='Celestial Equator', zorder=2)
    
    def _plot_sun(self, M, dt_utc):
        """Plot the Sun"""
        sun_ra, sun_dec = sun_ra_dec_degrees(dt_utc)
        sun_eq = ra_dec_to_unit_vector_equatorial(sun_ra, sun_dec)
        sun_local = (M @ sun_eq).astype(np.float32)
        sun_local = normalize_vector(sun_local)
        alt, az = unit_vector_enu_to_alt_az(sun_local)
        
        if alt > 0:  # Only plot if above horizon
            zenith_dist = 90 - alt
            az_rad = np.deg2rad(az)
            
            # Draw sun as a large yellow circle
            self.ax.scatter([az_rad], [zenith_dist], s=300, c='#ffd700', 
                          marker='o', edgecolors='#ffaa00', linewidths=2,
                          label=f'Sun (Alt {alt:.1f}°)', zorder=10)
    
    def _plot_galactic_center(self, M):
        """Plot the Galactic Center"""
        gc_eq = galactic_center_unit_eq()
        gc_local = (M @ gc_eq).astype(np.float32)
        gc_local = normalize_vector(gc_local)
        alt, az = unit_vector_enu_to_alt_az(gc_local)
        
        if alt > 0:  # Only plot if above horizon
            zenith_dist = 90 - alt
            az_rad = np.deg2rad(az)
            
            # Draw galactic center as a blue star
            self.ax.scatter([az_rad], [zenith_dist], s=150, c='#4080ff', 
                          marker='*', edgecolors='white', linewidths=1,
                          label=f'Galactic Center (Alt {alt:.1f}°)', zorder=9)
    
    def _add_compass_rose(self):
        """Add compass direction markers at the horizon"""
        directions = {
            'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
            'S': 180, 'SW': 225, 'W': 270, 'NW': 315
        }
        
        for label, azimuth in directions.items():
            if len(label) == 2:  # Intermediate directions smaller and farther out
                fontsize = 8
                color = self.fg_color
                alpha = 0.5
                radius = 97  # Place intermediate directions farther from center
            else:  # Cardinal directions
                fontsize = 10
                color = '#ff6060' if label == 'N' else self.fg_color
                alpha = 0.8
                radius = 95  # Cardinal directions closer
            
            self.ax.text(np.deg2rad(azimuth), radius, label,
                        ha='center', va='center',
                        fontsize=fontsize, color=color, 
                        alpha=alpha, weight='bold')
    
    def toggle_colors(self):
        """Toggle between dark and light color scheme for printing"""
        self.inverted = not self.inverted
        self._apply_theme()
        self.figure.set_facecolor(self.bg_color)
        # Trigger a redraw by calling update_chart from parent
        # (parent will need to call this)
    
    def save_chart(self):
        """Save the chart as a file"""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Star Chart",
            "star_chart.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        
        if filename:
            self.figure.savefig(filename, dpi=300, bbox_inches='tight',
                              facecolor=self.bg_color)
            QtWidgets.QMessageBox.information(
                self,
                "Saved",
                f"Star chart saved to:\n{filename}"
            )