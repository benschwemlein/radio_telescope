"""
Star Chart View - Flat 2D projection of the sky as seen from observer's location
Similar to a planisphere or star finder that can be printed
"""
import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from astronomy.coordinates import (
    equatorial_to_local_enu_matrix,
    unit_vector_enu_to_alt_az,
    ra_dec_to_unit_vector_equatorial,
)
from astronomy.celestial_objects import sun_ra_dec_degrees, galactic_center_unit_eq
from astronomy.galactic import build_milky_way_band_equatorial


class StarChartView(QtWidgets.QWidget):
    """
    Flat 2D star chart view - overhead projection of the sky
    Shows what you would see looking up, suitable for printing
    """
    
    def __init__(self, radius=1.0):
        super().__init__()
        self.radius = radius
        
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
        
        # Color scheme
        self.inverted = False
        self.bg_color = '#0a0a0e'
        self.fg_color = 'white'
        self.grid_color = '#404050'
        
        # Initialize the axes
        self.ax = self.figure.add_subplot(111, projection='polar')
        self._setup_axes()
    
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
        
        # Azimuth labels (N, E, S, W)
        self.ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        self.ax.set_xticklabels(['N', 'E', 'S', 'W'], color=self.fg_color, fontsize=14, weight='bold')
        
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
        """Plot the Milky Way band"""
        # Generate Milky Way points in equatorial coordinates
        mw_pts_eq, mw_a = build_milky_way_band_equatorial(
            self.radius, half_width_deg=10.0, n=1600, m=33, seed=7
        )
        
        # Transform to local coordinates
        mw_dirs_eq = mw_pts_eq / self.radius
        mw_local = (M @ mw_dirs_eq.T).T
        
        # Convert to alt/az
        alts = []
        azs = []
        alphas = []
        
        for i, v in enumerate(mw_local):
            v = v / (np.linalg.norm(v) + 1e-12)
            alt, az = unit_vector_enu_to_alt_az(v)
            
            if alt > 0:  # Only plot if above horizon
                alts.append(90 - alt)  # Convert altitude to zenith distance
                # Matplotlib polar: 0°=N (top), increases clockwise with theta_direction=-1
                # Astronomy: azimuth typically 0°=N, increases clockwise
                # So we use az directly
                azs.append(np.deg2rad(az))
                alphas.append(0.1 + 0.7 * np.clip(mw_a[i], 0.0, 1.0))
        
        if alts:
            # Plot as scatter
            colors = [(1, 1, 1, a) for a in alphas]
            self.ax.scatter(azs, alts, s=2, c=colors, marker='.', zorder=1)
    
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
            local_vec = local_vec / (np.linalg.norm(local_vec) + 1e-12)
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
        sun_local = sun_local / (np.linalg.norm(sun_local) + 1e-12)
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
        gc_local = gc_local / (np.linalg.norm(gc_local) + 1e-12)
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
            if len(label) == 2:  # Intermediate directions smaller
                fontsize = 8
                color = self.fg_color
                alpha = 0.5
            else:  # Cardinal directions
                fontsize = 10
                color = '#ff6060' if label == 'N' else self.fg_color
                alpha = 0.8
            
            self.ax.text(np.deg2rad(azimuth), 95, label,
                        ha='center', va='center',
                        fontsize=fontsize, color=color, 
                        alpha=alpha, weight='bold')
    
    def toggle_colors(self):
        """Toggle between dark and light color scheme for printing"""
        self.inverted = not self.inverted
        
        if self.inverted:
            self.bg_color = 'white'
            self.fg_color = 'black'
            self.grid_color = '#c0c0c0'
        else:
            self.bg_color = '#0a0a0e'
            self.fg_color = 'white'
            self.grid_color = '#404050'
        
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