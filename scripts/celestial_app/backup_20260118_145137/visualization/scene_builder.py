import numpy as np
import pyqtgraph.opengl as gl
from geometry.mesh_generation import (
    make_uv_sphere, 
    get_earth_texture, 
    sample_texture,
    make_ring,
    make_disk_mesh
)
from astronomy.galactic import build_milky_way_band_equatorial

# Colors (r,g,b,a) in 0..1
ORANGE = (1.0, 0.55, 0.0, 1.0)
WHITE = (0.92, 0.92, 0.92, 1.0)
GOLD = (1.0, 0.84, 0.0, 1.0)
SUN_ANG_RADIUS_DEG = 0.265
SKY_ALWAYS_ON_TOP = True

class SceneBuilder:
    """Build and manage 3D scene components"""
    
    def __init__(self, radius: float = 1.0, earth_radius: float = 0.36):
        self.radius = radius
        self.earth_radius = earth_radius
    
    def _sky_options(self, item, depth_value: int):
        """Apply sky rendering options"""
        if SKY_ALWAYS_ON_TOP:
            item.setGLOptions("additive")
            item.setDepthValue(depth_value)
        else:
            item.setGLOptions("translucent")
            item.setDepthValue(0)
    
    def build_celestial_sphere(self) -> gl.GLMeshItem:
        """Create celestial sphere shell"""
        s_verts, s_faces, _ = make_uv_sphere(self.radius, n_lon=120, n_lat=60)
        sky_md = gl.MeshData(vertexes=s_verts, faces=s_faces)
        item = gl.GLMeshItem(meshdata=sky_md, smooth=True, drawFaces=True, 
                            drawEdges=False, shader="shaded")
        item.setColor((0.50, 0.55, 0.65, 0.18))
        self._sky_options(item, -10000)
        return item
    
    def build_earth(self, texture_path: str = "earth_texture.jpg") -> gl.GLMeshItem:
        """Create Earth with texture"""
        verts, faces, uv = make_uv_sphere(self.earth_radius, n_lon=180, n_lat=90)
        tex = get_earth_texture(texture_path)
        colors = sample_texture(tex, uv)
        colors_rgba = np.concatenate([colors, np.ones((colors.shape[0], 1), dtype=np.float32)], axis=1)
        earth_md = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors_rgba)
        item = gl.GLMeshItem(meshdata=earth_md, smooth=True, drawEdges=False, shader="shaded")
        item.setGLOptions("opaque")
        item.setDepthValue(10000)
        return item
    
    def build_location_marker(self) -> gl.GLMeshItem:
        """Create location marker sphere"""
        marker_r = self.earth_radius * 0.05
        m_verts, m_faces, _ = make_uv_sphere(marker_r, n_lon=36, n_lat=18)
        m_md = gl.MeshData(vertexes=m_verts, faces=m_faces)
        item = gl.GLMeshItem(meshdata=m_md, smooth=True, drawEdges=False, shader="shaded")
        item.setColor(ORANGE)
        item.setGLOptions("translucent")
        item.setDepthValue(20000)
        return item
    
    def build_horizon_ring(self, initial_pts: np.ndarray) -> gl.GLLinePlotItem:
        """Create horizon ring"""
        item = gl.GLLinePlotItem(pos=initial_pts, width=2.5, antialias=True, color=ORANGE)
        item.setGLOptions("translucent")
        item.setDepthValue(25000)
        return item
    
    def build_celestial_equator(self) -> gl.GLLinePlotItem:
        """Create celestial equator ring"""
        eq = make_ring(self.radius, 600, "xy")
        item = gl.GLLinePlotItem(pos=eq, width=1.4, antialias=True, color=WHITE)
        self._sky_options(item, 999998)
        return item
    
    def build_milky_way(self) -> tuple[gl.GLScatterPlotItem, np.ndarray, np.ndarray]:
        """Create Milky Way band"""
        mw_pts, mw_a = build_milky_way_band_equatorial(
            self.radius, half_width_deg=10.0, n=1600, m=33, seed=7
        )
        mw_cols = np.ones((mw_pts.shape[0], 4), dtype=np.float32)
        mw_cols[:, 3] = 0.08 + 0.75 * np.clip(mw_a, 0.0, 1.0)
        item = gl.GLScatterPlotItem(pos=mw_pts, size=2.0, color=mw_cols, pxMode=True)
        self._sky_options(item, 999999)
        return item, mw_pts, mw_cols
    
    def build_sun(self) -> tuple[gl.GLMeshItem, gl.GLScatterPlotItem]:
        """Create Sun disk and dot"""
        disk_radius = self.radius * np.sin(np.deg2rad(SUN_ANG_RADIUS_DEG))
        md = make_disk_mesh(
            center=np.array([self.radius, 0.0, 0.0], dtype=np.float32),
            normal=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            radius=disk_radius,
            segments=56
        )
        sun_disk = gl.GLMeshItem(meshdata=md, smooth=True, drawEdges=False, shader="shaded")
        sun_disk.setColor((1.0, 0.78, 0.12, 1.0))
        sun_disk.setGLOptions("translucent")
        sun_disk.setDepthValue(30000)
        
        sun_dot = gl.GLScatterPlotItem(
            pos=np.array([[self.radius, 0.0, 0.0]], dtype=np.float32),
            size=12.0,
            color=np.array([[1.0, 0.78, 0.12, 1.0]], dtype=np.float32),
            pxMode=True
        )
        sun_dot.setGLOptions("translucent")
        sun_dot.setDepthValue(30000)
        
        return sun_disk, sun_dot
    
    def build_earth_axis(self) -> gl.GLLinePlotItem:
        """Create Earth rotation axis line"""
        axis_pts = np.array([
            [0.0, 0.0, -self.radius],
            [0.0, 0.0,  self.radius]
        ], dtype=np.float32)
        item = gl.GLLinePlotItem(pos=axis_pts, width=4.5, antialias=True, color=GOLD)
        item.setGLOptions("translucent")
        item.setDepthValue(20000)
        return item
    
    def build_galactic_center_dot(self) -> gl.GLScatterPlotItem:
        """Create galactic center marker"""
        item = gl.GLScatterPlotItem(
            pos=np.array([[self.radius, 0.0, 0.0]], dtype=np.float32),
            size=9.0,
            color=np.array([[0.65, 0.80, 1.0, 1.0]], dtype=np.float32),
            pxMode=True
        )
        self._sky_options(item, 999997)
        return item