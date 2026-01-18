
import pyqtgraph.opengl as gl
import numpy as np
from geometry.mesh_generation import make_uv_sphere

class FixedGLViewWidget(gl.GLViewWidget):
    """Custom GL view widget with fixed camera distance"""
    
    def wheelEvent(self, ev):
        """Disable zoom via wheel or trackpad scroll"""
        ev.ignore()
    
    def set_fixed_distance(self, d: float):
        """Set fixed camera distance"""
        self.opts["distance"] = float(d)
        self.update()


# Mesh Factory Functions

def create_sphere_marker(radius: float, 
                        color: tuple, 
                        position: tuple = (0, 0, 0),
                        n_lon: int = 16, 
                        n_lat: int = 8,
                        translucent: bool = True,
                        depth_value: int = None) -> gl.GLMeshItem:
    """
    Create a colored sphere mesh marker at a given position.
    
    Args:
        radius: Sphere radius
        color: RGBA tuple (r, g, b, a) with values 0.0-1.0
        position: (x, y, z) position tuple
        n_lon: Number of longitude segments
        n_lat: Number of latitude segments
        translucent: Enable transparency
        depth_value: Optional depth value for rendering order
    
    Returns:
        Configured GLMeshItem
    """
    verts, faces, _ = make_uv_sphere(radius, n_lon=n_lon, n_lat=n_lat)
    md = gl.MeshData(vertexes=verts, faces=faces)
    marker = gl.GLMeshItem(meshdata=md, smooth=True, drawEdges=False, shader="shaded")
    marker.setColor(color)
    
    if translucent:
        marker.setGLOptions("translucent")
    
    if depth_value is not None:
        marker.setDepthValue(depth_value)
    
    marker.translate(*position)
    return marker


def create_mesh_from_vertices(vertices: np.ndarray,
                              faces: np.ndarray = None,
                              color: tuple = None,
                              draw_edges: bool = False,
                              translucent: bool = False) -> gl.GLMeshItem:
    """
    Create a mesh from vertices and optional faces.
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices (if None, no faces drawn)
        color: RGBA color tuple (if None, use default)
        draw_edges: Whether to draw edges
        translucent: Enable transparency
    
    Returns:
        Configured GLMeshItem
    """
    md = gl.MeshData(vertexes=vertices, faces=faces)
    mesh = gl.GLMeshItem(meshdata=md, smooth=True, drawEdges=draw_edges, shader="shaded")
    
    if color is not None:
        mesh.setColor(color)
    
    if translucent:
        mesh.setGLOptions("translucent")
    
    return mesh

