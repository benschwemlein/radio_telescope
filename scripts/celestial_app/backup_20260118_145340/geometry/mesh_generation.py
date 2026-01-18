import os
import numpy as np
import requests
from PIL import Image
import pyqtgraph.opengl as gl

def make_uv_sphere(radius: float, n_lon: int, n_lat: int):
    """Generate UV sphere mesh"""
    lon = np.linspace(-np.pi, np.pi, n_lon, endpoint=False).astype(np.float32)
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat).astype(np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")
    x = radius * np.cos(lat_grid) * np.cos(lon_grid)
    y = radius * np.cos(lat_grid) * np.sin(lon_grid)
    z = radius * np.sin(lat_grid)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    faces = []
    for j in range(n_lat - 1):
        for i in range(n_lon):
            i2 = (i + 1) % n_lon
            a = j * n_lon + i
            b = j * n_lon + i2
            c = (j + 1) * n_lon + i
            d = (j + 1) * n_lon + i2
            faces.append([a, c, b])
            faces.append([b, c, d])
    faces = np.array(faces, dtype=np.int32)
    u = (lon_grid + np.pi) / (2 * np.pi)
    v = (0.5 - lat_grid / np.pi)
    uv = np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32)
    return verts, faces, uv

def sample_texture(texture_rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample RGB texture at UV coordinates"""
    h, w, _ = texture_rgb.shape
    u = uv[:, 0] % 1.0
    v = np.clip(uv[:, 1], 0.0, 1.0)
    x = (u * (w - 1)).astype(int)
    y = (v * (h - 1)).astype(int)
    colors = texture_rgb[y, x].astype(np.float32) / 255.0
    return colors

def get_earth_texture(path="earth_texture.jpg") -> np.ndarray:
    """Download and load Earth texture"""
    if os.path.exists(path):
        return np.asarray(Image.open(path).convert("RGB"))
    url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57730/land_ocean_ice_2048.png"
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return np.asarray(Image.open(path).convert("RGB"))

def make_ring(radius: float, n: int, plane="xy") -> np.ndarray:
    """Generate ring of points"""
    t = np.linspace(0, 2 * np.pi, n, endpoint=True).astype(np.float32)
    if plane == "xy":
        pts = np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)], axis=1)
    elif plane == "xz":
        pts = np.stack([radius * np.cos(t), np.zeros_like(t), radius * np.sin(t)], axis=1)
    else:
        pts = np.stack([np.zeros_like(t), radius * np.cos(t), radius * np.sin(t)], axis=1)
    return pts.astype(np.float32)

def orthonormal_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate orthonormal basis from normal vector"""
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(a, n))) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(n, a)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    return u.astype(np.float32), v.astype(np.float32)

def make_disk_mesh(center: np.ndarray, normal: np.ndarray, radius: float, segments: int = 56) -> gl.MeshData:
    """Create disk mesh perpendicular to normal vector"""
    u, v = orthonormal_basis_from_normal(normal)
    verts = [center.astype(np.float32)]
    for k in range(segments + 1):
        t = 2.0 * np.pi * k / segments
        p = center + radius * (np.cos(t) * u + np.sin(t) * v)
        verts.append(p.astype(np.float32))
    verts = np.array(verts, dtype=np.float32)
    faces = []
    for k in range(1, segments + 1):
        faces.append([0, k, k + 1])
    faces = np.array(faces, dtype=np.int32)
    return gl.MeshData(vertexes=verts, faces=faces)