import os
import sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import requests
from PIL import Image

from PyQt6 import QtWidgets
import pyqtgraph.opengl as gl


APP_TZ = ZoneInfo("America/New_York")

# Colors (r,g,b,a) in 0..1
ORANGE = (1.0, 0.55, 0.0, 1.0)
WHITE = (0.92, 0.92, 0.92, 1.0)
GOLD = (1.0, 0.84, 0.0, 1.0)

DEBUG_VERBOSE = True

# If the Earth appears to rotate the wrong way relative to time, flip this to -1.0
EARTH_ROT_SIGN = 1.0

# Sun angular radius is about 0.265 degrees
SUN_ANG_RADIUS_DEG = 0.265

# If True, sky items draw on top of Earth (visualization mode)
SKY_ALWAYS_ON_TOP = True


class FixedGLViewWidget(gl.GLViewWidget):
    # Mouse rotates camera only
    # Disable zoom via wheel or trackpad scroll
    def wheelEvent(self, ev):
        ev.ignore()

    def set_fixed_distance(self, d: float):
        self.opts["distance"] = float(d)
        self.update()


def clamp_lat_lon(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    lat = max(-90.0, min(90.0, float(lat_deg)))
    lon = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    return lat, lon


def julian_day(dt_utc_naive: datetime) -> float:
    y = dt_utc_naive.year
    m = dt_utc_naive.month
    d = dt_utc_naive.day + (dt_utc_naive.hour + (dt_utc_naive.minute + dt_utc_naive.second / 60.0) / 60.0) / 24.0

    if m <= 2:
        y -= 1
        m += 12

    A = int(y / 100)
    B = 2 - A + int(A / 4)

    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return float(jd)


def gmst_degrees(dt_utc_naive: datetime) -> float:
    jd = julian_day(dt_utc_naive)
    T = (jd - 2451545.0) / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T * T
        - (T * T * T) / 38710000.0
    )
    return float(gmst % 360.0)


def rotz_deg(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [ca, -sa, 0.0],
        [sa,  ca, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def sun_ra_dec_degrees(dt_utc_naive: datetime) -> tuple[float, float]:
    jd = julian_day(dt_utc_naive)
    n = jd - 2451545.0

    L = (280.460 + 0.9856474 * n) % 360.0
    g = np.deg2rad((357.528 + 0.9856003 * n) % 360.0)

    lam = np.deg2rad((L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)) % 360.0)
    eps = np.deg2rad(23.439 - 0.0000004 * n)

    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    alpha = np.arctan2(np.cos(eps) * sin_lam, cos_lam)
    delta = np.arcsin(np.sin(eps) * sin_lam)

    ra_deg = (np.rad2deg(alpha) % 360.0)
    dec_deg = np.rad2deg(delta)
    return float(ra_deg), float(dec_deg)


def ra_dec_to_unit_vector_equatorial(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z], dtype=np.float32)


def equatorial_to_local_enu_matrix(lat_deg: float, lst_deg: float) -> np.ndarray:
    """
    v_enu = M @ v_eq
    Rows are east, north, up expressed in equatorial XYZ basis.
    """
    lat = np.deg2rad(lat_deg)
    lst = np.deg2rad(lst_deg)

    sl, cl = np.sin(lat), np.cos(lat)
    st, ct = np.sin(lst), np.cos(lst)

    east = np.array([-st, ct, 0.0], dtype=np.float32)
    north = np.array([-sl * ct, -sl * st, cl], dtype=np.float32)
    up = np.array([cl * ct, cl * st, sl], dtype=np.float32)

    M = np.vstack([east, north, up]).astype(np.float32)
    return M


def unit_vector_enu_to_alt_az(v_enu: np.ndarray) -> tuple[float, float]:
    x, y, z = float(v_enu[0]), float(v_enu[1]), float(v_enu[2])
    alt = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    az = np.rad2deg(np.arctan2(x, y)) % 360.0
    return float(alt), float(az)


def alt_az_from_ra_dec(lat_deg: float, lst_deg: float, ra_deg: float, dec_deg: float) -> tuple[float, float]:
    lat = np.deg2rad(lat_deg)
    ha = np.deg2rad((lst_deg - ra_deg) % 360.0)
    dec = np.deg2rad(dec_deg)

    sin_alt = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha)
    alt = np.arcsin(np.clip(sin_alt, -1.0, 1.0))

    cos_az = (np.sin(dec) - np.sin(alt) * np.sin(lat)) / (np.cos(alt) * np.cos(lat) + 1e-12)
    cos_az = np.clip(cos_az, -1.0, 1.0)
    az = np.arccos(cos_az)

    if np.sin(ha) > 0:
        az = 2 * np.pi - az

    return float(np.rad2deg(alt)), float(np.rad2deg(az) % 360.0)


def get_earth_texture(path="earth_texture.jpg") -> np.ndarray:
    if os.path.exists(path):
        return np.asarray(Image.open(path).convert("RGB"))

    url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57730/land_ocean_ice_2048.png"
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return np.asarray(Image.open(path).convert("RGB"))


def make_uv_sphere(radius: float, n_lon: int, n_lat: int):
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
    h, w, _ = texture_rgb.shape
    u = uv[:, 0] % 1.0
    v = np.clip(uv[:, 1], 0.0, 1.0)

    x = (u * (w - 1)).astype(int)
    y = (v * (h - 1)).astype(int)

    colors = texture_rgb[y, x].astype(np.float32) / 255.0
    return colors


def make_ring(radius: float, n: int, plane="xy") -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=True).astype(np.float32)
    if plane == "xy":
        pts = np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)], axis=1)
    elif plane == "xz":
        pts = np.stack([radius * np.cos(t), np.zeros_like(t), radius * np.sin(t)], axis=1)
    else:
        pts = np.stack([np.zeros_like(t), radius * np.cos(t), radius * np.sin(t)], axis=1)
    return pts.astype(np.float32)


def eq_to_gal_matrix_j2000() -> np.ndarray:
    """
    IAU J2000 Equatorial to Galactic rotation.
    This is the matrix with the widely published coefficients.
    """
    return np.array([
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [ 0.4941094279, -0.4448296300,  0.7469822445],
        [-0.8676661490, -0.1980763734,  0.4559837762],
    ], dtype=np.float32)


def gal_to_eq_matrix_j2000() -> np.ndarray:
    """
    Galactic to Equatorial is the transpose of the orthonormal rotation.
    """
    return eq_to_gal_matrix_j2000().T.astype(np.float32)


def build_milky_way_band_equatorial(radius=1.0, half_width_deg=10.0, n=1600, m=33, seed=7):
    rng = np.random.default_rng(seed)

    # Correct direction: Galactic -> Equatorial for drawing the galactic plane on the equatorial sky
    R_g2e = gal_to_eq_matrix_j2000()

    l = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False).astype(np.float32)
    hw = np.deg2rad(half_width_deg).astype(np.float32)
    b_vals = np.linspace(-hw, hw, m).astype(np.float32)

    pts = []
    alpha = []

    for b in b_vals:
        cb = np.cos(b)
        sb = np.sin(b)

        xg = cb * np.cos(l)
        yg = cb * np.sin(l)
        zg = np.full_like(l, sb)
        g = np.stack([xg, yg, zg], axis=1).astype(np.float32)

        noise = rng.normal(0.0, 1.0, size=g.shape).astype(np.float32)
        noise = noise / np.linalg.norm(noise, axis=1, keepdims=True)
        g = g + 0.008 * noise
        g = g / np.linalg.norm(g, axis=1, keepdims=True)

        e = (R_g2e @ g.T).T.astype(np.float32)
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        e = radius * e

        edge = abs(float(b) / float(hw)) if float(hw) > 0.0 else 0.0
        a = (1.0 - edge) ** 1.8

        pts.append(e)
        alpha.append(np.full(n, a, dtype=np.float32))

    P = np.concatenate(pts, axis=0).astype(np.float32)
    A = np.concatenate(alpha, axis=0).astype(np.float32)
    return P, A


def orthonormal_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def latlon_to_ecef(lat_deg: float, lon_deg: float, r: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z], dtype=np.float32)


def latlon_to_unit_ecef(lat_deg: float, lon_deg: float) -> np.ndarray:
    v = latlon_to_ecef(lat_deg, lon_deg, 1.0)
    return (v / np.linalg.norm(v)).astype(np.float32)


def ecef_basis_at(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = np.deg2rad(lon_deg)
    up = latlon_to_unit_ecef(lat_deg, lon_deg)

    east = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=np.float32)
    east = east / np.linalg.norm(east)

    north = np.cross(up, east).astype(np.float32)
    north = north / np.linalg.norm(north)

    return east, north, up


def make_horizon_ring_ecef(radius: float, lat_deg: float, lon_deg: float, n: int = 600) -> np.ndarray:
    east, north, _up = ecef_basis_at(lat_deg, lon_deg)
    t = np.linspace(0, 2 * np.pi, n, endpoint=True).astype(np.float32)
    pts = radius * (np.cos(t)[:, None] * east[None, :] + np.sin(t)[:, None] * north[None, :])
    return pts.astype(np.float32)


def vstr(v: np.ndarray) -> str:
    v = np.asarray(v).astype(float).ravel()
    return f"[{v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f}]"


def mstr(M: np.ndarray) -> str:
    M = np.asarray(M).astype(float)
    return (
        "\n"
        f"  [{M[0,0]: .6f} {M[0,1]: .6f} {M[0,2]: .6f}]\n"
        f"  [{M[1,0]: .6f} {M[1,1]: .6f} {M[1,2]: .6f}]\n"
        f"  [{M[2,0]: .6f} {M[2,1]: .6f} {M[2,2]: .6f}]"
    )


def det3(M: np.ndarray) -> float:
    return float(np.linalg.det(np.asarray(M).astype(float)))


def row_norms(M: np.ndarray) -> list[float]:
    M = np.asarray(M).astype(float)
    return [float(np.linalg.norm(M[i, :])) for i in range(3)]


def dot_rows(M: np.ndarray) -> tuple[float, float, float]:
    M = np.asarray(M).astype(float)
    return (
        float(np.dot(M[0, :], M[1, :])),
        float(np.dot(M[0, :], M[2, :])),
        float(np.dot(M[1, :], M[2, :])),
    )


def angular_sep_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    return float(np.rad2deg(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def sun_galactic_l_b_deg(sun_eq_unit: np.ndarray) -> tuple[float, float]:
    E2G = eq_to_gal_matrix_j2000()
    g = (E2G @ sun_eq_unit.reshape(3, 1)).ravel()
    g = g / (np.linalg.norm(g) + 1e-12)
    x, y, z = float(g[0]), float(g[1]), float(g[2])
    b = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    l = np.rad2deg(np.arctan2(y, x)) % 360.0
    return float(l), float(b)


def gc_unit_eq() -> np.ndarray:
    # J2000 galactic center approx: RA 266.4051 deg, Dec -28.936175 deg
    return ra_dec_to_unit_vector_equatorial(266.4051, -28.936175)


def find_max_alt_over_24h(lat_deg: float, lon_deg: float, start_local: datetime, step_minutes: int = 4):
    v_gc = gc_unit_eq()
    best_alt = None
    best_az = None
    best_time = None

    for k in range(int(24 * 60 / step_minutes) + 1):
        dt_local = start_local + timedelta(minutes=k * step_minutes)
        dt_utc = dt_local.astimezone(timezone.utc)
        dt_utc_naive = dt_utc.replace(tzinfo=None)

        gmst = gmst_degrees(dt_utc_naive)
        lst = (gmst + lon_deg) % 360.0

        M = equatorial_to_local_enu_matrix(lat_deg, lst).astype(np.float32)
        v_loc = (M @ v_gc.reshape(3, 1)).ravel().astype(np.float32)
        v_loc = v_loc / (np.linalg.norm(v_loc) + 1e-12)
        alt, az = unit_vector_enu_to_alt_az(v_loc)

        if best_alt is None or alt > best_alt:
            best_alt = alt
            best_az = az
            best_time = dt_local

    return float(best_alt), float(best_az), best_time


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Celestial Sphere")

        self.radius = 1.0
        self.earth_radius = 0.36

        self.lat = 39.9612
        self.lon = -82.9988
        self.dt_local = datetime.now(APP_TZ)
        self.dt_utc = self.dt_local.astimezone(timezone.utc)

        self._build_ui()
        self._build_scene()
        self.update_scene()

    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, 0)

        self.lat_edit = QtWidgets.QLineEdit(str(self.lat))
        self.lon_edit = QtWidgets.QLineEdit(str(self.lon))
        self.time_edit = QtWidgets.QLineEdit(self.dt_local.strftime("%Y-%m-%d %H:%M:%S"))

        left.addWidget(QtWidgets.QLabel("Latitude"))
        left.addWidget(self.lat_edit)
        left.addWidget(QtWidgets.QLabel("Longitude"))
        left.addWidget(self.lon_edit)
        left.addWidget(QtWidgets.QLabel("Date and time (America/New_York)"))
        left.addWidget(self.time_edit)

        btn_row = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.now_btn = QtWidgets.QPushButton("Now")
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.now_btn)
        left.addLayout(btn_row)

        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        left.addWidget(self.info)

        left.addStretch(1)

        self.view = FixedGLViewWidget()
        self.view.setBackgroundColor((10, 10, 14))
        self.view.set_fixed_distance(2.6)
        layout.addWidget(self.view, 1)

        self.apply_btn.clicked.connect(self.on_apply)
        self.now_btn.clicked.connect(self.on_now)

    def _sky_options(self, item, depth_value: int):
        if SKY_ALWAYS_ON_TOP:
            item.setGLOptions("additive")
            item.setDepthValue(depth_value)
        else:
            item.setGLOptions("translucent")
            item.setDepthValue(0)

    def _build_scene(self):
        # Celestial sphere shell
        s_verts, s_faces, _ = make_uv_sphere(self.radius, n_lon=120, n_lat=60)
        sky_md = gl.MeshData(vertexes=s_verts, faces=s_faces)
        self.sky_item = gl.GLMeshItem(meshdata=sky_md, smooth=True, drawFaces=True, drawEdges=False, shader="shaded")
        self.sky_item.setColor((0.50, 0.55, 0.65, 0.18))
        self._sky_options(self.sky_item, -10000)
        self.view.addItem(self.sky_item)

        # Earth
        verts, faces, uv = make_uv_sphere(self.earth_radius, n_lon=180, n_lat=90)
        tex = get_earth_texture("earth_texture.jpg")
        colors = sample_texture(tex, uv)
        colors_rgba = np.concatenate([colors, np.ones((colors.shape[0], 1), dtype=np.float32)], axis=1)
        earth_md = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors_rgba)
        self.earth_item = gl.GLMeshItem(meshdata=earth_md, smooth=True, drawEdges=False, shader="shaded")
        self.earth_item.setGLOptions("opaque")
        self.earth_item.setDepthValue(10000)
        self.view.addItem(self.earth_item)

        # Location marker
        marker_r = self.earth_radius * 0.05
        m_verts, m_faces, _ = make_uv_sphere(marker_r, n_lon=36, n_lat=18)
        m_md = gl.MeshData(vertexes=m_verts, faces=m_faces)
        self.loc_marker = gl.GLMeshItem(meshdata=m_md, smooth=True, drawEdges=False, shader="shaded")
        self.loc_marker.setColor(ORANGE)
        self.loc_marker.setGLOptions("translucent")
        self.loc_marker.setDepthValue(20000)
        self.view.addItem(self.loc_marker)

        # Horizon ring (Earth fixed, rotated with Earth)
        hz0 = make_horizon_ring_ecef(self.radius, self.lat, self.lon, n=600)
        self.horizon_item = gl.GLLinePlotItem(pos=hz0, width=2.5, antialias=True, color=ORANGE)
        self.horizon_item.setGLOptions("translucent")
        self.horizon_item.setDepthValue(25000)
        self.view.addItem(self.horizon_item)

        # Celestial equator (will be transformed to local ENU)
        eq = make_ring(self.radius, 600, "xy")
        self.eq_item = gl.GLLinePlotItem(pos=eq, width=1.4, antialias=True, color=WHITE)
        self._sky_options(self.eq_item, 999998)
        self.view.addItem(self.eq_item)

        # Milky Way band in equatorial coordinates (corrected via g2e transpose)
        mw_pts, mw_a = build_milky_way_band_equatorial(self.radius, half_width_deg=10.0, n=1600, m=33, seed=7)
        self.mw_pts_eq = mw_pts
        self.mw_cols = np.ones((mw_pts.shape[0], 4), dtype=np.float32)
        self.mw_cols[:, 3] = 0.08 + 0.75 * np.clip(mw_a, 0.0, 1.0)
        self.mw_item = gl.GLScatterPlotItem(pos=mw_pts, size=2.0, color=self.mw_cols, pxMode=True)
        self._sky_options(self.mw_item, 999999)
        self.view.addItem(self.mw_item)

        # Sun disk and dot (will be transformed to local ENU)
        disk_radius = self.radius * np.sin(np.deg2rad(SUN_ANG_RADIUS_DEG))
        md = make_disk_mesh(
            center=np.array([self.radius, 0.0, 0.0], dtype=np.float32),
            normal=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            radius=disk_radius,
            segments=56
        )
        self.sun_item = gl.GLMeshItem(meshdata=md, smooth=True, drawEdges=False, shader="shaded")
        self.sun_item.setColor((1.0, 0.78, 0.12, 1.0))
        self.sun_item.setGLOptions("translucent")
        self.sun_item.setDepthValue(30000)
        self.view.addItem(self.sun_item)

        self.sun_dot = gl.GLScatterPlotItem(
            pos=np.array([[self.radius, 0.0, 0.0]], dtype=np.float32),
            size=12.0,
            color=np.array([[1.0, 0.78, 0.12, 1.0]], dtype=np.float32),
            pxMode=True
        )
        self.sun_dot.setGLOptions("translucent")
        self.sun_dot.setDepthValue(30000)
        self.view.addItem(self.sun_dot)

        # Earth rotation axis line
        axis_pts = np.array([[0.0, 0.0, -self.radius],
                             [0.0, 0.0,  self.radius]], dtype=np.float32)
        self.earth_axis_item = gl.GLLinePlotItem(pos=axis_pts, width=4.5, antialias=True, color=GOLD)
        self.earth_axis_item.setGLOptions("translucent")
        self.earth_axis_item.setDepthValue(20000)
        self.view.addItem(self.earth_axis_item)

        # Optional galactic center dot on the sky (for sanity checking)
        self.gc_dot = gl.GLScatterPlotItem(
            pos=np.array([[self.radius, 0.0, 0.0]], dtype=np.float32),
            size=9.0,
            color=np.array([[0.65, 0.80, 1.0, 1.0]], dtype=np.float32),
            pxMode=True
        )
        self._sky_options(self.gc_dot, 999997)
        self.view.addItem(self.gc_dot)

    def on_now(self):
        self.dt_local = datetime.now(APP_TZ)
        self.time_edit.setText(self.dt_local.strftime("%Y-%m-%d %H:%M:%S"))
        self.on_apply()

    def on_apply(self):
        try:
            lat = float(self.lat_edit.text().strip())
            lon = float(self.lon_edit.text().strip())
            lat, lon = clamp_lat_lon(lat, lon)
            self.lat = lat
            self.lon = lon
        except Exception:
            pass

        try:
            dt = datetime.strptime(self.time_edit.text().strip(), "%Y-%m-%d %H:%M:%S")
            self.dt_local = dt.replace(tzinfo=APP_TZ)
        except Exception:
            self.dt_local = datetime.now(APP_TZ)

        self.dt_utc = self.dt_local.astimezone(timezone.utc)
        self.update_scene()

    def update_scene(self):
        dt_utc_naive = self.dt_utc.replace(tzinfo=None)

        jd = julian_day(dt_utc_naive)
        gmst = gmst_degrees(dt_utc_naive)
        lst = (gmst + self.lon) % 360.0

        # Earth rotates in inertial frame by GMST
        Rearth = rotz_deg(EARTH_ROT_SIGN * gmst)

        # Earth fixed geometry, rotate by Rearth into view
        p_ecef = latlon_to_ecef(self.lat, self.lon, self.earth_radius)
        p_view = (Rearth @ p_ecef).astype(np.float32)

        self.loc_marker.resetTransform()
        self.loc_marker.translate(float(p_view[0]), float(p_view[1]), float(p_view[2]))

        horizon_ecef = make_horizon_ring_ecef(self.radius, self.lat, self.lon, n=600)
        horizon_view = (Rearth @ horizon_ecef.T).T.astype(np.float32)
        self.horizon_item.setData(pos=horizon_view)

        axis_ecef = np.array([[0.0, 0.0, -self.radius],
                              [0.0, 0.0,  self.radius]], dtype=np.float32)
        axis_view = (Rearth @ axis_ecef.T).T.astype(np.float32)
        self.earth_axis_item.setData(pos=axis_view)

        # Sky transform: equatorial to local ENU using LST
        M = equatorial_to_local_enu_matrix(self.lat, lst).astype(np.float32)

        # Celestial equator ring into local ENU
        eq = make_ring(self.radius, 600, "xy")
        eq_local = (M @ eq.T).T.astype(np.float32)
        self.eq_item.setData(pos=eq_local)

        # Milky Way points are in equatorial XYZ, convert to local ENU
        mw_local = (M @ self.mw_pts_eq.T).T.astype(np.float32)
        self.mw_item.setData(pos=mw_local, color=self.mw_cols)

        # Sun in equatorial
        sun_ra, sun_dec = sun_ra_dec_degrees(dt_utc_naive)
        sun_eq = ra_dec_to_unit_vector_equatorial(sun_ra, sun_dec)

        # Sun to local ENU
        sun_local = (M @ sun_eq.reshape(3, 1)).ravel().astype(np.float32)
        sun_local = sun_local / (np.linalg.norm(sun_local) + 1e-12)

        sun_pos = (self.radius * sun_local).astype(np.float32)
        self.sun_dot.setData(pos=sun_pos.reshape(1, 3))

        disk_radius = self.radius * np.sin(np.deg2rad(SUN_ANG_RADIUS_DEG))
        self.sun_item.setMeshData(meshdata=make_disk_mesh(
            center=sun_pos,
            normal=sun_local.astype(np.float32),
            radius=disk_radius,
            segments=56
        ))

        alt_vec, az_vec = unit_vector_enu_to_alt_az(sun_local)
        alt_formula, az_formula = alt_az_from_ra_dec(self.lat, lst, sun_ra, sun_dec)

        l_gal, b_gal = sun_galactic_l_b_deg(sun_eq)

        self.info.setText(
            f"Lat {self.lat:.4f}  Lon {self.lon:.4f}\n"
            f"Local {self.dt_local.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"UTC {self.dt_utc.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Sun alt {alt_vec:.2f} deg  az {az_vec:.2f} deg"
        )

        # Galactic center dot for sanity checking
        gc_eq = gc_unit_eq()
        gc_local = (M @ gc_eq.reshape(3, 1)).ravel().astype(np.float32)
        gc_local = gc_local / (np.linalg.norm(gc_local) + 1e-12)
        gc_pos = (self.radius * gc_local).astype(np.float32)
        self.gc_dot.setData(pos=gc_pos.reshape(1, 3))

        if DEBUG_VERBOSE:
            ha = (lst - sun_ra) % 360.0
            scp = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            ncp = np.array([0.0, 0.0,  1.0], dtype=np.float32)

            gc_sep_scp = angular_sep_deg(gc_eq, scp)
            gc_sep_ncp = angular_sep_deg(gc_eq, ncp)
            gc_alt, gc_az = unit_vector_enu_to_alt_az(gc_local)

            # Strong internal consistency check: closest Milky Way point to GC should be small separation
            mw_dirs_eq = self.mw_pts_eq / (self.radius + 1e-12)
            dots = np.clip(mw_dirs_eq @ (gc_eq / (np.linalg.norm(gc_eq) + 1e-12)), -1.0, 1.0)
            min_sep = float(np.rad2deg(np.arccos(float(np.max(dots)))))

            best_alt_24h, best_az_24h, best_time_24h = find_max_alt_over_24h(self.lat, self.lon, self.dt_local, step_minutes=4)

            print("\nDEBUG")
            print(f"Local time {self.dt_local.strftime('%Y-%m-%d %H:%M:%S')} {APP_TZ.key}")
            print(f"UTC time   {self.dt_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"Lat deg   {self.lat: .6f}")
            print(f"Lon deg   {self.lon: .6f}")
            print(f"JD        {jd: .12f}")
            print(f"GMST deg  {gmst: .6f}")
            print(f"LST deg   {lst: .6f}")
            print(f"EARTH_ROT_SIGN {EARTH_ROT_SIGN: .1f}")
            print(f"Rearth = Rz({EARTH_ROT_SIGN * gmst: .6f})")

            print(f"\nSun RA deg  {sun_ra: .6f}")
            print(f"Sun Dec deg {sun_dec: .6f}")
            print(f"Hour angle deg {ha: .6f}")

            print(f"\nsun_eq unit {vstr(sun_eq)}")
            print(f"sun_local unit {vstr(sun_local)}")
            print(f"norm sun_eq    {np.linalg.norm(sun_eq): .9f}")
            print(f"norm sun_local {np.linalg.norm(sun_local): .9f}")

            print("\nM eq to ENU", mstr(M))
            print(f"det M {det3(M): .9f}")
            rn = row_norms(M)
            dr = dot_rows(M)
            print(f"row norms M {rn}")
            print(f"dot rows M {dr[0]: .9e} {dr[1]: .9e} {dr[2]: .9e}")

            print(f"\nAlt deg from vector  {alt_vec: .6f}")
            print(f"Az deg from vector   {az_vec: .6f}")
            print(f"Alt deg from formula {alt_formula: .6f}")
            print(f"Az deg from formula  {az_formula: .6f}")

            print(f"\nSun galactic l deg   {l_gal: .6f}")
            print(f"Sun galactic b deg   {b_gal: .6f}")
            print(f"Sun absolute off plane deg {abs(b_gal): .6f}")

            print(f"\nObserver ECEF model {vstr(p_ecef)}")
            print(f"Observer view  model {vstr(p_view)}")

            print("\nGalactic Center sanity checks")
            print(f"  GC eq unit {vstr(gc_eq)}")
            print(f"  GC local alt {gc_alt: .6f} deg  az {gc_az: .6f} deg")
            print(f"  GC angle to south celestial pole {gc_sep_scp: .6f} deg (expected about 61.064)")
            print(f"  GC angle to north celestial pole {gc_sep_ncp: .6f} deg (expected about 118.936)")
            print(f"  Min angular sep between GC and Milky Way points {min_sep: .6f} deg (should be small)")

            print("\nGC visibility over next 24h")
            print(f"  GC max alt {best_alt_24h: .6f} deg at {best_time_24h.strftime('%Y-%m-%d %H:%M:%S')} local  az {best_az_24h: .6f} deg")

            print("END DEBUG")

        # Mouse movement does not trigger any recompute
        # Only Apply, Now, and startup call update_scene


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1400, 800)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
