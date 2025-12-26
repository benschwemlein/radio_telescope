import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import requests
from PIL import Image

from PyQt6 import QtCore, QtWidgets

import pyqtgraph as pg
import pyqtgraph.opengl as gl


APP_TZ = ZoneInfo("America/New_York")


class FixedGLViewWidget(gl.GLViewWidget):
    # Disable zoom via mouse wheel or trackpad scroll events
    def wheelEvent(self, ev):
        ev.ignore()

    def set_fixed_distance(self, d: float):
        self.opts["distance"] = float(d)
        self.update()


def clamp_lat_lon(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    lat = max(-90.0, min(90.0, float(lat_deg)))
    lon = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    return lat, lon


def rotation_matrix_from_euler(deg_x=0.0, deg_y=0.0, deg_z=0.0) -> np.ndarray:
    rx, ry, rz = np.deg2rad([deg_x, deg_y, deg_z])

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])

    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [0,   0,  1]])

    return Rz @ Ry @ Rx


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
    return jd


def gmst_degrees(dt_utc_naive: datetime) -> float:
    jd = julian_day(dt_utc_naive)
    T = (jd - 2451545.0) / 36525.0
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T * T - (T * T * T) / 38710000.0
    return gmst % 360.0


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
    return ra_deg, dec_deg


def ra_dec_to_unit_vector_equatorial(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z], dtype=np.float32)


def equatorial_to_local_enu_matrix(lat_deg: float, lst_deg: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lst = np.deg2rad(lst_deg)

    up = np.array([np.cos(lat) * np.cos(lst), np.cos(lat) * np.sin(lst), np.sin(lat)], dtype=np.float32)

    north = np.array([-np.sin(lat) * np.cos(lst), -np.sin(lat) * np.sin(lst), np.cos(lat)], dtype=np.float32)
    north = north / np.linalg.norm(north)

    east = np.cross(north, up)
    east = east / np.linalg.norm(east)

    M = np.vstack([east, north, up]).astype(np.float32)
    return M


def unit_vector_enu_to_alt_az(v_enu: np.ndarray) -> tuple[float, float]:
    x, y, z = float(v_enu[0]), float(v_enu[1]), float(v_enu[2])
    alt = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    az = np.rad2deg(np.arctan2(x, y)) % 360.0
    return float(alt), float(az)


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
    lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    lat = np.linspace(-np.pi/2, np.pi/2, n_lat)

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

    uv = np.stack([(lon_grid / (2*np.pi)), (0.5 - lat_grid / np.pi)], axis=-1).reshape(-1, 2).astype(np.float32)
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
    t = np.linspace(0, 2*np.pi, n, endpoint=True)
    if plane == "xy":
        pts = np.stack([radius*np.cos(t), radius*np.sin(t), np.zeros_like(t)], axis=1)
    elif plane == "xz":
        pts = np.stack([radius*np.cos(t), np.zeros_like(t), radius*np.sin(t)], axis=1)
    else:
        pts = np.stack([np.zeros_like(t), radius*np.cos(t), radius*np.sin(t)], axis=1)
    return pts.astype(np.float32)


def build_milky_way_band_equatorial(radius=1.0, half_width_deg=8.0, n=900, m=16, seed=7):
    rng = np.random.default_rng(seed)
    hw = np.deg2rad(half_width_deg)

    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    base = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)

    offsets = np.linspace(-hw, hw, m)
    pts = []
    alpha = []

    for off in offsets:
        strand = base.copy()
        strand[:, 2] = np.sin(off)
        strand[:, 0] *= np.cos(off)
        strand[:, 1] *= np.cos(off)

        noise = rng.normal(0.0, 1.0, size=strand.shape)
        noise = noise / np.linalg.norm(noise, axis=1, keepdims=True)
        strand = strand + 0.01 * noise
        strand = strand / np.linalg.norm(strand, axis=1, keepdims=True)
        strand = radius * strand

        edge = abs(off) / hw if hw > 0 else 0.0
        a = (1.0 - edge) ** 1.8
        pts.append(strand)
        alpha.append(np.full(n, a, dtype=np.float32))

    P = np.concatenate(pts, axis=0).astype(np.float32)
    A = np.concatenate(alpha, axis=0).astype(np.float32)

    Rmw = rotation_matrix_from_euler(25.0, -20.0, 15.0).astype(np.float32)
    P = (Rmw @ P.T).T
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


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Celestial Sphere")

        self.radius = 1.0
        self.earth_radius = 0.30

        self.lat = 39.9612
        self.lon = -82.9988
        self.dt_local = datetime.now(APP_TZ)
        self.dt_utc = self.dt_local.astimezone(timezone.utc)

        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

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

        left.addWidget(QtWidgets.QLabel("Manual rotation"))

        self.yaw_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.pitch_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.roll_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        for s in [self.yaw_slider, self.pitch_slider, self.roll_slider]:
            s.setMinimum(-180)
            s.setMaximum(180)
            s.setValue(0)

        left.addWidget(QtWidgets.QLabel("Yaw (deg)"))
        left.addWidget(self.yaw_slider)
        left.addWidget(QtWidgets.QLabel("Pitch (deg)"))
        left.addWidget(self.pitch_slider)
        left.addWidget(QtWidgets.QLabel("Roll (deg)"))
        left.addWidget(self.roll_slider)

        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        left.addWidget(self.info)

        left.addStretch(1)

        self.view = FixedGLViewWidget()
        self.view.setBackgroundColor("w")
        self.view.set_fixed_distance(2.6)
        layout.addWidget(self.view, 1)

        self.apply_btn.clicked.connect(self.on_apply)
        self.now_btn.clicked.connect(self.on_now)

        self.yaw_slider.valueChanged.connect(self.on_rotation)
        self.pitch_slider.valueChanged.connect(self.on_rotation)
        self.roll_slider.valueChanged.connect(self.on_rotation)

    def _build_scene(self):
        verts, faces, uv = make_uv_sphere(self.earth_radius, n_lon=160, n_lat=80)
        tex = get_earth_texture("earth_texture.jpg")
        colors = sample_texture(tex, uv)
        colors_rgba = np.concatenate([colors, np.ones((colors.shape[0], 1), dtype=np.float32)], axis=1)

        earth_md = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors_rgba)
        self.earth_item = gl.GLMeshItem(meshdata=earth_md, smooth=True, drawEdges=False, shader="shaded")
        self.earth_item.setGLOptions("opaque")
        self.earth_item.setDepthValue(1000)
        self.view.addItem(self.earth_item)

        s_verts, s_faces, _ = make_uv_sphere(self.radius, n_lon=90, n_lat=45)
        sky_md = gl.MeshData(vertexes=s_verts, faces=s_faces)

        self.sky_item = gl.GLMeshItem(
            meshdata=sky_md,
            smooth=False,
            drawFaces=False,
            drawEdges=True,
            edgeColor=(0.7, 0.7, 0.7, 0.25),
        )
        self.sky_item.setGLOptions("additive")   # never blocks what is inside
        self.sky_item.setDepthValue(-1000)      # draw behind everything
        self.view.addItem(self.sky_item)

      

        hz = make_ring(self.radius, 500, "xy")
        self.horizon_item = gl.GLLinePlotItem(pos=hz, width=2.0, antialias=True)
        self.view.addItem(self.horizon_item)

        eq = make_ring(self.radius, 600, "xy")
        self.eq_item = gl.GLLinePlotItem(pos=eq, width=1.5, antialias=True)
        self.view.addItem(self.eq_item)

        self.axis_item = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, self.radius]], dtype=np.float32),
            width=2.0,
            antialias=True
        )
        self.view.addItem(self.axis_item)

        mw_pts, mw_a = build_milky_way_band_equatorial(self.radius, 8.0, n=900, m=16, seed=7)
        self.mw_pts_eq = mw_pts
        cols = np.ones((mw_pts.shape[0], 4), dtype=np.float32)
        cols[:, 3] = 0.10 + 0.70 * np.clip(mw_a, 0.0, 1.0)
        self.mw_item = gl.GLScatterPlotItem(pos=mw_pts, size=2.0, color=cols, pxMode=True)
        self.view.addItem(self.mw_item)

        sun_ang_radius_deg = 0.265
        disk_radius = self.radius * np.sin(np.deg2rad(sun_ang_radius_deg))
        md = make_disk_mesh(
            center=np.array([self.radius, 0.0, 0.0], dtype=np.float32),
            normal=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            radius=disk_radius,
            segments=56
        )
        self.sun_item = gl.GLMeshItem(meshdata=md, smooth=True, drawEdges=False, shader="shaded")
        self.sun_item.setColor((1.0, 0.75, 0.1, 1.0))
        self.view.addItem(self.sun_item)

    def on_now(self):
        self.dt_local = datetime.now(APP_TZ)
        self.time_edit.setText(self.dt_local.strftime("%Y-%m-%d %H:%M:%S"))
        self.on_apply()

    def on_rotation(self):
        self.yaw = float(self.yaw_slider.value())
        self.pitch = float(self.pitch_slider.value())
        self.roll = float(self.roll_slider.value())
        self.update_scene()

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
        gmst = gmst_degrees(self.dt_utc.replace(tzinfo=None))
        lst = (gmst + self.lon) % 360.0
        M = equatorial_to_local_enu_matrix(self.lat, lst)
        Ruser = rotation_matrix_from_euler(self.roll, self.pitch, self.yaw).astype(np.float32)
        X = (Ruser @ M).astype(np.float32)

        eq = make_ring(self.radius, 600, "xy")
        eq_local = (X @ eq.T).T
        self.eq_item.setData(pos=eq_local)

        ncp_eq = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        ncp_local = X @ ncp_eq
        axis = np.array([[0, 0, 0], self.radius * ncp_local], dtype=np.float32)
        self.axis_item.setData(pos=axis)

        mw_local = (X @ self.mw_pts_eq.T).T
        self.mw_item.setData(pos=mw_local)

        sun_ra, sun_dec = sun_ra_dec_degrees(self.dt_utc.replace(tzinfo=None))
        sun_eq = ra_dec_to_unit_vector_equatorial(sun_ra, sun_dec)
        sun_local = X @ sun_eq
        sun_pos = (self.radius * sun_local).astype(np.float32)

        sun_ang_radius_deg = 0.265
        disk_radius = self.radius * np.sin(np.deg2rad(sun_ang_radius_deg))
        md = make_disk_mesh(
            center=sun_pos,
            normal=sun_local.astype(np.float32),
            radius=disk_radius,
            segments=56
        )
        self.sun_item.setMeshData(meshdata=md)

        alt, az = unit_vector_enu_to_alt_az(sun_local)
        self.info.setText(
            f"Lat {self.lat:.4f}  Lon {self.lon:.4f}\n"
            f"Local {self.dt_local.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"UTC {self.dt_utc.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Sun alt {alt:.1f} deg  az {az:.1f} deg"
        )


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1400, 800)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
