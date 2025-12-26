import os
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from PIL import Image


APP_TZ = ZoneInfo("America/New_York")


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


def julian_day(dt_utc: datetime) -> float:
    y = dt_utc.year
    m = dt_utc.month
    d = dt_utc.day + (dt_utc.hour + (dt_utc.minute + dt_utc.second / 60.0) / 60.0) / 24.0

    if m <= 2:
        y -= 1
        m += 12

    A = int(y / 100)
    B = 2 - A + int(A / 4)

    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return jd


def gmst_degrees(dt_utc: datetime) -> float:
    jd = julian_day(dt_utc)
    T = (jd - 2451545.0) / 36525.0
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T * T - (T * T * T) / 38710000.0
    return gmst % 360.0


def sun_ra_dec_degrees(dt_utc: datetime) -> tuple[float, float]:
    jd = julian_day(dt_utc)
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
    return np.array([x, y, z], dtype=float)


def equatorial_to_local_enu_matrix(lat_deg: float, lst_deg: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lst = np.deg2rad(lst_deg)

    up = np.array([np.cos(lat) * np.cos(lst), np.cos(lat) * np.sin(lst), np.sin(lat)])

    north = np.array([-np.sin(lat) * np.cos(lst), -np.sin(lat) * np.sin(lst), np.cos(lat)])
    north = north / np.linalg.norm(north)

    east = np.cross(north, up)
    east = east / np.linalg.norm(east)

    M = np.vstack([east, north, up])
    return M


def unit_vector_enu_to_alt_az(v_enu: np.ndarray) -> tuple[float, float]:
    x, y, z = v_enu
    alt = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    az = np.rad2deg(np.arctan2(x, y)) % 360.0
    return alt, az


def get_earth_texture(path="earth_texture.jpg") -> np.ndarray:
    if os.path.exists(path):
        return np.asarray(Image.open(path).convert("RGB"))

    try:
        import requests
        url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57730/land_ocean_ice_2048.png"
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception as e:
        raise RuntimeError(
            "Could not download Earth texture. Install requests or manually place an "
            "equirectangular Earth texture at earth_texture.jpg. "
            f"Original error: {e}"
        )


def draw_textured_sphere(ax, radius: float, texture_rgb: np.ndarray, center=(0.0, 0.0, 0.0), stride=3):
    cx, cy, cz = center
    tex = texture_rgb.astype(np.float32) / 255.0
    h, w, _ = tex.shape

    u = np.linspace(0, 2 * np.pi, max(60, (2 * w) // stride), endpoint=False)
    v = np.linspace(0, np.pi, max(40, h // stride))

    uu, vv = np.meshgrid(u, v)

    x = radius * np.cos(uu) * np.sin(vv) + cx
    y = radius * np.sin(uu) * np.sin(vv) + cy
    z = radius * np.cos(vv) + cz

    tex_x = (uu / (2 * np.pi) * (w - 1)).astype(int)
    tex_y = ((vv / np.pi) * (h - 1)).astype(int)
    facecolors = tex[tex_y, tex_x]

    ax.plot_surface(
        x, y, z,
        rstride=1, cstride=1,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=False
    )


def make_sphere_mesh(radius=1.0, nu=120, nv=80):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def build_milky_way_band_equatorial(
    radius=1.0,
    half_width_deg=8.0,
    point_count=2600,
    cross_count=70,
    turbulence=0.35,
    base_rotation_euler=(25.0, -20.0, 15.0),
    seed=7
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    t = np.linspace(0, 2 * np.pi, point_count, endpoint=False)
    base = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)

    hw = np.deg2rad(half_width_deg)
    offsets = np.linspace(-hw, hw, cross_count)

    all_pts = []
    intensities = []

    for off in offsets:
        strand = base.copy()
        strand[:, 2] = np.sin(off)
        strand[:, 0] *= np.cos(off)
        strand[:, 1] *= np.cos(off)

        noise = rng.normal(0.0, 1.0, size=strand.shape)
        noise = noise / np.linalg.norm(noise, axis=1, keepdims=True)
        strand = strand + (turbulence * 0.02) * noise

        strand = strand / np.linalg.norm(strand, axis=1, keepdims=True)
        strand = radius * strand

        edge = abs(off) / hw if hw > 0 else 0.0
        band_strength = (1.0 - edge) ** 1.8
        clump = 0.5 + 0.5 * np.sin(6 * t + rng.uniform(0, 2 * np.pi))
        intensity = band_strength * (0.55 + 0.45 * clump)

        all_pts.append(strand)
        intensities.append(intensity)

    P = np.concatenate(all_pts, axis=0)
    I = np.concatenate(intensities, axis=0)

    Rmw = rotation_matrix_from_euler(*base_rotation_euler)
    P = (Rmw @ P.T).T
    return P, I


class CelestialSphereApp:
    def __init__(self, root):
        self.root = root
        root.title("Celestial Sphere with Earth, Horizon, Equator, Sun, Milky Way")

        self.radius = 1.0
        self.sphere_alpha = 0.10

        self.earth_radius = 0.18
        self.earth_texture_path = "earth_texture.jpg"
        self.earth_texture = None

        self.lat_str = tk.StringVar(value="39.9612")
        self.lon_str = tk.StringVar(value="-82.9988")
        self.time_str = tk.StringVar(value=datetime.now(APP_TZ).strftime("%Y-%m-%d %H:%M:%S"))

        self.yaw_deg = tk.DoubleVar(value=0.0)
        self.pitch_deg = tk.DoubleVar(value=0.0)
        self.roll_deg = tk.DoubleVar(value=0.0)

        self.is_playing = False
        self.play_after_id = None
        self.last_tick = None
        self.play_deg_per_second = 10.0

        self._build_ui()
        self.sphere_x, self.sphere_y, self.sphere_z = make_sphere_mesh(self.radius, 120, 80)
        self.mw_eq_points, self.mw_intensity = build_milky_way_band_equatorial(
            radius=self.radius,
            half_width_deg=8.0,
            point_count=2600,
            cross_count=70,
            turbulence=0.35,
            base_rotation_euler=(25.0, -20.0, 15.0),
            seed=7
        )

        self._draw()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        view = ttk.Frame(main)
        view.grid(row=0, column=1, sticky="nsew")
        view.rowconfigure(0, weight=1)
        view.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Latitude").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.lat_str, width=18).grid(row=1, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(controls, text="Longitude").grid(row=2, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.lon_str, width=18).grid(row=3, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(controls, text="Date and time (America/New_York)").grid(row=4, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.time_str, width=18).grid(row=5, column=0, sticky="ew", pady=(2, 8))

        row = ttk.Frame(controls)
        row.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(row, text="Apply", command=self.apply_all).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(row, text="Now", command=self.set_now).grid(row=0, column=1)

        row2 = ttk.Frame(controls)
        row2.grid(row=7, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(row2, text="Play", command=self.play).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(row2, text="Pause", command=self.pause).grid(row=0, column=1)

        ttk.Label(controls, text="Manual rotation").grid(row=8, column=0, sticky="w", pady=(0, 4))
        self._add_slider(controls, "Yaw", self.yaw_deg, 9)
        self._add_slider(controls, "Pitch", self.pitch_deg, 10)
        self._add_slider(controls, "Roll", self.roll_deg, 11)

        ttk.Button(controls, text="Reset", command=self.reset).grid(row=12, column=0, sticky="ew", pady=(10, 0))

        controls.columnconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(9, 7), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_box_aspect((1, 1, 1))

        self.canvas = FigureCanvasTkAgg(self.fig, master=view)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _add_slider(self, parent, label, var, row):
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, sticky="ew", pady=3)
        ttk.Label(f, text=f"{label} (deg)").grid(row=0, column=0, sticky="w")
        s = ttk.Scale(f, from_=-180, to=180, variable=var, command=lambda _=None: self._draw())
        s.grid(row=1, column=0, sticky="ew")
        f.columnconfigure(0, weight=1)

    def set_now(self):
        self.time_str.set(datetime.now(APP_TZ).strftime("%Y-%m-%d %H:%M:%S"))
        self.apply_all()

    def reset(self):
        self.pause()
        self.yaw_deg.set(0.0)
        self.pitch_deg.set(0.0)
        self.roll_deg.set(0.0)
        self._draw()

    def play(self):
        if self.is_playing:
            return
        self.is_playing = True
        self.last_tick = datetime.now().timestamp()
        self._tick()

    def pause(self):
        self.is_playing = False
        if self.play_after_id is not None:
            self.root.after_cancel(self.play_after_id)
            self.play_after_id = None

    def _tick(self):
        if not self.is_playing:
            return

        now = datetime.now().timestamp()
        dt = 0.0 if self.last_tick is None else (now - self.last_tick)
        self.last_tick = now

        self.yaw_deg.set((float(self.yaw_deg.get()) + self.play_deg_per_second * dt) % 360.0)
        self._draw()
        self.play_after_id = self.root.after(16, self._tick)

    def apply_all(self):
        self._draw()

    def _parse_inputs(self):
        try:
            lat = float(self.lat_str.get().strip())
            lon = float(self.lon_str.get().strip())
        except Exception:
            lat, lon = 0.0, 0.0
        lat, lon = clamp_lat_lon(lat, lon)

        try:
            dt_local = datetime.strptime(self.time_str.get().strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=APP_TZ)
        except Exception:
            dt_local = datetime.now(APP_TZ)

        dt_utc = dt_local.astimezone(timezone.utc)
        return lat, lon, dt_local, dt_utc

    def _ensure_earth_texture(self):
        if self.earth_texture is None:
            self.earth_texture = get_earth_texture(self.earth_texture_path)

    def _draw(self):
        lat, lon, dt_local, dt_utc = self._parse_inputs()

        gmst = gmst_degrees(dt_utc.replace(tzinfo=None))
        lst = (gmst + lon) % 360.0
        M = equatorial_to_local_enu_matrix(lat_deg=lat, lst_deg=lst)

        Ruser = rotation_matrix_from_euler(
            deg_x=float(self.roll_deg.get()),
            deg_y=float(self.pitch_deg.get()),
            deg_z=float(self.yaw_deg.get()),
        )

        self.ax.cla()

        self.ax.set_axis_off()

        self.ax.plot_surface(
            self.sphere_x, self.sphere_y, self.sphere_z,
            linewidth=0, antialiased=True, alpha=self.sphere_alpha
        )

        try:
            self._ensure_earth_texture()
            draw_textured_sphere(self.ax, self.earth_radius, self.earth_texture, center=(0.0, 0.0, 0.0), stride=3)
        except Exception:
            self.ax.scatter([0], [0], [0], s=80)

        ring_t = np.linspace(0, 2 * np.pi, 400)
        hx = self.radius * np.cos(ring_t)
        hy = self.radius * np.sin(ring_t)
        hz = np.zeros_like(ring_t)
        self.ax.plot(hx, hy, hz, linewidth=2)

        eq_ra = np.linspace(0, 360, 600, endpoint=False)
        eq_pts = np.stack([ra_dec_to_unit_vector_equatorial(r, 0.0) for r in eq_ra], axis=0)
        eq_enu = (M @ eq_pts.T).T
        eq_enu = (Ruser @ eq_enu.T).T
        self.ax.plot(eq_enu[:, 0], eq_enu[:, 1], eq_enu[:, 2], linewidth=1)

        ncp_eq = np.array([0.0, 0.0, 1.0])
        ncp_enu = M @ ncp_eq
        ncp_enu = Ruser @ ncp_enu
        self.ax.plot([0, self.radius * ncp_enu[0]],
                     [0, self.radius * ncp_enu[1]],
                     [0, self.radius * ncp_enu[2]],
                     linewidth=2)

        sun_ra, sun_dec = sun_ra_dec_degrees(dt_utc.replace(tzinfo=None))
        sun_eq = ra_dec_to_unit_vector_equatorial(sun_ra, sun_dec)
        sun_enu = M @ sun_eq
        sun_enu = Ruser @ sun_enu
        sun_pos = self.radius * sun_enu
        self.ax.scatter([sun_pos[0]], [sun_pos[1]], [sun_pos[2]], s=80)

        alt, az = unit_vector_enu_to_alt_az(sun_enu)
        text = (
            f"Lat {lat:.4f}  Lon {lon:.4f}\n"
            f"Local {dt_local.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"UTC {dt_utc.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Sun alt {alt:.1f} deg  az {az:.1f} deg"
        )
        self.ax.text(0.0, 0.0, -self.radius * 1.10, text, fontsize=9)

        mw_local = (M @ self.mw_eq_points.T).T
        mw_local = (Ruser @ mw_local.T).T

        I = self.mw_intensity
        sizes = 1.0 + 10.0 * (I ** 1.6)
        alphas = 0.08 + 0.75 * (I ** 1.8)

        colors = np.zeros((len(I), 4))
        colors[:, 0] = 1.0
        colors[:, 1] = 1.0
        colors[:, 2] = 1.0
        colors[:, 3] = np.clip(alphas, 0.0, 1.0)

        self.ax.scatter(mw_local[:, 0], mw_local[:, 1], mw_local[:, 2], s=sizes, c=colors, depthshade=False)

        lim = self.radius * 1.25
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)

        self.ax.view_init(elev=22, azim=40)

        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = CelestialSphereApp(root)
    root.mainloop()
