import tkinter as tk
from tkinter import ttk
from datetime import datetime, timezone
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


def rotation_matrix_from_euler(deg_x=0.0, deg_y=0.0, deg_z=0.0):
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


def unix_seconds_local(dt_local_naive: datetime) -> float:
    # Treat the user input as local time and convert to a timestamp.
    # This uses the system local timezone rules.
    return dt_local_naive.timestamp()


def sidereal_angle_degrees_from_unix(t_unix: float, t0_unix: float) -> float:
    # Approx Earth sidereal rotation rate.
    # One sidereal day is about 86164.0905 seconds.
    SIDEREAL_DAY_S = 86164.0905
    delta = t_unix - t0_unix
    angle = (delta / SIDEREAL_DAY_S) * 360.0
    return angle % 360.0


class CelestialSphereApp:
    def __init__(self, root):
        self.root = root
        root.title("Celestial Sphere with Milky Way")

        self.radius = 1.0
        self.sphere_alpha = 0.12

        self.mw_half_width_deg = 8.0
        self.mw_point_count = 2500
        self.mw_cross_count = 60
        self.mw_turbulence = 0.35
        self.mw_base_rotation_euler = (25.0, -20.0, 15.0)
        self.seed = 7

        self.is_playing = False
        self.play_after_id = None
        self.play_deg_per_second = 12.0  # visual speed, not physical
        self.last_tick_unix = None

        # Reference time for sidereal angle
        self.t0_unix = datetime(2025, 1, 1, 0, 0, 0).timestamp()

        # State
        self.yaw_deg = tk.DoubleVar(value=0.0)    # about Z
        self.pitch_deg = tk.DoubleVar(value=0.0)  # about Y
        self.roll_deg = tk.DoubleVar(value=0.0)   # about X

        self.time_str = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        self._build_ui()
        self._build_scene_geometry()
        self._draw()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")

        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        # Controls
        ttk.Label(left, text="Date and time (local)").grid(row=0, column=0, sticky="w")
        time_entry = ttk.Entry(left, textvariable=self.time_str, width=22)
        time_entry.grid(row=1, column=0, sticky="ew", pady=(2, 8))

        btns = ttk.Frame(left)
        btns.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(btns, text="Apply Time", command=self.apply_time).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(btns, text="Now", command=self.set_now).grid(row=0, column=1)

        playrow = ttk.Frame(left)
        playrow.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(playrow, text="Play", command=self.play).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(playrow, text="Pause", command=self.pause).grid(row=0, column=1)

        ttk.Label(left, text="Manual rotation").grid(row=4, column=0, sticky="w", pady=(0, 4))

        self._add_slider(left, "Yaw (deg)", self.yaw_deg, row=5)
        self._add_slider(left, "Pitch (deg)", self.pitch_deg, row=6)
        self._add_slider(left, "Roll (deg)", self.roll_deg, row=7)

        ttk.Button(left, text="Reset", command=self.reset).grid(row=8, column=0, sticky="ew", pady=(10, 0))

        for r in range(9):
            left.rowconfigure(r, weight=0)
        left.columnconfigure(0, weight=1)

        # Figure
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_box_aspect((1, 1, 1))

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _add_slider(self, parent, label, var, row):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=3)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        scale = ttk.Scale(frame, from_=-180, to=180, variable=var, command=lambda _=None: self._draw())
        scale.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)

    def _build_scene_geometry(self):
        # Sphere mesh
        u = np.linspace(0, 2*np.pi, 120)
        v = np.linspace(0, np.pi, 80)
        self.sphere_x = self.radius * np.outer(np.cos(u), np.sin(v))
        self.sphere_y = self.radius * np.outer(np.sin(u), np.sin(v))
        self.sphere_z = self.radius * np.outer(np.ones_like(u), np.cos(v))

        # Milky Way band points in base frame, then rotated by mw_base_rotation_euler
        rng = np.random.default_rng(self.seed)

        t = np.linspace(0, 2*np.pi, self.mw_point_count, endpoint=False)
        base = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)

        hw = np.deg2rad(self.mw_half_width_deg)
        offsets = np.linspace(-hw, hw, self.mw_cross_count)

        all_pts = []
        intensities = []

        for off in offsets:
            strand = base.copy()
            strand[:, 2] = np.sin(off)
            strand[:, 0] *= np.cos(off)
            strand[:, 1] *= np.cos(off)

            noise = rng.normal(0.0, 1.0, size=strand.shape)
            noise = noise / np.linalg.norm(noise, axis=1, keepdims=True)
            strand = strand + (self.mw_turbulence * 0.02) * noise

            strand = strand / np.linalg.norm(strand, axis=1, keepdims=True)
            strand = self.radius * strand

            edge = abs(off) / hw if hw > 0 else 0.0
            band_strength = (1.0 - edge) ** 1.8
            clump = 0.5 + 0.5 * np.sin(6*t + rng.uniform(0, 2*np.pi))
            intensity = band_strength * (0.55 + 0.45 * clump)

            all_pts.append(strand)
            intensities.append(intensity)

        P = np.concatenate(all_pts, axis=0)
        I = np.concatenate(intensities, axis=0)

        Rmw = rotation_matrix_from_euler(*self.mw_base_rotation_euler)
        P = (Rmw @ P.T).T

        self.mw_points_base = P
        self.mw_intensity = I

    def _draw(self):
        self.ax.cla()

        # Transparent sphere
        self.ax.plot_surface(
            self.sphere_x, self.sphere_y, self.sphere_z,
            linewidth=0, antialiased=True, alpha=self.sphere_alpha
        )

        # Earth at center
        self.ax.scatter([0], [0], [0], s=80)

        # Axes triad
        a = self.radius * 1.15
        self.ax.plot([0, a], [0, 0], [0, 0], linewidth=1)
        self.ax.plot([0, 0], [0, a], [0, 0], linewidth=1)
        self.ax.plot([0, 0], [0, 0], [0, a], linewidth=1)

        # Apply user rotation
        Ruser = rotation_matrix_from_euler(
            deg_x=float(self.roll_deg.get()),
            deg_y=float(self.pitch_deg.get()),
            deg_z=float(self.yaw_deg.get()),
        )
        P = (Ruser @ self.mw_points_base.T).T

        # Use intensity to drive alpha and point size
        I = self.mw_intensity
        sizes = 1.0 + 10.0 * (I ** 1.6)
        alphas = 0.08 + 0.75 * (I ** 1.8)

        # matplotlib scatter cannot take per point alpha directly via a single alpha param,
        # so we build RGBA colors.
        colors = np.zeros((len(I), 4))
        colors[:, 0] = 1.0
        colors[:, 1] = 1.0
        colors[:, 2] = 1.0
        colors[:, 3] = np.clip(alphas, 0.0, 1.0)

        self.ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=sizes, c=colors, depthshade=False)

        lim = self.radius * 1.25
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.canvas.draw_idle()

    def set_now(self):
        self.time_str.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def apply_time(self):
        try:
            dt_local = datetime.strptime(self.time_str.get().strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return

        t_unix = unix_seconds_local(dt_local)
        angle = sidereal_angle_degrees_from_unix(t_unix, self.t0_unix)

        # Map time rotation to yaw around Z
        self.yaw_deg.set(angle)
        self._draw()

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
        self.last_tick_unix = datetime.now().timestamp()
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
        dt = 0.0 if self.last_tick_unix is None else (now - self.last_tick_unix)
        self.last_tick_unix = now

        new_yaw = float(self.yaw_deg.get()) + self.play_deg_per_second * dt
        self.yaw_deg.set(new_yaw % 360.0)
        self._draw()

        self.play_after_id = self.root.after(16, self._tick)  # about 60 fps


if __name__ == "__main__":
    root = tk.Tk()
    app = CelestialSphereApp(root)
    root.mainloop()
