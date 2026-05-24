# Radio Telescope

A PyQt6 application for visualizing and planning radio telescope observations with an RTL-SDR dongle on a Raspberry Pi, focused on hydrogen line (1420 MHz) observations.

## Features

- **3D Globe View** — celestial sphere with Earth, Milky Way band, Sun, galactic center, horizon ring, and compass markers
- **2D Star Chart** — printable planisphere showing the sky as seen from your location and time
- **Scan Planner** — define telescope scans (altitude, azimuth, duration, beam width), persist them to a local database, and visualize them on both views
- **Hydrogen line analysis scripts** — standalone scripts in `scripts/` for plotting recorded SDR data

## Requirements

- Python 3.11+
- RTL-SDR Blog V4 dongle (for actual observations)
- OpenGL-capable display

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Running

The app must be run from the `celestial_app` directory so that relative package imports resolve correctly:

```bash
cd scripts/celestial_app
python app.py
```

On first launch the app downloads a 2 MB Earth texture from NASA and caches it as `earth_texture.jpg` in the working directory. If the download fails, a plain dark-blue fallback is used.

## Data storage

Scan plans are stored in `~/.radio_telescope/scans.db` (SQLite). They persist across sessions and are loaded automatically at startup.

## Architecture

```
scripts/celestial_app/
├── app.py                        # Entry point
├── astronomy/                    # Pure-Python astronomy calculations
│   ├── celestial_objects.py      # Sun RA/Dec, galactic center
│   ├── coordinates.py            # Coordinate transforms (EQ ↔ ENU, ECEF, alt/az)
│   ├── galactic.py               # Milky Way band mesh generation
│   └── time_utils.py             # Julian Day, GMST, LST
├── database/
│   └── scan_db.py                # SQLite scan storage
├── geometry/
│   ├── mesh_generation.py        # UV sphere, disk mesh, Earth texture download
│   └── transformations.py        # Vector normalization, rotation matrices
├── radio_telescope/
│   └── scan_path.py              # Scan path geometry (band mesh, 2D projection)
├── ui/
│   ├── main_window.py            # Top-level Qt window and controller
│   ├── globe_view.py             # 3D globe view mode
│   ├── star_chart_view.py        # 2D Matplotlib star chart
│   ├── scan_dialog.py            # New-scan input dialog
│   └── theme.py                  # Color theme constants
├── visualization/
│   ├── gl_widgets.py             # Custom pyqtgraph OpenGL widget
│   ├── milky_way_renderer.py     # Milky Way image sampling utility (unused by default)
│   └── scene_builder.py          # Constructs all 3D scene items
└── debug/
    └── debug_output.py           # Verbose debug printing (toggled by DEBUG_ENABLED)
```

Coordinate systems used:
- **Equatorial (J2000)** — RA/Dec, used for celestial objects and the Milky Way band
- **ENU (East-North-Up)** — local horizontal frame at the observer's position
- **ECEF** — Earth-centered frame used for the Earth mesh and horizon ring
- **Galactic** — used internally for Milky Way band generation (IAU 1958 matrix)

## Hardware setup

See [SETUP.md](SETUP.md) for detailed RTL-SDR hardware setup, `rtl_power` scanning commands, and Raspberry Pi configuration.
