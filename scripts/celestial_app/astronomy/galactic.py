
import numpy as np
from .coordinates import gal_to_eq_matrix_j2000


def _brightness_profile(l_rad: np.ndarray) -> np.ndarray:
    """
    Longitudinal brightness along the galactic plane (0–1).

    Key landmarks:
      l=0°   Sagittarius / galactic center bulge  → 1.0  (brightest)
      l=80°  Cygnus / Scutum arm                  → 0.85 (secondary peak)
      l=180° Auriga / anticenter                  → 0.22 (dimmest)
      l=270° Perseus arm                          → 0.45 (moderate)
    """
    # Smooth primary gradient: 1.0 at l=0, 0.22 at l=π
    primary = 0.61 + 0.39 * np.cos(l_rad)

    # Cygnus / Scutum arm bump at l ≈ 75°
    cygnus = 0.22 * np.exp(-0.5 * ((l_rad - np.deg2rad(75)) / np.deg2rad(22)) ** 2)

    # Slight Perseus arm lift at l ≈ 135°
    perseus = 0.08 * np.exp(-0.5 * ((l_rad - np.deg2rad(135)) / np.deg2rad(20)) ** 2)

    return np.clip(primary + cygnus + perseus, 0.10, 1.0).astype(np.float32)


def _half_width_profile(l_rad: np.ndarray, max_hw_deg: float = 15.0) -> np.ndarray:
    """
    Band half-width in radians as a function of galactic longitude.

    Widest at the galactic center (Sagittarius bulge), narrowest at anticenter.
    max_hw_deg sets the half-width at l=0; anticenter gets ~38% of that.
    """
    hw_center = np.deg2rad(max_hw_deg)
    hw_anti   = hw_center * 0.38
    # Smooth cosine interpolation: 1.0 at l=0, 0.0 at l=π
    t = 0.5 + 0.5 * np.cos(l_rad)
    return (hw_anti + (hw_center - hw_anti) * t).astype(np.float32)


def build_milky_way_band_equatorial(
    radius: float = 1.0,
    half_width_deg: float = 15.0,
    n: int = 1800,
    m: int = 40,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Milky Way band points in equatorial coordinates.

    Band width and brightness vary with galactic longitude so the rendering
    matches the real sky:
      - Wide, dense, bright toward Sagittarius / galactic center (l=0°)
      - Secondary brightness peak toward Cygnus (l≈75°)
      - Narrow, sparse, dim toward the anticenter in Auriga (l=180°)

    Args:
        radius:         Celestial sphere radius.
        half_width_deg: Maximum band half-width in degrees (at galactic center).
        n:              Number of longitude samples around the band.
        m:              Number of latitude strips across the band width.
        seed:           RNG seed for positional jitter (deterministic output).

    Returns:
        P: (N, 3) float32 array of equatorial Cartesian positions.
        A: (N,)   float32 array of per-point alpha weights (0–1).
    """
    rng = np.random.default_rng(seed)
    R_g2e = gal_to_eq_matrix_j2000()

    # 1-D grids
    l_vals = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    t_vals = np.linspace(-1.0, 1.0, m, dtype=np.float32)  # normalised band pos

    # Per-longitude profiles — shape (n,)
    hw_l     = _half_width_profile(l_vals, max_hw_deg=half_width_deg)
    bright_l = _brightness_profile(l_vals)

    # 2-D grids — shape (m, n)
    l_grid     = l_vals[np.newaxis, :]       # broadcast → (m, n)
    t_grid     = t_vals[:, np.newaxis]       # broadcast → (m, n)
    hw_grid    = hw_l[np.newaxis, :]
    bright_grid = bright_l[np.newaxis, :]

    b_grid = (t_grid * hw_grid).astype(np.float32)  # actual galactic latitude

    # Galactic Cartesian coords — each (m, n)
    x = np.cos(b_grid) * np.cos(l_grid)
    y = np.cos(b_grid) * np.sin(l_grid)
    z = np.sin(b_grid)

    # Flatten to (m*n, 3) and add small angular jitter for a natural scatter look
    g = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    noise = rng.normal(0.0, 1.0, size=g.shape).astype(np.float32)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True) + 1e-12
    g = g + 0.008 * noise
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12

    # Rotate from galactic to equatorial frame
    e = (R_g2e @ g.T).T.astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
    P = (radius * e).astype(np.float32)

    # Alpha = latitude edge falloff × longitudinal brightness
    edge_falloff = (1.0 - np.abs(t_grid)) ** 1.8   # (m, n), peaks at band centre
    A = (edge_falloff * bright_grid).reshape(-1).astype(np.float32)

    return P, A
