
from datetime import datetime
import numpy as np
from .time_utils import julian_day
from .coordinates import ra_dec_to_unit_vector_equatorial

def sun_ra_dec_degrees(dt_utc_naive: datetime) -> tuple[float, float]:
    """Calculate Sun's RA and Dec for given UTC time"""
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

def galactic_center_unit_eq() -> np.ndarray:
    """Get unit vector to galactic center in equatorial coordinates"""
    # J2000 galactic center approx: RA 266.4051 deg, Dec -28.936175 deg
    return ra_dec_to_unit_vector_equatorial(266.4051, -28.936175)

