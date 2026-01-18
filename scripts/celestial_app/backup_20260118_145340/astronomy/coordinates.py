import numpy as np

def ra_dec_to_unit_vector_equatorial(ra_deg: float, dec_deg: float) -> np.ndarray:
    """Convert RA/Dec to unit vector in equatorial coordinates"""
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
    """Convert ENU unit vector to altitude/azimuth"""
    x, y, z = float(v_enu[0]), float(v_enu[1]), float(v_enu[2])
    alt = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    az = np.rad2deg(np.arctan2(x, y)) % 360.0
    return float(alt), float(az)

def alt_az_from_ra_dec(lat_deg: float, lst_deg: float, ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Calculate altitude and azimuth from RA/Dec"""
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

def latlon_to_ecef(lat_deg: float, lon_deg: float, r: float) -> np.ndarray:
    """Convert lat/lon to ECEF (Earth-Centered Earth-Fixed) coordinates"""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z], dtype=np.float32)

def latlon_to_unit_ecef(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Convert lat/lon to unit ECEF vector"""
    v = latlon_to_ecef(lat_deg, lon_deg, 1.0)
    return (v / np.linalg.norm(v)).astype(np.float32)

def ecef_basis_at(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get east, north, up basis vectors at given lat/lon"""
    lon = np.deg2rad(lon_deg)
    up = latlon_to_unit_ecef(lat_deg, lon_deg)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=np.float32)
    east = east / np.linalg.norm(east)
    north = np.cross(up, east).astype(np.float32)
    north = north / np.linalg.norm(north)
    return east, north, up

def make_horizon_ring_ecef(radius: float, lat_deg: float, lon_deg: float, n: int = 600) -> np.ndarray:
    """Generate horizon ring in ECEF coordinates"""
    east, north, _up = ecef_basis_at(lat_deg, lon_deg)
    t = np.linspace(0, 2 * np.pi, n, endpoint=True).astype(np.float32)
    pts = radius * (np.cos(t)[:, None] * east[None, :] + np.sin(t)[:, None] * north[None, :])
    return pts.astype(np.float32)

def eq_to_gal_matrix_j2000() -> np.ndarray:
    """IAU J2000 Equatorial to Galactic rotation matrix"""
    return np.array([
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [+0.4941094279, -0.4448296300, +0.7469822445],
        [-0.8676661490, -0.1980763734, +0.4559837762],
    ], dtype=np.float32)

def gal_to_eq_matrix_j2000() -> np.ndarray:
    """Galactic to Equatorial (transpose of eq_to_gal for orthonormal matrix)"""
    return eq_to_gal_matrix_j2000().T.astype(np.float32)

def clamp_lat_lon(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Clamp latitude and longitude to valid ranges"""
    lat = max(-90.0, min(90.0, float(lat_deg)))
    lon = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    return lat, lon