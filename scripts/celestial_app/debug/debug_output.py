"""
Debug output module for celestial sphere calculations
"""
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

def vstr(v: np.ndarray) -> str:
    """Format vector as string"""
    v = np.asarray(v).astype(float).ravel()
    return f"[{v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f}]"

def mstr(M: np.ndarray) -> str:
    """Format matrix as string"""
    M = np.asarray(M).astype(float)
    return (
        "\n"
        f"  [{M[0,0]: .6f} {M[0,1]: .6f} {M[0,2]: .6f}]\n"
        f"  [{M[1,0]: .6f} {M[1,1]: .6f} {M[1,2]: .6f}]\n"
        f"  [{M[2,0]: .6f} {M[2,1]: .6f} {M[2,2]: .6f}]"
    )

def det3(M: np.ndarray) -> float:
    """Calculate 3x3 matrix determinant"""
    return float(np.linalg.det(np.asarray(M).astype(float)))

def row_norms(M: np.ndarray) -> list[float]:
    """Calculate norms of matrix rows"""
    M = np.asarray(M).astype(float)
    return [float(np.linalg.norm(M[i, :])) for i in range(3)]

def dot_rows(M: np.ndarray) -> tuple[float, float, float]:
    """Calculate dot products between matrix rows"""
    M = np.asarray(M).astype(float)
    return (
        float(np.dot(M[0, :], M[1, :])),
        float(np.dot(M[0, :], M[2, :])),
        float(np.dot(M[1, :], M[2, :])),
    )

def angular_sep_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Calculate angular separation in degrees between two unit vectors"""
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    return float(np.rad2deg(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))

def print_celestial_debug(dt_local, dt_utc, lat, lon, jd, gmst, lst, 
                         sun_ra, sun_dec, sun_eq, sun_local, 
                         alt_vec, az_vec, alt_formula, az_formula,
                         M, p_ecef, p_view, gc_eq, mw_pts_eq, radius,
                         earth_rot_sign, app_tz):
    """Print comprehensive debug information"""
    
    ha = (lst - sun_ra) % 360.0
    
    from astronomy.coordinates import eq_to_gal_matrix_j2000, unit_vector_enu_to_alt_az
    
    E2G = eq_to_gal_matrix_j2000()
    g = (E2G @ sun_eq.reshape(3, 1)).ravel()
    g = g / (np.linalg.norm(g) + 1e-12)
    b_gal = np.rad2deg(np.arcsin(np.clip(g[2], -1.0, 1.0)))
    l_gal = np.rad2deg(np.arctan2(g[1], g[0])) % 360.0
    
    # GC sanity checks
    gc_gal = (E2G @ gc_eq.reshape(3, 1)).ravel()
    gc_gal = gc_gal / (np.linalg.norm(gc_gal) + 1e-12)
    gc_l = np.rad2deg(np.arctan2(gc_gal[1], gc_gal[0])) % 360.0
    gc_b = np.rad2deg(np.arcsin(np.clip(gc_gal[2], -1.0, 1.0)))
    
    print(f"  GC in galactic coords: l={gc_l:.4f} deg, b={gc_b:.4f} deg (should be ~0, ~0)")
    
    scp = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    ncp = np.array([0.0, 0.0,  1.0], dtype=np.float32)
    
    gc_sep_scp = angular_sep_deg(gc_eq, scp)
    gc_sep_ncp = angular_sep_deg(gc_eq, ncp)
    gc_local = (M @ gc_eq.reshape(3, 1)).ravel().astype(np.float32)
    gc_local = gc_local / (np.linalg.norm(gc_local) + 1e-12)
    gc_alt, gc_az = unit_vector_enu_to_alt_az(gc_local)
    
    mw_dirs_eq = mw_pts_eq / (radius + 1e-12)
    dots = np.clip(mw_dirs_eq @ (gc_eq / (np.linalg.norm(gc_eq) + 1e-12)), -1.0, 1.0)
    min_sep = float(np.rad2deg(np.arccos(float(np.max(dots)))))
    
    print("\n" + "="*60)
    print("CELESTIAL SPHERE DEBUG OUTPUT")
    print("="*60)
    print(f"\nTime Information:")
    print(f"  Local time: {dt_local.strftime('%Y-%m-%d %H:%M:%S')} {app_tz.key}")
    print(f"  UTC time:   {dt_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Julian Day: {jd:.12f}")
    
    print(f"\nObserver Location:")
    print(f"  Latitude:  {lat: .6f}°")
    print(f"  Longitude: {lon: .6f}°")
    print(f"  ECEF position: {vstr(p_ecef)}")
    print(f"  View position: {vstr(p_view)}")
    
    print(f"\nSidereal Time:")
    print(f"  GMST: {gmst: .6f}°")
    print(f"  LST:  {lst: .6f}°")
    print(f"  Earth rotation sign: {earth_rot_sign: .1f}")
    
    print(f"\nSun Position:")
    print(f"  RA:          {sun_ra: .6f}°")
    print(f"  Dec:         {sun_dec: .6f}°")
    print(f"  Hour Angle:  {ha: .6f}°")
    print(f"  Equatorial unit vector: {vstr(sun_eq)}")
    print(f"  Local ENU unit vector:  {vstr(sun_local)}")
    print(f"  Altitude (vector):   {alt_vec: .6f}°")
    print(f"  Azimuth (vector):    {az_vec: .6f}°")
    print(f"  Altitude (formula):  {alt_formula: .6f}°")
    print(f"  Azimuth (formula):   {az_formula: .6f}°")
    print(f"  Difference (vec-formula):")
    print(f"    Alt: {alt_vec-alt_formula:+.9f}°")
    print(f"    Az:  {((az_vec-az_formula+540)%360-180):+.9f}°")
    
    print(f"\nSun Galactic Coordinates:")
    print(f"  Galactic longitude (l): {l_gal: .6f}°")
    print(f"  Galactic latitude (b):  {b_gal: .6f}°")
    print(f"  Off-plane distance:     {abs(b_gal): .6f}°")
    
    print(f"\nCoordinate Transformation Matrix (Eq->ENU):")
    print(mstr(M))
    print(f"  Determinant:  {det3(M): .9f}")
    rn = row_norms(M)
    print(f"  Row norms:    {rn}")
    dr = dot_rows(M)
    print(f"  Row dot products: {dr[0]: .9e}, {dr[1]: .9e}, {dr[2]: .9e}")
    
    print(f"\nGalactic Center Validation:")
    print(f"  GC equatorial unit vector: {vstr(gc_eq)}")
    print(f"  GC local altitude:  {gc_alt: .6f}°")
    print(f"  GC local azimuth:   {gc_az: .6f}°")
    print(f"  GC angle to SCP:    {gc_sep_scp: .6f}° (expected ~61.064°)")
    print(f"  GC angle to NCP:    {gc_sep_ncp: .6f}° (expected ~118.936°)")
    print(f"  Min separation GC<->Milky Way points: {min_sep: .6f}° (should be small)")
    
    print("="*60 + "\n")

def print_sun_debug(lat, lon, dt_local, dt_utc, app_tz):
    """Print focused sun position debug info"""
    from astronomy.time_utils import julian_day, gmst_degrees
    from astronomy.celestial_objects import sun_ra_dec_degrees
    from astronomy.coordinates import (
        equatorial_to_local_enu_matrix,
        ra_dec_to_unit_vector_equatorial,
        unit_vector_enu_to_alt_az,
        alt_az_from_ra_dec
    )
    
    dt_utc_naive = dt_utc.replace(tzinfo=None)
    jd = julian_day(dt_utc_naive)
    gmst = gmst_degrees(dt_utc_naive)
    lst = (gmst + lon) % 360.0
    sun_ra, sun_dec = sun_ra_dec_degrees(dt_utc_naive)
    M = equatorial_to_local_enu_matrix(lat, lst).astype(np.float32)
    sun_eq = ra_dec_to_unit_vector_equatorial(sun_ra, sun_dec)
    sun_local = (M @ sun_eq.reshape(3, 1)).ravel().astype(np.float32)
    sun_local = sun_local / (np.linalg.norm(sun_local) + 1e-12)
    alt_vec, az_vec = unit_vector_enu_to_alt_az(sun_local)
    alt_formula, az_formula = alt_az_from_ra_dec(lat, lst, sun_ra, sun_dec)
    ha = (lst - sun_ra) % 360.0
    max_alt_theory = 90.0 - abs(lat - sun_dec)
    
    print("\n" + "="*60)
    print("SUN POSITION DEBUG")
    print("="*60)
    print(f"Local: {dt_local.strftime('%Y-%m-%d %H:%M:%S')} {app_tz.key}")
    print(f"UTC:   {dt_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Location: Lat {lat:.6f}°  Lon {lon:.6f}°")
    print(f"Julian Day: {jd:.12f}")
    print(f"GMST: {gmst:.6f}°  LST: {lst:.6f}°")
    print(f"Sun RA: {sun_ra:.6f}°  Dec: {sun_dec:.6f}°  HA: {ha:.6f}°")
    print(f"Altitude (vector):  {alt_vec:.6f}°   Azimuth (vector):  {az_vec:.6f}°")
    print(f"Altitude (formula): {alt_formula:.6f}°   Azimuth (formula): {az_formula:.6f}°")
    print(f"Theory max altitude (at transit): {max_alt_theory:.6f}°")
    print(f"Difference (vec-formula):")
    print(f"  Alt: {alt_vec-alt_formula:+.9f}°")
    print(f"  Az:  {((az_vec-az_formula+540)%360-180):+.9f}°")
    print("="*60 + "\n")

def find_max_alt_over_24h(lat_deg, lon_deg, start_local, step_minutes=4):
    """Find maximum altitude of galactic center over 24 hours"""
    from datetime import timedelta, timezone
    from astronomy.time_utils import gmst_degrees
    from astronomy.celestial_objects import galactic_center_unit_eq
    from astronomy.coordinates import equatorial_to_local_enu_matrix, unit_vector_enu_to_alt_az
    
    v_gc = galactic_center_unit_eq()
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

def print_gc_visibility(lat, lon, dt_local, app_tz):
    """Print galactic center visibility over next 24 hours"""
    best_alt, best_az, best_time = find_max_alt_over_24h(lat, lon, dt_local, step_minutes=4)
    print(f"\nGalactic Center Visibility (next 24h):")
    print(f"  Maximum altitude: {best_alt:.6f}°")
    print(f"  At time: {best_time.strftime('%Y-%m-%d %H:%M:%S')} {app_tz.key}")
    print(f"  Azimuth: {best_az:.6f}°")