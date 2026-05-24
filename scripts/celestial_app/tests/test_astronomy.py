"""
Unit tests for astronomy calculations.

Reference values from USNO/JPL and standard textbooks.
Run from the celestial_app directory: python -m pytest tests/
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from datetime import datetime
import numpy as np
import pytest

from astronomy.time_utils import julian_day, gmst_degrees
from astronomy.coordinates import (
    ra_dec_to_unit_vector_equatorial,
    equatorial_to_local_enu_matrix,
    unit_vector_enu_to_alt_az,
    alt_az_from_ra_dec,
    clamp_lat_lon,
    latlon_to_ecef,
    gal_to_eq_matrix_j2000,
    eq_to_gal_matrix_j2000,
)
from astronomy.celestial_objects import sun_ra_dec_degrees, galactic_center_unit_eq


# ---------------------------------------------------------------------------
# time_utils
# ---------------------------------------------------------------------------

class TestJulianDay:
    def test_j2000_epoch(self):
        # J2000.0 = 2000-01-01 12:00:00 UTC → JD 2451545.0
        dt = datetime(2000, 1, 1, 12, 0, 0)
        assert abs(julian_day(dt) - 2451545.0) < 1e-6

    def test_known_date(self):
        # 1987-04-10 0h UT → JD 2446895.5  (Meeus, Astronomical Algorithms example)
        dt = datetime(1987, 4, 10, 0, 0, 0)
        assert abs(julian_day(dt) - 2446895.5) < 1e-4

    def test_unix_epoch(self):
        # 1970-01-01 00:00:00 UTC → JD 2440587.5
        dt = datetime(1970, 1, 1, 0, 0, 0)
        assert abs(julian_day(dt) - 2440587.5) < 1e-4

    def test_monotonic(self):
        dt1 = datetime(2024, 1, 1, 0, 0, 0)
        dt2 = datetime(2024, 1, 2, 0, 0, 0)
        assert julian_day(dt2) > julian_day(dt1)

    def test_one_day_increment(self):
        dt1 = datetime(2024, 6, 1, 0, 0, 0)
        dt2 = datetime(2024, 6, 2, 0, 0, 0)
        assert abs((julian_day(dt2) - julian_day(dt1)) - 1.0) < 1e-9


class TestGMST:
    def test_j2000_epoch(self):
        # GMST at J2000.0 (2000-01-01 12:00 UT) ≈ 280.46061837°
        dt = datetime(2000, 1, 1, 12, 0, 0)
        gmst = gmst_degrees(dt)
        assert abs(gmst - 280.46061837) < 0.01

    def test_in_range(self):
        for month in range(1, 13):
            dt = datetime(2024, month, 15, 0, 0, 0)
            gmst = gmst_degrees(dt)
            assert 0.0 <= gmst < 360.0

    def test_advances_with_time(self):
        dt1 = datetime(2024, 6, 1, 0, 0, 0)
        dt2 = datetime(2024, 6, 1, 6, 0, 0)  # 6 hours later
        g1 = gmst_degrees(dt1)
        g2 = gmst_degrees(dt2)
        # Earth rotates ~360.985° per solar day; 6h ≈ 90.25°
        diff = (g2 - g1) % 360.0
        assert abs(diff - 90.246) < 0.1


# ---------------------------------------------------------------------------
# coordinates
# ---------------------------------------------------------------------------

class TestRaDecToUnitVector:
    def test_ra0_dec0_points_along_x(self):
        v = ra_dec_to_unit_vector_equatorial(0.0, 0.0)
        assert abs(v[0] - 1.0) < 1e-6
        assert abs(v[1]) < 1e-6
        assert abs(v[2]) < 1e-6

    def test_north_celestial_pole(self):
        # Dec=90 → points along +Z
        v = ra_dec_to_unit_vector_equatorial(0.0, 90.0)
        assert abs(v[2] - 1.0) < 1e-5

    def test_south_celestial_pole(self):
        v = ra_dec_to_unit_vector_equatorial(0.0, -90.0)
        assert abs(v[2] + 1.0) < 1e-5

    def test_unit_length(self):
        for ra, dec in [(0, 0), (90, 30), (180, -45), (270, 60)]:
            v = ra_dec_to_unit_vector_equatorial(ra, dec)
            assert abs(np.linalg.norm(v) - 1.0) < 1e-5


class TestEquatorialToENUMatrix:
    def test_orthonormal(self):
        M = equatorial_to_local_enu_matrix(40.0, 100.0)
        # Rows should be unit vectors
        for i in range(3):
            assert abs(np.linalg.norm(M[i]) - 1.0) < 1e-5
        # Rows should be mutually orthogonal
        assert abs(np.dot(M[0], M[1])) < 1e-5
        assert abs(np.dot(M[0], M[2])) < 1e-5
        assert abs(np.dot(M[1], M[2])) < 1e-5

    def test_determinant_plus_one(self):
        # Proper rotation: det = +1
        M = equatorial_to_local_enu_matrix(39.96, 280.0)
        assert abs(np.linalg.det(M) - 1.0) < 1e-5

    def test_various_locations(self):
        for lat, lst in [(0, 0), (45, 90), (-33, 150), (90, 270)]:
            M = equatorial_to_local_enu_matrix(lat, lst)
            assert abs(np.linalg.det(M) - 1.0) < 1e-5


class TestUnitVectorToAltAz:
    def test_zenith(self):
        # ENU up vector → alt=90, any az
        v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        alt, az = unit_vector_enu_to_alt_az(v)
        assert abs(alt - 90.0) < 1e-4

    def test_north_horizon(self):
        # ENU north at horizon → alt=0, az=0
        v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        alt, az = unit_vector_enu_to_alt_az(v)
        assert abs(alt) < 1e-4
        assert abs(az) < 1e-4

    def test_east_horizon(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        alt, az = unit_vector_enu_to_alt_az(v)
        assert abs(alt) < 1e-4
        assert abs(az - 90.0) < 1e-4

    def test_altitude_range(self):
        for v in [np.array([1, 0, 0]), np.array([0, 1, 1]), np.array([-1, -1, -0.5])]:
            v = v / np.linalg.norm(v)
            alt, az = unit_vector_enu_to_alt_az(v.astype(np.float32))
            assert -90.0 <= alt <= 90.0
            assert 0.0 <= az < 360.0


class TestAltAzConsistency:
    """alt_az_from_ra_dec and the matrix route must agree."""

    def test_matches_matrix_method(self):
        lat, lst, ra, dec = 39.96, 100.0, 45.0, 30.0
        alt_f, az_f = alt_az_from_ra_dec(lat, lst, ra, dec)

        v_eq = ra_dec_to_unit_vector_equatorial(ra, dec)
        M = equatorial_to_local_enu_matrix(lat, lst)
        v_loc = (M @ v_eq.reshape(3, 1)).ravel().astype(np.float32)
        v_loc = v_loc / np.linalg.norm(v_loc)
        alt_m, az_m = unit_vector_enu_to_alt_az(v_loc)

        assert abs(alt_f - alt_m) < 0.001
        assert abs((az_f - az_m + 180) % 360 - 180) < 0.001


class TestClampLatLon:
    def test_valid_passthrough(self):
        assert clamp_lat_lon(45.0, -90.0) == (45.0, -90.0)

    def test_clamps_latitude(self):
        lat, lon = clamp_lat_lon(100.0, 0.0)
        assert lat == 90.0
        lat, lon = clamp_lat_lon(-100.0, 0.0)
        assert lat == -90.0

    def test_wraps_longitude(self):
        lat, lon = clamp_lat_lon(0.0, 270.0)
        assert abs(lon - (-90.0)) < 1e-9
        lat, lon = clamp_lat_lon(0.0, -190.0)
        assert abs(lon - 170.0) < 1e-9


class TestLatLonToECEF:
    def test_equator_prime_meridian(self):
        # lat=0, lon=0, r=1 → (1, 0, 0)
        v = latlon_to_ecef(0.0, 0.0, 1.0)
        assert abs(v[0] - 1.0) < 1e-5
        assert abs(v[1]) < 1e-5
        assert abs(v[2]) < 1e-5

    def test_north_pole(self):
        v = latlon_to_ecef(90.0, 0.0, 1.0)
        assert abs(v[2] - 1.0) < 1e-5

    def test_unit_length(self):
        v = latlon_to_ecef(39.96, -83.0, 1.0)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5


class TestGalacticMatrix:
    def test_eq_gal_is_rotation(self):
        M = eq_to_gal_matrix_j2000()
        assert abs(np.linalg.det(M) - 1.0) < 1e-4

    def test_gal_to_eq_is_inverse(self):
        R = eq_to_gal_matrix_j2000()
        Rinv = gal_to_eq_matrix_j2000()
        product = R @ Rinv
        assert np.allclose(product, np.eye(3), atol=1e-5)

    def test_galactic_center_roundtrip(self):
        # Galactic center at (l=0, b=0) in galactic frame = (1, 0, 0)
        gc_gal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        gc_eq = gal_to_eq_matrix_j2000().astype(np.float64) @ gc_gal
        gc_eq_unit = gc_eq / np.linalg.norm(gc_eq)

        # Should be very close to the known equatorial direction of galactic center
        expected = ra_dec_to_unit_vector_equatorial(266.4051, -28.936175).astype(np.float64)
        sep_deg = math.degrees(math.acos(float(np.clip(np.dot(gc_eq_unit, expected), -1, 1))))
        assert sep_deg < 0.1  # within 0.1° (matrix precision)


# ---------------------------------------------------------------------------
# celestial_objects
# ---------------------------------------------------------------------------

class TestSunRaDec:
    def test_returns_tuple_in_range(self):
        dt = datetime(2024, 6, 21, 12, 0, 0)
        ra, dec = sun_ra_dec_degrees(dt)
        assert 0.0 <= ra < 360.0
        assert -90.0 <= dec <= 90.0

    def test_summer_solstice_dec(self):
        # Near summer solstice sun declination should be ~+23.4°
        dt = datetime(2024, 6, 21, 0, 0, 0)
        _, dec = sun_ra_dec_degrees(dt)
        assert abs(dec - 23.44) < 0.5

    def test_winter_solstice_dec(self):
        # Near winter solstice sun declination should be ~-23.4°
        dt = datetime(2024, 12, 21, 0, 0, 0)
        _, dec = sun_ra_dec_degrees(dt)
        assert abs(dec + 23.44) < 0.5

    def test_equinox_dec_near_zero(self):
        # At vernal equinox sun declination ≈ 0°
        dt = datetime(2024, 3, 20, 3, 6, 0)  # approx equinox time
        _, dec = sun_ra_dec_degrees(dt)
        assert abs(dec) < 1.0


class TestGalacticCenter:
    def test_unit_vector(self):
        v = galactic_center_unit_eq()
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    def test_known_direction(self):
        # GC: RA≈266.4°, Dec≈-28.9°
        v = galactic_center_unit_eq()
        expected = ra_dec_to_unit_vector_equatorial(266.4051, -28.936175)
        assert np.allclose(v, expected, atol=1e-4)
