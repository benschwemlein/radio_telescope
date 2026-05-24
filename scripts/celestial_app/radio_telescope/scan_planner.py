"""
Scan Planner — suggest optimal drift-scan parameters for hydrogen line observations.

A drift scan points the telescope at a fixed alt/az and lets Earth's rotation
sweep the beam across the galactic plane, producing a transit from empty sky
through the Milky Way and back to empty sky.  The 21-cm hydrogen line signal
appears as a peak in the spectrogram while the galactic plane is in the beam.

The planner searches the next 24 hours for all galactic-plane crossings that
are visible from the observer's location and scores each by:
  - Galactic brightness at that longitude (galactic centre best, anticenter worst)
  - Transit altitude (higher = less atmosphere = better signal)
  - Crossing geometry (angle of the galactic plane to the RA sweep direction)
  - Hours until the crossing (sooner = more immediately useful)
  - Whether the peak time falls within the caller's preferred waking hours
    (in-window scans score 6.7× higher than out-of-window ones)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from astronomy.coordinates import eq_to_gal_matrix_j2000, gal_to_eq_matrix_j2000
from astronomy.time_utils import gmst_degrees
from astronomy.galactic import _brightness_profile, _half_width_profile


# ---------------------------------------------------------------------------
# Galactic longitude → human-readable region name
# ---------------------------------------------------------------------------

_REGIONS = [
    (  0,  30, "Galactic Centre / Sagittarius"),
    ( 30,  60, "Scutum-Centaurus Arm"),
    ( 60, 100, "Cygnus / Scutum Arm"),
    (100, 150, "Perseus Arm"),
    (150, 210, "Anticenter / Auriga-Gemini"),
    (210, 260, "Monoceros / Outer Arm"),
    (260, 310, "Sagittarius Arm (far side)"),
    (310, 360, "Galactic Centre approach / Norma Arm"),
]


def _region_name(l_deg: float) -> str:
    l = l_deg % 360
    for lo, hi, name in _REGIONS:
        if lo <= l < hi:
            return name
    return "Galactic Plane"


# ---------------------------------------------------------------------------
# Main result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScanSuggestion:
    """A recommended drift-scan opportunity."""

    # Pointing
    altitude_deg: float          # fixed telescope altitude (elevation)
    azimuth_deg: float           # fixed telescope azimuth (0=N, 90=E, 180=S)

    # Timing
    start_time: datetime         # when to start recording (local tz preserved)
    peak_time: datetime          # expected time of peak HI signal
    end_time: datetime           # when to stop recording

    # What will be observed
    galactic_longitude_deg: float
    galactic_region: str
    dec_deg: float               # declination being scanned

    # Geometry
    crossing_duration_min: float  # time the galactic plane is in the beam
    baseline_min: float           # recommended baseline on each side

    # Quality
    brightness: float            # 0-1, brightness at this galactic longitude
    score: float                 # overall quality score (higher = better)
    hours_until_start: float
    in_preferred_window: bool = True  # peak time falls within user's waking hours

    @property
    def total_duration_min(self) -> float:
        return self.crossing_duration_min + 2 * self.baseline_min

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"{self.galactic_region}  |  "
            f"Alt {self.altitude_deg:.0f}°  Az {self.azimuth_deg:.0f}°  |  "
            f"Peak {self.peak_time.strftime('%H:%M')}  |  "
            f"Total {self.total_duration_min:.0f} min"
        )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def suggest_scans(
    lat_deg: float,
    lon_deg: float,
    dt_utc_naive: datetime,
    local_tz=None,
    min_alt_deg: float = 20.0,
    lookahead_hours: float = 24.0,
    beam_width_deg: float = 5.0,
    max_results: int = 8,
    earliest_local_hour: float = 0.0,
    latest_local_hour: float = 24.0,
) -> list[ScanSuggestion]:
    """
    Find the best drift-scan opportunities across the Milky Way.

    Args:
        lat_deg:              Observer latitude in degrees (North positive).
        lon_deg:              Observer longitude in degrees (East positive).
        dt_utc_naive:         Current UTC time (naive datetime).
        local_tz:             tzinfo for output times (None = UTC).
        min_alt_deg:          Minimum acceptable transit altitude in degrees.
        lookahead_hours:      Search window in hours from now.
        beam_width_deg:       Telescope beam FWHM in degrees (affects duration).
        max_results:          Maximum number of suggestions to return.
        earliest_local_hour:  Preferred start of waking hours (0–23, local time).
                              Scans whose PEAK falls within [earliest, latest)
                              are scored ~6.7× higher than out-of-window ones.
                              Default 0 means no preference applied.
        latest_local_hour:    Preferred end of waking hours (1–24, local time).
                              Default 24 means no preference applied.

    Returns:
        List of ScanSuggestion objects sorted best-first.  ``in_preferred_window``
        is True for scans whose peak time falls inside the preferred hours.
    """
    R_g2e = gal_to_eq_matrix_j2000().astype(np.float64)
    R_e2g = eq_to_gal_matrix_j2000().astype(np.float64)

    gmst = gmst_degrees(dt_utc_naive)
    lst_deg = (gmst + lon_deg) % 360.0

    # Effective beam radius added to the galactic half-width for start/end times
    beam_radius = beam_width_deg / 2.0

    # Fine-sample galactic longitudes
    n_samples = 3600
    l_arr = np.linspace(0.0, 360.0, n_samples, endpoint=False)
    l_rad = np.deg2rad(l_arr)

    # Galactic plane (b=0) in equatorial coordinates for each l
    g = np.stack([np.cos(l_rad), np.sin(l_rad), np.zeros(n_samples)], axis=1)  # (N,3)
    e = (R_g2e @ g.T).T  # (N,3) equatorial unit vectors
    ra_arr  = np.degrees(np.arctan2(e[:, 1], e[:, 0])) % 360.0
    dec_arr = np.degrees(np.arcsin(np.clip(e[:, 2], -1.0, 1.0)))

    # Per-longitude band properties
    hw_arr     = np.degrees(_half_width_profile(l_rad.astype(np.float32))).astype(np.float64)
    bright_arr = _brightness_profile(l_rad.astype(np.float32)).astype(np.float64)

    # Transit altitude and azimuth
    # Due-south transit (dec < lat): az=180, alt = 90 - (lat - dec)
    # Due-north transit (dec > lat): az=0,   alt = 90 - (dec - lat)
    alt_arr = 90.0 - np.abs(lat_deg - dec_arr)
    az_arr  = np.where(dec_arr <= lat_deg, 180.0, 0.0)

    # Hours until each RA transits the meridian (LST = RA)
    hours_to_transit = ((ra_arr - lst_deg) % 360.0) / 15.0  # sidereal hours

    # Crossing geometry: how wide is the band in the RA sweep direction?
    # Use a central difference of equatorial coords w.r.t. galactic longitude.
    ra_next  = np.roll(ra_arr, -1);  ra_next[-1]  += 360.0  # handle wrap
    dec_next = np.roll(dec_arr, -1)

    dra_deg  = ((ra_next - ra_arr + 180) % 360) - 180  # signed, degrees RA
    ddec_deg = dec_next - dec_arr                        # degrees Dec

    # On-sky angular step (account for cos dec compression in RA)
    dra_sky  = dra_deg * np.cos(np.deg2rad(dec_arr))
    band_len = np.hypot(dra_sky, ddec_deg)  # total on-sky step in degrees

    # RA-sky extent of the band (how far in the sky-RA direction the edges are)
    # = 2 * hw / |d| * |ddec|  (see module docstring for derivation)
    # Zero when band runs E-W (ddec≈0) → not useful for a drift scan
    with np.errstate(divide='ignore', invalid='ignore'):
        ra_sky_extent = np.where(
            band_len > 1e-6,
            2.0 * hw_arr * np.abs(ddec_deg) / band_len,
            0.0,
        )

    # Convert RA-sky extent to RA-angle extent and then to time
    # Earth rotates 15.04°/hr in HA, giving a sky-speed of 15.04*cos(dec) deg/hr
    # Time = ra_sky_extent / (15.04 * cos(dec))
    cos_dec = np.cos(np.deg2rad(dec_arr))
    with np.errstate(divide='ignore', invalid='ignore'):
        crossing_hours = np.where(
            cos_dec > 1e-6,
            ra_sky_extent / (15.04 * cos_dec),
            np.inf,
        )
    crossing_min = crossing_hours * 60.0

    # Baseline: enough time outside the band to establish a clean noise floor
    # Use max(20 min, crossing_min * 0.6) so short crossings still get context
    baseline_min = np.maximum(20.0, crossing_min * 0.6)

    # --- Filter ---
    mask = (
        (alt_arr >= min_alt_deg) &           # observable
        (hours_to_transit <= lookahead_hours) &   # within window
        (crossing_min >= 5.0) &              # at least 5 min of MW signal
        (crossing_min <= 240.0) &            # not an impossibly long crossing
        (band_len > 0.01)                    # valid geometry
    )

    if not np.any(mask):
        return []

    # --- Score ---
    alt_score  = np.sin(np.deg2rad(alt_arr))           # higher altitude = better
    time_score = 1.0 - hours_to_transit / (lookahead_hours + 1)
    dur_score  = np.exp(-crossing_min / 90.0)           # prefer ~30-90 min crossings

    # Preferred waking-hours window
    # Compute the local hour of each scan's PEAK as a float 0–24.
    has_window = (
        local_tz is not None
        and not (earliest_local_hour == 0.0 and latest_local_hour == 24.0)
    )
    if has_window:
        try:
            utc_offset_h = (
                dt_utc_naive.replace(tzinfo=timezone.utc)
                .astimezone(local_tz)
                .utcoffset()
                .total_seconds()
                / 3600.0
            )
        except Exception:
            utc_offset_h = 0.0
        # Peak time local hour (float, 0–24)
        peak_local_h = (
            dt_utc_naive.hour + dt_utc_naive.minute / 60.0
            + hours_to_transit + utc_offset_h
        ) % 24.0

        if earliest_local_hour <= latest_local_hour:
            # Normal window e.g. 07:00–23:00
            in_window = (
                (peak_local_h >= earliest_local_hour)
                & (peak_local_h < latest_local_hour)
            )
        else:
            # Overnight window e.g. 22:00–06:00
            in_window = (
                (peak_local_h >= earliest_local_hour)
                | (peak_local_h < latest_local_hour)
            )
        # In-window scans get a strong boost; out-of-window are kept but ranked low
        waking_score = np.where(in_window, 1.0, 0.15)
    else:
        in_window    = np.ones(n_samples, dtype=bool)
        waking_score = np.ones(n_samples)

    score = bright_arr * alt_score * waking_score * (0.6 + 0.4 * time_score) * (0.7 + 0.3 * dur_score)

    # Only consider masked candidates
    indices = np.where(mask)[0]
    sorted_idx = indices[np.argsort(-score[indices])]

    # De-duplicate: skip longitudes within 15° of an already-picked one
    picked: list[ScanSuggestion] = []
    picked_l: list[float] = []

    for idx in sorted_idx:
        if len(picked) >= max_results:
            break

        l_deg = float(l_arr[idx])

        # Skip if too close to an already-selected longitude
        if any(min(abs(l_deg - p), 360 - abs(l_deg - p)) < 15.0 for p in picked_l):
            continue

        # Build datetime objects
        h = float(hours_to_transit[idx])
        peak_utc  = dt_utc_naive + timedelta(hours=h)
        c_half_h  = float(crossing_min[idx]) / 120.0     # half crossing in hours
        b_h       = float(baseline_min[idx]) / 60.0      # baseline in hours

        start_utc = peak_utc - timedelta(hours=c_half_h + b_h)
        end_utc   = peak_utc + timedelta(hours=c_half_h + b_h)

        def _localise(dt: datetime) -> datetime:
            if local_tz is None:
                return dt
            return dt.replace(tzinfo=timezone.utc).astimezone(local_tz)

        picked_l.append(l_deg)
        picked.append(ScanSuggestion(
            altitude_deg          = round(float(alt_arr[idx]), 1),
            azimuth_deg           = float(az_arr[idx]),
            start_time            = _localise(start_utc),
            peak_time             = _localise(peak_utc),
            end_time              = _localise(end_utc),
            galactic_longitude_deg= round(l_deg, 1),
            galactic_region       = _region_name(l_deg),
            dec_deg               = round(float(dec_arr[idx]), 1),
            crossing_duration_min = round(float(crossing_min[idx]), 0),
            baseline_min          = round(float(baseline_min[idx]), 0),
            brightness            = round(float(bright_arr[idx]), 2),
            score                 = round(float(score[idx]), 4),
            hours_until_start     = round(max(0.0, h - float(crossing_min[idx]) / 120.0 - b_h), 2),
            in_preferred_window   = bool(in_window[idx]),
        ))

    return picked
