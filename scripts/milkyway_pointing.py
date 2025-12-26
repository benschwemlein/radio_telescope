#!/usr/bin/env python3
"""
milkyway_pointing_window.py

Pick a dish pointing for Milky Way HI work that stays above your obstruction limits
for the full scan duration.

Outputs azimuth and altitude (degrees above horizon) at the start time.

Install deps:
  python3 -m pip install astropy pytz numpy

Example:
  python3 milkyway_pointing_window.py 39.9612 -82.9988 "2025-12-26 23:30:00" \
    --tz "America/New_York" --minutes 60 --sample_sec 30 \
    --min_alt 10 \
    --blocked 300 60 35 \
    --blocked 220 260 25 \
    --blocked 90 120 20
"""

import argparse
from datetime import datetime, timedelta
import numpy as np


def az_in_sector(az_deg: np.ndarray, start: float, end: float) -> np.ndarray:
    """
    Returns boolean mask for azimuths inside the sector [start, end] in degrees.
    Handles wraparound, eg 300 to 60 crosses north.
    """
    az = az_deg % 360.0
    start = start % 360.0
    end = end % 360.0

    if start <= end:
        return (az >= start) & (az <= end)
    else:
        return (az >= start) | (az <= end)


def required_min_alt(az_deg: np.ndarray, default_min_alt: float, sectors) -> np.ndarray:
    """
    sectors is a list of tuples (start_az, end_az, min_alt).
    Later sectors override earlier ones if they overlap.
    """
    req = np.full_like(az_deg, float(default_min_alt), dtype=float)
    for start_az, end_az, min_alt in sectors:
        m = az_in_sector(az_deg, start_az, end_az)
        req[m] = float(min_alt)
    return req


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lat_deg", type=float)
    ap.add_argument("lon_deg", type=float, help="East positive. West is negative.")
    ap.add_argument("local_time", type=str, help='Format: "YYYY-MM-DD HH:MM:SS"')
    ap.add_argument("--tz", type=str, default="America/New_York")
    ap.add_argument("--height_m", type=float, default=0.0)

    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--sample_sec", type=float, default=30.0)

    ap.add_argument("--min_alt", type=float, default=10.0, help="Default minimum altitude in degrees.")
    ap.add_argument(
        "--blocked",
        nargs=3,
        action="append",
        metavar=("START_AZ", "END_AZ", "MIN_ALT"),
        help="Add a blocked sector: start az deg, end az deg, min altitude deg.",
    )

    ap.add_argument("--step_deg", type=float, default=2.0, help="Candidate step along the galactic plane.")
    args = ap.parse_args()

    try:
        import pytz
        import astropy.units as u
        from astropy.time import Time
        from astropy.coordinates import EarthLocation, SkyCoord, AltAz
    except Exception as e:
        raise SystemExit(
            "Missing dependency. Run:\n"
            "  python3 -m pip install astropy pytz numpy\n\n"
            f"Import error: {e}"
        )

    sectors = []
    if args.blocked:
        for s in args.blocked:
            sectors.append((float(s[0]), float(s[1]), float(s[2])))

    tz = pytz.timezone(args.tz)
    dt_local = tz.localize(datetime.strptime(args.local_time, "%Y-%m-%d %H:%M:%S"))
    dt_utc = dt_local.astimezone(pytz.utc)

    duration_sec = float(args.minutes) * 60.0
    sample_sec = float(args.sample_sec)
    if sample_sec <= 0:
        raise SystemExit("sample_sec must be > 0")

    n = int(np.floor(duration_sec / sample_sec)) + 1
    times_utc = [dt_utc + timedelta(seconds=i * sample_sec) for i in range(n)]
    obstimes = Time(times_utc)

    location = EarthLocation(
        lat=args.lat_deg * u.deg,
        lon=args.lon_deg * u.deg,
        height=args.height_m * u.m,
    )
    frame = AltAz(obstime=obstimes, location=location)

    def eval_target(sc: "SkyCoord"):
        aa = sc.transform_to(frame)
        alt = aa.alt.to_value(u.deg).astype(float)
        az = aa.az.to_value(u.deg).astype(float)
        req = required_min_alt(az, args.min_alt, sectors)
        ok = alt >= req
        return alt, az, req, ok

    def fmt_deg(x):
        return f"{x:.1f}"

    # Evaluate Galactic Center visibility for the entire window
    gc = SkyCoord(l=0 * u.deg, b=0 * u.deg, frame="galactic")
    gc_alt, gc_az, gc_req, gc_ok = eval_target(gc)

    # Search for the best point on the galactic plane (b=0) that stays clear for the full window
    best = None
    step = float(args.step_deg)
    if step <= 0:
        raise SystemExit("step_deg must be > 0")

    l_vals = np.arange(0.0, 360.0, step, dtype=float)

    for l in l_vals:
        sc = SkyCoord(l=l * u.deg, b=0 * u.deg, frame="galactic")
        alt, az, req, ok = eval_target(sc)

        if not np.all(ok):
            continue

        score = float(np.min(alt - req))  # best worst case clearance over the whole scan
        min_alt = float(np.min(alt))
        if best is None or score > best["score"]:
            best = {
                "l_deg": float(l),
                "score": score,
                "min_alt": min_alt,
                "start_alt": float(alt[0]),
                "start_az": float(az[0]),
                "end_alt": float(alt[-1]),
                "end_az": float(az[-1]),
            }

    print("Pointing window")
    print(f"  Local start time: {dt_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Duration: {args.minutes:.1f} minutes")
    print(f"  Sampling: {sample_sec:.0f} seconds")
    print()

    print("Obstruction model")
    print(f"  Default minimum altitude: {fmt_deg(args.min_alt)} deg")
    if sectors:
        for start_az, end_az, min_alt in sectors:
            print(f"  Sector az {fmt_deg(start_az)} to {fmt_deg(end_az)} requires alt >= {fmt_deg(min_alt)} deg")
    else:
        print("  No blocked sectors provided")
    print()

    print("Galactic Center at start time")
    print(f"  Azimuth:  {fmt_deg(gc_az[0])} deg")
    print(f"  Altitude: {fmt_deg(gc_alt[0])} deg")
    print(f"  Clear for full window: {'YES' if bool(np.all(gc_ok)) else 'NO'}")
    print()

    if best is None:
        print("Best Milky Way plane pointing")
        print("  None found that stays clear for the full window given your limits.")
        print("  Try a different start time, reduce min_alt limits, or increase blocked sector realism.")
        return

    print("Best Milky Way plane pointing for the full window")
    print(f"  Point dish at start time to")
    print(f"    Azimuth:  {fmt_deg(best['start_az'])} deg")
    print(f"    Altitude: {fmt_deg(best['start_alt'])} deg")
    print()
    print("Window behavior for that pointing")
    print(f"  End azimuth:  {fmt_deg(best['end_az'])} deg")
    print(f"  End altitude: {fmt_deg(best['end_alt'])} deg")
    print(f"  Minimum altitude during window: {fmt_deg(best['min_alt'])} deg")
    print(f"  Worst case clearance margin: {fmt_deg(best['score'])} deg")
    print()
    print("Notes")
    print("  Azimuth is compass degrees. 0 north. 90 east. 180 south. 270 west.")
    print("  Altitude is degrees above horizon. Negative means below horizon.")


if __name__ == "__main__":
    main()
