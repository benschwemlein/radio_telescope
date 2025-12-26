#!/usr/bin/env python3
"""
milkyway_pointing.py

Given your Earth coordinates and a time, compute:
1) The azimuth and altitude of the Galactic Center
2) The azimuth and altitude of the Galactic Anti Center
3) The single best pointing on the Galactic plane (b=0) at that time
   defined as the point on the plane with the highest altitude above your horizon

Usage:
  python milkyway_pointing.py 39.9612 -82.9988 "2025-12-26 23:30:00" --tz "America/New_York"

Notes:
- Requires astropy and pytz:
    pip install astropy pytz
"""

import argparse
from datetime import datetime
import math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lat_deg", type=float, help="Latitude in degrees, north positive")
    ap.add_argument("lon_deg", type=float, help="Longitude in degrees, east positive (west is negative)")
    ap.add_argument("local_time", type=str, help='Local time like "2025-12-26 23:30:00"')
    ap.add_argument("--tz", type=str, default="America/New_York", help="IANA timezone name")
    ap.add_argument("--height_m", type=float, default=0.0, help="Observer height in meters")
    ap.add_argument("--step_deg", type=float, default=5.0, help="Sampling step for galactic longitude search")
    args = ap.parse_args()

    try:
        import pytz
        from astropy.time import Time
        from astropy.coordinates import EarthLocation, SkyCoord, AltAz
        import astropy.units as u
    except Exception as e:
        raise SystemExit(
            "Missing dependency. Install with:\n"
            "  pip install astropy pytz\n\n"
            f"Import error: {e}"
        )

    tz = pytz.timezone(args.tz)
    dt_local_naive = datetime.strptime(args.local_time, "%Y-%m-%d %H:%M:%S")
    dt_local = tz.localize(dt_local_naive)
    dt_utc = dt_local.astimezone(pytz.utc)

    location = EarthLocation(lat=args.lat_deg * u.deg, lon=args.lon_deg * u.deg, height=args.height_m * u.m)
    obstime = Time(dt_utc)

    frame = AltAz(obstime=obstime, location=location)

    # Galactic Center and Anti Center in galactic coordinates
    gc = SkyCoord(l=0 * u.deg, b=0 * u.deg, frame="galactic").transform_to(frame)
    ac = SkyCoord(l=180 * u.deg, b=0 * u.deg, frame="galactic").transform_to(frame)

    # Find best point on the galactic plane at this time: maximize altitude over l in [0,360)
    best = None
    step = args.step_deg
    l = 0.0
    while l < 360.0 - 1e-9:
        p = SkyCoord(l=l * u.deg, b=0 * u.deg, frame="galactic").transform_to(frame)
        alt = float(p.alt.to_value(u.deg))
        az = float(p.az.to_value(u.deg))
        if best is None or alt > best["alt_deg"]:
            best = {"l_deg": l, "alt_deg": alt, "az_deg": az}
        l += step

    def fmt_deg(x):
        return f"{x:.1f}"

    def above_horizon(alt_deg):
        return "YES" if alt_deg > 0.0 else "NO"

    print("Input")
    print(f"  Location lat, lon: {args.lat_deg:.6f}, {args.lon_deg:.6f}")
    print(f"  Local time:        {dt_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  UTC time:          {dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    gc_alt = float(gc.alt.to_value(u.deg))
    gc_az = float(gc.az.to_value(u.deg))
    print("Galactic Center (l=0, b=0)")
    print(f"  Altitude: {fmt_deg(gc_alt)} deg   Above horizon: {above_horizon(gc_alt)}")
    print(f"  Azimuth:  {fmt_deg(gc_az)} deg")
    print()

    ac_alt = float(ac.alt.to_value(u.deg))
    ac_az = float(ac.az.to_value(u.deg))
    print("Galactic Anti Center (l=180, b=0)")
    print(f"  Altitude: {fmt_deg(ac_alt)} deg   Above horizon: {above_horizon(ac_alt)}")
    print(f"  Azimuth:  {fmt_deg(ac_az)} deg")
    print()

    print("Best Galactic plane pointing right now (max altitude over b=0)")
    print(f"  Galactic longitude l: {best['l_deg']:.1f} deg")
    print(f"  Altitude:             {fmt_deg(best['alt_deg'])} deg   Above horizon: {above_horizon(best['alt_deg'])}")
    print(f"  Azimuth:              {fmt_deg(best['az_deg'])} deg")
    print()

    # Simple recommendation
    # Prefer GC if above horizon and reasonably high, else use the best plane point
    if gc_alt > 15.0:
        print("Recommendation")
        print("  Point at the Galactic Center now for strongest HI signal.")
        print(f"  Set azimuth {fmt_deg(gc_az)} deg, altitude {fmt_deg(gc_alt)} deg.")
    else:
        print("Recommendation")
        print("  Galactic Center is low or below the horizon at this time.")
        print("  Point at the highest part of the Galactic plane now.")
        print(f"  Set azimuth {fmt_deg(best['az_deg'])} deg, altitude {fmt_deg(best['alt_deg'])} deg.")

if __name__ == "__main__":
    main()
