
from datetime import datetime
import numpy as np

def julian_day(dt_utc_naive: datetime) -> float:
    """Calculate Julian Day from UTC datetime"""
    y = dt_utc_naive.year
    m = dt_utc_naive.month
    d = dt_utc_naive.day + (dt_utc_naive.hour + (dt_utc_naive.minute + dt_utc_naive.second / 60.0) / 60.0) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    A = int(y / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return float(jd)

def gmst_degrees(dt_utc_naive: datetime) -> float:
    """Calculate Greenwich Mean Sidereal Time in degrees"""
    jd = julian_day(dt_utc_naive)
    T = (jd - 2451545.0) / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T * T
        - (T * T * T) / 38710000.0
    )
    return float(gmst % 360.0)

def local_sidereal_time(dt_utc, longitude_deg: float) -> float:
    """
    Calculate Local Sidereal Time for a given UTC datetime and longitude.
    
    Args:
        dt_utc: datetime object (timezone-aware or naive UTC)
        longitude_deg: Observer longitude in degrees (East positive)
    
    Returns:
        Local sidereal time in degrees
    """
    # Remove timezone info if present
    if hasattr(dt_utc, 'tzinfo') and dt_utc.tzinfo is not None:
        dt_naive = dt_utc.replace(tzinfo=None)
    else:
        dt_naive = dt_utc
    
    gmst = gmst_degrees(dt_naive)
    lst = (gmst + longitude_deg) % 360.0
    return float(lst)

def astronomical_time_components(dt_utc, longitude_deg: float) -> dict:
    """
    Get all astronomical time components at once.
    
    Args:
        dt_utc: datetime object (timezone-aware or naive UTC)
        longitude_deg: Observer longitude in degrees
    
    Returns:
        Dictionary with keys: 'jd', 'gmst', 'lst'
    """
    # Remove timezone info if present
    if hasattr(dt_utc, 'tzinfo') and dt_utc.tzinfo is not None:
        dt_naive = dt_utc.replace(tzinfo=None)
    else:
        dt_naive = dt_utc
    
    jd = julian_day(dt_naive)
    gmst = gmst_degrees(dt_naive)
    lst = (gmst + longitude_deg) % 360.0
    
    return {
        'jd': float(jd),
        'gmst': float(gmst),
        'lst': float(lst)
    }

