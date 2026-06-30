"""
Geocoding utility — forward address lookup via Nominatim (OpenStreetMap).

No API key required.  Rate limit: 1 request/second per Nominatim policy.
"""
from __future__ import annotations

import json
import logging
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass

_log = logging.getLogger(__name__)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT    = "RadioTelescopePlanner/1.0 (personal hobby project)"
_TIMEOUT_S     = 10


@dataclass
class GeocodedLocation:
    lat: float
    lon: float
    display_name: str


def _ssl_context() -> ssl.SSLContext:
    """Return the best available SSL context for this platform.

    macOS Python installations often lack bundled CA certificates.
    We prefer certifi when installed; otherwise fall back to an unverified
    context.  This is acceptable because we only ever contact the
    well-known nominatim.openstreetmap.org host.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl._create_unverified_context()


def lookup_address(address: str) -> GeocodedLocation:
    """Geocode *address* and return the top result.

    Args:
        address: Free-form address string, e.g. "700 White Tail Dr, Gahanna OH".

    Returns:
        GeocodedLocation with lat, lon, and the resolved display name.

    Raises:
        ValueError: If the address returns no results.
        OSError:    On network / SSL errors (re-raised from urllib).
    """
    url = _NOMINATIM_URL + "?" + urllib.parse.urlencode({
        "q":      address,
        "format": "json",
        "limit":  1,
    })
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    _log.debug("Geocoding: %s", address)
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S, context=_ssl_context()) as resp:
        results = json.loads(resp.read().decode())

    if not results:
        raise ValueError(f"Address not found: {address!r}")

    hit = results[0]
    loc = GeocodedLocation(
        lat=float(hit["lat"]),
        lon=float(hit["lon"]),
        display_name=hit.get("display_name", address),
    )
    _log.info("Geocoded %r → lat=%.6f lon=%.6f", address, loc.lat, loc.lon)
    return loc
