from __future__ import annotations

import numpy as np

OMEGA = 7.2921159e-5  # Earth rotation rate, rad/s


def _require_pyproj():
    """Lazy import pyproj; raise GeoNotConfiguredError with install hint if absent."""
    try:
        import pyproj

        return pyproj
    except ImportError as exc:
        from cgrid_tools.grid import GeoNotConfiguredError

        raise GeoNotConfiguredError(
            "This function requires pyproj. Install it with: pip install cgrid-tools[geo]"
        ) from exc


def coriolis(lat: np.ndarray) -> np.ndarray:
    """Coriolis parameter f = 2 * Omega * sin(lat).

    Works without pyproj. lat in degrees.
    """
    return 2.0 * OMEGA * np.sin(np.deg2rad(np.asarray(lat, dtype=float)))


def great_circle_dx(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Great-circle cell width along xi (columns) direction, in metres.

    Output shape: same as (ny, nx-1) i.e. one fewer column than input.
    Requires pyproj.
    """
    pyproj = _require_pyproj()
    geod = pyproj.Geod(ellps="WGS84")
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    _, _, dist = geod.inv(lon[:, :-1], lat[:, :-1], lon[:, 1:], lat[:, 1:])
    return np.abs(dist)


def great_circle_dy(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Great-circle cell height along eta (rows) direction, in metres.

    Output shape: (ny-1, nx).
    Requires pyproj.
    """
    pyproj = _require_pyproj()
    geod = pyproj.Geod(ellps="WGS84")
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    _, _, dist = geod.inv(lon[:-1, :], lat[:-1, :], lon[1:, :], lat[1:, :])
    return np.abs(dist)


def project_lonlat_to_xy(
    lon: np.ndarray, lat: np.ndarray, crs: str
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat (degrees) to x/y using pyproj CRS string.

    Requires pyproj.
    """
    pyproj = _require_pyproj()
    proj = pyproj.Proj(crs)
    return proj(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))


def project_xy_to_lonlat(x: np.ndarray, y: np.ndarray, crs: str) -> tuple[np.ndarray, np.ndarray]:
    """Inverse project x/y to lon/lat (degrees).

    Requires pyproj.
    """
    pyproj = _require_pyproj()
    proj = pyproj.Proj(crs)
    return proj(np.asarray(x, dtype=float), np.asarray(y, dtype=float), inverse=True)


def geographic_rho_points(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rho coordinates in degrees, with longitude averaged across its branch cut.

    Each cell must span less than 180 degrees in longitude. Latitude uses
    the same four-vertex average as the planar rho convention.
    """
    from cgrid_tools.metrics import rho_points

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.ndim != 2 or lon.shape != lat.shape or min(lon.shape) < 2:
        raise ValueError("lon and lat must have identical 2D vertex shapes")
    corners = np.stack((lon[:-1, :-1], lon[:-1, 1:], lon[1:, :-1], lon[1:, 1:]))
    origin = corners[0]
    offsets = (corners - origin + 180.0) % 360.0 - 180.0
    if np.any(np.ptp(offsets, axis=0) >= 180.0):
        raise ValueError("geographic cells must span less than 180 degrees longitude")
    lon_rho = (origin + np.mean(offsets, axis=0) + 180.0) % 360.0 - 180.0
    _, lat_rho = rho_points(lon, lat)
    return lon_rho, lat_rho
