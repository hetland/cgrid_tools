"""Serialization helpers for cgrid_tools Grid objects.

Provides dict/JSON round-trip serialization and xarray.Dataset export
following ROMS grid variable conventions.

"""

from __future__ import annotations

import numpy as np
import xarray as xr


def dict_from_grid(grid) -> dict:
    """Convert a Grid to a JSON-safe nested dict.

    The returned dict contains ``x_vert``, ``y_vert``, ``mask_rho`` as
    nested lists (NaN coordinates become JSON null); optional
    ``lon_vert``, ``lat_vert`` are omitted when unset; a ``crs`` string; and the ``options`` dict.

    Parameters
    ----------
    grid : cgrid_tools.grid.Grid
        Source grid object.

    Returns
    -------
    dict
        JSON-serializable representation that round-trips via
        :func:`grid_from_dict`.
    """
    return grid.to_dict()


def grid_from_dict(d: dict):
    """Reconstruct a Grid from a dict produced by :func:`dict_from_grid`.

    Parameters
    ----------
    d : dict
        Dict previously returned by :func:`dict_from_grid`.

    Returns
    -------
    cgrid_tools.grid.Grid
    """
    from cgrid_tools.grid import Grid

    lon_raw = d.get("lon_vert")
    lat_raw = d.get("lat_vert")
    mask_raw = d.get("mask_rho")

    return Grid(
        x_vert=np.array(d["x_vert"], dtype=float),
        y_vert=np.array(d["y_vert"], dtype=float),
        mask_rho=np.array(mask_raw, dtype=bool) if mask_raw is not None else None,
        lon_vert=np.array(lon_raw, dtype=float) if lon_raw is not None else None,
        lat_vert=np.array(lat_raw, dtype=float) if lat_raw is not None else None,
        crs=d.get("crs"),
        options=d.get("options"),
    )


def roms_grid_from_grid(grid) -> xr.Dataset:
    """Build an ``xarray.Dataset`` matching ROMS grid variable conventions.

    Variables always present
    ------------------------
    ``x_rho``, ``y_rho``, ``pm``, ``pn``, ``mask_rho``, ``mask_u``,
    ``mask_v``, ``mask_psi``, ``angle``, ``h``, ``f``.

    Additional variables when ``grid.lat`` is set
    ----------------------------------------------
    ``lon_rho``, ``lat_rho``.  ``f`` is computed from latitude via
    :func:`cgrid_tools.geo.coriolis`; otherwise it is filled with NaN.

    ``h`` (bathymetry) is always filled with NaN and must be set
    separately by the caller.

    Parameters
    ----------
    grid : cgrid_tools.grid.Grid
        Source grid object.

    Returns
    -------
    xr.Dataset
        Dataset with ROMS-convention dimension names and a ``spherical``
        global attribute (``"F"`` for Cartesian, ``"T"`` for
        lat/lon grids).
    """
    from cgrid_tools.metrics import (
        grid_angle,
        metric_coefficients,
        rho_points,
        uvp_masks,
    )

    x_rho, y_rho = rho_points(grid.x, grid.y)
    pm, pn = metric_coefficients(grid.x, grid.y)
    mask_u, mask_v, mask_psi = uvp_masks(grid.mask_rho)
    angle_vert = grid_angle(grid.x, grid.y, unwrap=True)
    angle = angle_vert[:-1, :-1]  # trim to rho-point shape (Mp, Lp)

    ny, nx = grid.mask_rho.shape

    ds = xr.Dataset(
        {
            "x_rho": (["eta_rho", "xi_rho"], x_rho),
            "y_rho": (["eta_rho", "xi_rho"], y_rho),
            "pm": (["eta_rho", "xi_rho"], pm),
            "pn": (["eta_rho", "xi_rho"], pn),
            "mask_rho": (["eta_rho", "xi_rho"], grid.mask_rho.astype(float)),
            "mask_u": (["eta_rho", "xi_u"], mask_u.astype(float)),
            "mask_v": (["eta_v", "xi_rho"], mask_v.astype(float)),
            "mask_psi": (["eta_psi", "xi_psi"], mask_psi.astype(float)),
            "angle": (["eta_rho", "xi_rho"], angle),
            "h": (["eta_rho", "xi_rho"], np.full((ny, nx), np.nan)),
        }
    )

    if grid.lat is not None:
        from cgrid_tools.geo import geographic_rho_points

        lon_rho, lat_rho = geographic_rho_points(grid.lon, grid.lat)
        ds["lon_rho"] = (["eta_rho", "xi_rho"], lon_rho)
        ds["lat_rho"] = (["eta_rho", "xi_rho"], lat_rho)
        from cgrid_tools.geo import coriolis

        ds["f"] = (["eta_rho", "xi_rho"], coriolis(lat_rho))
        ds.attrs["spherical"] = "T"
    else:
        ds["f"] = (["eta_rho", "xi_rho"], np.full((ny, nx), np.nan))
        ds.attrs["spherical"] = "F"

    ds.attrs["created_by"] = "cgrid_tools"

    return ds
