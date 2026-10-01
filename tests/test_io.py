import json

import numpy as np

from cgrid_tools import Grid
from cgrid_tools.io import dict_from_grid, grid_from_dict, roms_grid_from_grid


def _grid() -> Grid:
    y, x = np.mgrid[:4, :5]
    return Grid(x.astype(float), y.astype(float), options={"mode": "io-test"})


def test_dict_round_trip_is_json_serializable() -> None:
    grid = _grid()

    payload = json.loads(json.dumps(dict_from_grid(grid)))
    round_tripped = grid_from_dict(payload)

    assert round_tripped == grid


def test_roms_grid_from_grid_has_required_variables() -> None:
    ds = roms_grid_from_grid(_grid())

    required = {
        "x_rho",
        "y_rho",
        "pm",
        "pn",
        "mask_rho",
        "mask_u",
        "mask_v",
        "mask_psi",
        "angle",
        "h",
        "f",
    }

    assert required.issubset(ds.data_vars)
    assert ds.attrs["spherical"] == "F"


def test_roms_grid_from_grid_variable_shapes() -> None:
    ds = roms_grid_from_grid(_grid())

    ny, nx = 3, 4

    assert ds["x_rho"].shape == (ny, nx)
    assert ds["y_rho"].shape == (ny, nx)
    assert ds["pm"].shape == (ny, nx)
    assert ds["pn"].shape == (ny, nx)
    assert ds["mask_rho"].shape == (ny, nx)
    assert ds["mask_u"].shape == (ny, nx - 1)
    assert ds["mask_v"].shape == (ny - 1, nx)
    assert ds["mask_psi"].shape == (ny - 1, nx - 1)
    assert ds["angle"].shape == (ny, nx)


def test_roms_geographic_export_handles_dateline_and_matches_coriolis():
    y, x = np.mgrid[:2, :2].astype(float)
    grid = Grid(x, y, lon_vert=np.array([[179, -179], [179, -179]]), lat_vert=y + 30)
    ds = roms_grid_from_grid(grid)
    assert np.isclose(abs(ds.lon_rho.item()), 180)
    assert np.allclose(ds.f, grid.f)
    assert ds.attrs["spherical"] == "T"
