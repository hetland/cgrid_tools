import json

import numpy as np
import pytest

from cgrid_tools import GeoNotConfiguredError, Grid


def _vertices(shape: tuple[int, int] = (4, 5)) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[: shape[0], : shape[1]]
    return x.astype(float), y.astype(float)


def test_grid_basic_construction() -> None:
    x, y = _vertices()

    grid = Grid(x, y, options={"mode": "test"})

    assert grid.shape == x.shape
    assert np.array_equal(grid.x, x)
    assert np.array_equal(grid.y, y)
    assert grid.mask_rho.shape == (x.shape[0] - 1, x.shape[1] - 1)
    assert np.all(grid.mask_rho)
    assert grid.options == {"mode": "test"}


def test_grid_mask_rho_derived_from_nan_vertices() -> None:
    x, y = _vertices()
    x[1, 2] = np.nan
    y[1, 2] = np.nan

    grid = Grid(x, y)

    assert not grid.mask_rho[0, 1]
    assert not grid.mask_rho[0, 2]
    assert not grid.mask_rho[1, 1]
    assert not grid.mask_rho[1, 2]
    assert np.count_nonzero(~grid.mask_rho) == 4


def test_grid_immutability_with_mask() -> None:
    x, y = _vertices()
    grid = Grid(x, y)
    masked = grid.with_mask(np.array([[1.25, 0.25], [2.75, 0.25], [2.75, 1.75], [1.25, 1.75]]))

    assert masked is not grid
    assert np.all(grid.mask_rho)
    assert np.any(~masked.mask_rho)


def test_grid_replace_returns_new_instance() -> None:
    x, y = _vertices()
    grid = Grid(x, y, options={"mode": "old"})

    replaced = grid.replace(options={"mode": "new"})

    assert replaced is not grid
    assert grid.options == {"mode": "old"}
    assert replaced.options == {"mode": "new"}
    assert np.array_equal(replaced.x, grid.x)


def test_grid_options_dict_is_copy() -> None:
    x, y = _vertices()
    original = {"nested": {"value": 1}}
    grid = Grid(x, y, options=original)

    original["nested"]["value"] = 2
    extracted = grid.options
    extracted["nested"]["value"] = 3

    assert grid.options == {"nested": {"value": 1}}


def test_grid_options_round_trip_through_json() -> None:
    x, y = _vertices()
    options = {"mode": "transfinite", "shape": [4, 5], "iterations": 0}
    grid = Grid(x, y, options=options)

    encoded = json.dumps(grid.to_dict())
    decoded = json.loads(encoded)

    round_tripped = Grid(
        np.asarray(decoded["x_vert"]),
        np.asarray(decoded["y_vert"]),
        mask_rho=np.asarray(decoded["mask_rho"]),
        crs=decoded["crs"],
        options=decoded["options"],
    )
    assert round_tripped == grid


def test_grid_to_xarray_returns_copy_not_reference() -> None:
    x, y = _vertices()
    grid = Grid(x, y)

    ds = grid.to_xarray()
    ds["x_vert"].values[0, 0] = 999.0

    assert grid.x[0, 0] != 999.0


def test_grid_to_netcdf_round_trip(tmp_path) -> None:
    x, y = _vertices()
    mask = np.ones((x.shape[0] - 1, x.shape[1] - 1), dtype=bool)
    mask[1, 2] = False
    path = tmp_path / "grid.nc"
    grid = Grid(x, y, mask_rho=mask, options={"mode": "round-trip"})

    grid.to_netcdf(path)
    loaded = Grid.from_netcdf(path)

    assert loaded == grid


def test_grid_geo_property_raises_when_unset() -> None:
    x, y = _vertices()
    grid = Grid(x, y)

    with pytest.raises(GeoNotConfiguredError):
        _ = grid.f


def test_grid_repr_contains_shape() -> None:
    x, y = _vertices()
    grid = Grid(x, y, options={"mode": "repr-mode"})

    text = repr(grid)

    assert "Grid(shape=(4, 5)" in text
    assert "mode='repr-mode'" in text


@pytest.mark.parametrize(
    "name",
    [
        "x",
        "y",
        "mask_rho",
        "lon",
        "lat",
        "dx",
        "dy",
        "pm",
        "pn",
        "angle",
        "area",
        "f",
        "mask_u",
        "mask_v",
        "mask_psi",
    ],
)
def test_grid_arrays_and_metrics_cannot_be_mutated(name):
    x, y = _vertices()
    grid = Grid(x, y, lon_vert=x, lat_vert=y)
    array = getattr(grid, name)
    with pytest.raises(ValueError):
        array.flat[0] = 42
    with pytest.raises(ValueError):
        array.setflags(write=True)
    with pytest.raises(AttributeError, match="immutable"):
        setattr(grid, name, np.zeros_like(array))
    with pytest.raises(AttributeError, match="immutable"):
        delattr(grid, name)


def test_xarray_roundtrip_respects_named_dimensions():
    grid = Grid(*_vertices())
    assert Grid.from_xarray(grid.to_xarray().transpose()) == grid
    ds = grid.to_xarray().rename({"xi_vert": "wrong"})
    with pytest.raises(ValueError, match="dimensions"):
        Grid.from_xarray(ds)


def test_coriolis_is_at_rho_points():
    from cgrid_tools.geo import coriolis

    x, y = _vertices()
    grid = Grid(x, y, lon_vert=x, lat_vert=y + 30)
    assert grid.f.shape == grid.mask_rho.shape
    assert np.allclose(grid.f[:, 0], coriolis(np.array([30.5, 31.5, 32.5])))


def test_strict_json_roundtrip_handles_masked_vertices():
    from cgrid_tools.io import dict_from_grid, grid_from_dict

    x, y = _vertices()
    x[1, 1] = y[1, 1] = np.nan
    grid = Grid(x, y, lon_vert=x, lat_vert=y)
    for data in (grid.to_dict(), dict_from_grid(grid)):
        assert data["x_vert"][1][1] is None
        encoded = json.dumps(data, allow_nan=False)
        assert grid_from_dict(json.loads(encoded)) == grid


def test_boundary_arrays_are_owned_and_immutable():
    from cgrid_tools.gridgen import BoundaryDefinition

    x, y = np.array([0.0, 1.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    beta = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    boundary = BoundaryDefinition(x, y, beta)
    x[1] = 99
    beta[0] = 99
    assert boundary.x[1] == boundary.beta[0] == 1
    with pytest.raises(ValueError):
        boundary.beta[0] = 99
