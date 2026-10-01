# Quickstart

From a checkout, install with `python -m pip install ".[geo]"`.
The [README](https://github.com/hetland/cgrid-tools#readme) shows a minimal Cartesian grid.

## Focus grid lines

```python
import numpy as np
from cgrid_tools import orthogonality_summary
from cgrid_tools.gridgen import BoundaryDefinition, Focus, GridGenerator, SolverMode

boundary = BoundaryDefinition([0, 10000, 10000, 0], [0, 0, 2000, 2000], [1, 1, 1, 1])
focus = Focus()
focus.add_focus(pos=0.2, axis="x", factor=3, extent=0.1)
grid = GridGenerator(mode=SolverMode.ELLIPTIC, focus=focus).generate(boundary, (13, 41))
assert grid.mask_rho.shape == (12, 40)
print(np.degrees(orthogonality_summary(grid.x, grid.y)["mean_abs"]))
```

Focus positions and extents use computational coordinates from zero to one.
Use `axis="y"` to cluster rows. `factor > 1` concentrates grid lines.

## Geographic boundary

Supply a projected target CRS appropriate for the domain. The solver works in
projected coordinates and inverse-projects every generated vertex to lon/lat.

```python
from cgrid_tools.gridgen import BoundaryDefinition, GridGenerator, SolverMode
from cgrid_tools.io import roms_grid_from_grid

boundary = BoundaryDefinition.from_lonlat(
    lon=[36, 37, 37, 36], lat=[35, 35, 36, 36], beta=[1, 1, 1, 1],
    target_crs="EPSG:32636",
)
grid = GridGenerator(mode=SolverMode.TRANSFINITE).generate(boundary, (11, 21))
assert grid.lon.shape == grid.shape
assert grid.f.shape == grid.mask_rho.shape
assert grid.crs == "EPSG:32636"
dataset = roms_grid_from_grid(grid)
assert dataset.attrs["spherical"] == "T"
```

A `pyproj.Proj` instance may also be passed as `projection`. Custom forward
callables require `inverse_projection` for geographic generation. `source_crs`
identifies the input geographic CRS (default EPSG:4326); `target_crs` identifies
projected x/y. Metrics use projected lengths and angles relative to the projected
x-axis. They are not automatically geodesic distances or bearings relative to
true east; `geo.great_circle_dx/dy` provide separate edge-distance helpers.

## Masks and export

`grid.with_mask(polygon)` returns a new grid with cell centers inside the polygon
marked as land. `grid.replace(...)` replaces constructor fields on a new grid.
`grid.to_xarray()` returns an editable copy; `Grid.from_xarray()` restores the grid
using named dimensions. `Grid.to_netcdf()` / `Grid.from_netcdf()` round-trip this
representation.

`roms_grid_from_grid()` exports ROMS-shaped variables. Its `h` is a placeholder
that must be filled with bathymetry; Cartesian `f` is also NaN. The result is not
a complete model-ready grid file. JSON serialization represents NaN vertices as
`null`; `io.grid_from_dict()` restores them as NaN.
