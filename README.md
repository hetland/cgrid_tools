# cgrid-tools

Generate two-dimensional curvilinear C-grids from polygon boundaries for ocean
models such as REMORA and ROMS. The package provides coordinates, land-water
masks, grid metrics, and NetCDF serialization.

***NOTE: THIS PACKAGE IS STILL UNDER DEVELOPMENT AND HAS NOT BEEN THOUROUGHLY TESTED***

## Install

Python 3.10 or later is required. Install from this checkout:

```bash
python -m pip install .
```

Use `python -m pip install ".[geo]"` for longitude/latitude projection support.
For development, install `".[dev,geo,docs]"` with `-e`.

## First grid

```python
from cgrid_tools import Grid
from cgrid_tools.gridgen import BoundaryDefinition, GridGenerator, SolverMode

boundary = BoundaryDefinition(
    x=[0, 10, 10, 0],
    y=[0, 0, 2, 2],
    beta=[1, 1, 1, 1],
)
grid = GridGenerator(mode=SolverMode.TRANSFINITE).generate(boundary, (11, 31))
assert grid.shape == (11, 31)       # vertices: (ny, nx)
assert grid.pm.shape == (10, 30)    # cell centers (rho points)
grid.to_netcdf("grid.nc")
assert Grid.from_netcdf("grid.nc") == grid
```

`beta` describes logical turns in quarter turns: `1` for a rectangular corner,
`0` along a side; the sum must be four. Coordinates use consistent Cartesian
units. Grid arrays are immutable; use `grid.replace(...)` or `grid.with_mask(...)`
to make a modified grid.

## Choose a solver

| Mode | Use |
| --- | --- |
| `TRANSFINITE` | Fast interpolation between four logical sides |
| `LAPLACIAN` | Linear smoothing of a four-sided grid |
| `ELLIPTIC` | Nonlinear smoothing of a four-sided grid |
| `GRIDGEN_CRDT` | Conformal mapping, including reentrant beta walks |

This is a pre-1.0 package. Unsupported topology and folded or degenerate active
cells raise errors. Difficult conformal maps can fail to converge; inspect
`grid.diagnostics` and grid quality before use in a model. Bathymetry and complete
model configuration are outside the generator's scope.

See the [quickstart](docs/quickstart.md), [examples](examples/README.md),
[metrics](docs/metrics.md), and [solver notes](docs/crdt_notes.md).

## License and credits

Original contributions use the MIT license. Adaptations from gridgen-c, octant,
pygridgen, and SCPACK retain their notices in [LICENSE](LICENSE).
The CRDT method is due to Driscoll and Vavasis; this implementation follows
Sakov's gridgen-c. Triangulation uses SciPy/Qhull; Triangle is not bundled or
required. See [provenance](docs/provenance.md).
