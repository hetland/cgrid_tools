# Metrics and array shapes

`GridGenerator.generate(boundary, shape=(ny, nx))` returns a `Grid` whose `x` and
`y` arrays contain vertices. With `Mp=ny-1` and `Lp=nx-1`:

| Location or property | Shape |
| --- | --- |
| Vertices; `grid.x`, `grid.y`, `grid.angle` | `(Mp+1, Lp+1)` |
| Cell centers (rho); `dx`, `dy`, `pm`, `pn`, `area`, `mask_rho`, `f` | `(Mp, Lp)` |
| u; `mask_u` | `(Mp, Lp-1)` |
| v; `mask_v` | `(Mp-1, Lp)` |
| psi; `mask_psi` | `(Mp-1, Lp-1)` |

ROMS interior rho points exclude the outer rho rows/columns: `[1:-1, 1:-1]`.
`f` requires geographic vertices and uses mean cell latitude. `pm=1/dx` and
`pn=1/dy`; lengths have the same units as x/y. Geographic inputs do not change
this planar metric convention.

```python
import numpy as np
from cgrid_tools import Grid, cell_jacobians, orthogonality_summary, rho_points

y, x = np.mgrid[:4, :6].astype(float)
grid = Grid(x, y)
x_rho, y_rho = rho_points(grid.x, grid.y)
assert x_rho.shape == (3, 5)
assert np.all(cell_jacobians(grid.x, grid.y) > 0)
assert orthogonality_summary(grid.x, grid.y)["max_abs"] == 0
```

`rho_points`, `u_points`, `v_points`, and `psi_points` return coordinate pairs.
The free functions `cell_widths`, `metric_coefficients`, `metric_derivatives`,
`cell_areas`, and `grid_angle` accept vertex arrays. `Grid` caches its exposed
properties; use free functions for other diagnostics.

`grid_angle` returns radians relative to the projected x-axis. It computes
rho-point directions, then pads the last row/column to preserve the historical
vertex-shaped API. The ROMS exporter removes this padding. Unwrapping acts
within finite runs so NaN holes do not contaminate entire rows or columns.

`orthogonality_error` is the mean absolute corner-direction deviation from 90
degrees, in radians. `orthogonality_signed` preserves the octant directional
sign convention; it is not an average of geometric interior angles.
`orthogonality_summary` provides `mean_abs`, `max_abs`, `rms`, and `signed_mean`.
Absolute `cell_areas` cannot detect orientation reversal. Use `cell_jacobians`
(shape `(4, Mp, Lp)`, four bilinear-map corner determinants per cell) to detect
folds and degeneracy.

## Masks

`True` means water. `Grid` intersects a supplied rho mask with the finite-vertex
mask. `mask_u`, `mask_v`, and `mask_psi` require their neighboring rho cells to
be water. Derived metrics follow coordinates; a manually masked finite cell
still has finite metrics, so apply the mask when computing water-only statistics.

Four-side generators recompute the center-in-polygon mask after smoothing.
CRDT masks cells adjacent to excluded image-plane vertices. A failed inverse
map is an error, not an additional land mask. Both paths validate every active
cell before returning. Constructing a `Grid` directly validates array shapes;
it does not run the generator's geometric quality checks.
