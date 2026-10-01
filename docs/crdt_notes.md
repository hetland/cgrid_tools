# CRDT solver notes

The CRDT method of Driscoll and Vavasis uses **cross-ratios and Delaunay
triangulation** for numerical conformal mapping. This Python implementation
follows Pavel Sakov's gridgen-c; see [provenance](provenance.md).

The pipeline builds constrained triangulations with SciPy/Qhull and Python edge
recovery, solves cross-ratio parameters, constructs the image polygon, and
inverse-maps a sampling grid to the physical polygon. Triangle is not required.

```python
from cgrid_tools.gridgen import BoundaryDefinition, CRDTOptions, GridGenerator, SolverMode

boundary = BoundaryDefinition([0, 2, 2, 0], [0, 0, 1, 1], [1, 1, 1, 1])
grid = GridGenerator(
    mode=SolverMode.GRIDGEN_CRDT,
    crdt_options=CRDTOptions(nnodes=10, precision=1e-10, max_iterations=100),
).generate(boundary, (9, 17))
assert grid.diagnostics["converged"]
assert grid.diagnostics["mapping"]["sc_map_failures"] == 0
```

## Options and diagnostics

`nnodes` controls Gauss-Jacobi quadrature; `precision` is the sigma-residual
target. `newton=True` selects SciPy's hybrid solver; `False` selects Broyden.
The hybrid solver uses a finite-difference scale that accounts for quadrature
noise instead of assuming machine-precision residual evaluations.
`nppe` controls sampling of image-region edges used to locate inverse maps.

By default, boundary beta markers specify the image polygon. `image_beta`
overrides these markers; `image_beta_blend` in `[0, 1]` blends toward physically
matched corner angles. Changes can improve spacing near corners but also change
the computational domain and its exclusions.

`grid.diagnostics` and its serialized copy in `grid.options` contain sigma
residual, convergence status, image-vertex mismatch, and mapping statistics:
failed maps, continuation/fallback counts, candidate attempts, and the largest
inverse residual. Inspect `solve_crdt_state()` and `CRDTSolverState.diagnostics`
for additional corner-angle and embedding details.

## Failure behavior

Sigma residuals are checked regardless of the optimizer's success flag.
Residuals above `max(100 * precision, 1e-8)` raise `CRDTError`; residuals above
the requested target but within that ceiling warn and set `converged=False`.
A node that cannot be mapped raises `CRDTError`. Folded or degenerate active
cells raise `GridGenerationError`; they are never silently converted to land.
These checks do not guarantee suitability for every model or topology.

Thin domains and crowded corners can exceed double-precision accuracy. Increasing
quadrature nodes or revising the boundary may help; a smaller requested tolerance
alone cannot guarantee a more accurate map. Reference example 3 has historically
needed much looser geometric comparisons than the other fixtures.

Image alignment only adjusts non-corner vertices on eligible axis-aligned walks.
Fractional walks that do not meet the alignment condition preserve those vertices.
