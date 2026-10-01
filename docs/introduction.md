# Choosing a grid

`GridGenerator` turns a polygon into a `Grid`. `Grid` owns immutable coordinate
arrays, masks, and cached metrics; exported datasets are independent copies.
All generation shapes mean **vertex rows and columns**, not numbers of cells.

For transfinite, Laplacian, and elliptic generation, mark four logical corners
with `beta=1` and any additional side vertices with `beta=0`. The sides must be
contiguous and form a simple polygon. Clockwise input is normalized. The fast
transfinite result is a useful starting point; smoothing may improve it but
does not guarantee low orthogonality error.

Use `GRIDGEN_CRDT` for reentrant or fractional beta walks. It maps a computational
image polygon into the physical domain using Schwarz-Christoffel transforms.
Parts of its rectangular sampling array can lie outside the image polygon;
these vertices are NaN and adjacent cells are masked.

All solvers reject detected folded or degenerate active cells. Four-side solvers
reject unsupported topology instead of joining disconnected sides. The conformal
solver raises on mapping failures or excessive sigma residuals. Very thin or
crowded domains may require different boundary vertices, quadrature settings,
or resolution. Review the resulting grid, masks, and diagnostics before running
a model.

Continue with the [quickstart](quickstart.md), [examples](examples/index.md),
[metrics](metrics.md), or [CRDT notes](crdt_notes.md).
