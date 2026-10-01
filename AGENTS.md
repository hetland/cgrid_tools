# Repository guidelines

## Layout

- `src/cgrid_tools/grid.py`: immutable `Grid` container, cached metrics and serialization.
- `src/cgrid_tools/metrics.py`: Cartesian C-grid metric functions.
- `src/cgrid_tools/geo.py`: optional geographic helpers.
- `src/cgrid_tools/io.py`: dictionaries and ROMS-shaped xarray export.
- `src/cgrid_tools/gridgen/`: boundaries, focus, four-side solvers, CRDT,
  triangulation and shared geometry.
- `tests/`: subsystem tests and licensed gridgen-c fixtures.
- `examples/`: runnable public-API smoke examples.
- `docs/`: Sphinx/MyST guide, API and executed notebooks.

`GridResult`, `CGrid` and the old top-level generation modules have been removed.
Import generation classes from `cgrid_tools.gridgen`.

## Development

The conda environment in `env.yaml` is `remora-grid-tools`. Make targets use it
by default; override `CONDA_ENV=<name>` or use `RUN=` in an active environment.
A virtual environment with `pip install -e ".[dev,geo,docs]"` also works.

- `make test`: normal pytest suite.
- `make perf`: include optional timing checks (`RUN_PERFORMANCE=1`).
- `make format`: isort then Black (line length 100).
- `make check`: format checks, Ruff lint and tests.
- `make examples`: execute all Python examples.
- `make docs`: build documentation, treating Sphinx warnings as errors.
- `python -m build` and `twine check dist/*`: validate release artifacts.

CI checks Python 3.10-3.12 both with and without the geographic extra, builds
docs, and tests an extracted source distribution. Keep local tools and CI aligned.

## Code and tests

Use four spaces and standard Python naming. Prefer ASCII unless another character
serves a clear purpose. Keep changes focused; add a regression for a numerical
bug and a main path plus edge case for a new feature. Tests are organized by
subsystem, not necessarily one file per source module. Run appropriate checks
before handing off changes. Use imperative commit titles and concise PR summaries
with validation notes.

Keep the solver Python-first with NumPy/SciPy. The Numba elliptic kernel has a
Python reference for parity tests. Geographic wrappers and bathymetry or AMR
post-processing belong outside the Cartesian core solver.

## Numerical contracts

- Vertex shape is `(Mp+1, Lp+1)`; rho is `(Mp, Lp)` including boundary rows.
  Interior rho is `[1:-1, 1:-1]`. Coriolis `f` is rho-shaped; `angle` is
  vertex-shaped with a padded final row/column.
- `Grid` owns immutable arrays, including cached metrics. Return modified grids
  with `replace()` / `with_mask()`, or edit a copy from `to_xarray()`.
- Four-side solvers require four `beta=1` corners and optional `beta=0` side
  vertices. They reject reentrant/fractional walks; use CRDT for these.
- All generated active cells must have positive bilinear corner Jacobians.
  Failed mappings and folded cells are errors, never additional land masks.
- Judge CRDT convergence by measured residual, independently of optimizer flags.
  Preserve returned mapping diagnostics and warnings for missed targets.
- Geographic boundaries retain an inverse projection for all generated vertices.
  Distinguish the input geographic CRS from the projected x/y CRS.
- See `docs/metrics.md` and `docs/crdt_notes.md` for mask and tolerance semantics.

## Publication hygiene

Keep the README short and suitable for a first-time user. Put numerical details
in the guide; Python blocks in the main introductory pages are tested. Include
test fixtures, notices and documentation in source distributions. Preserve the
complete root `LICENSE`, including upstream notices, in distributions. Record
new upstream derivations in `docs/provenance.md`. Triangle is neither bundled nor
a runtime dependency.

Never commit secrets, machine-specific paths, caches or large raw datasets.
Deferred scope includes geodesic grid wrappers, bathymetry and AMR processing;
these are not prerequisites for the current Cartesian API.
