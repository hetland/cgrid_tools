## [Unreleased]

### Fixed
- Reject unsupported four-side topology and invalid active cells; preserve distinct
  crowded CRDT prevertices and downstream image corners.
- Apply asymmetric focus consistently to opposing edges.
- Prevent coordinate, mask and cached metric mutation from invalidating a `Grid`.
- Preserve inverse projections and projected CRS metadata for geographic grids.
- Avoid boundary mutation during triangulation and origin-dependent closure tests.
- Check CRDT residuals independently of optimizer success and expose convergence
  and mapping diagnostics.
- Respect named xarray dimensions, emit strict JSON, use rho-shaped Coriolis and
  average longitude across the dateline correctly.
- Refresh user guides and examples, include fixture notices in distributions,
  align local and CI checks, and retain third-party terms alongside MIT.

## 0.2.0 (not yet released)

### Added
- `Grid` class — immutable xarray-backed core object with `with_mask()`, `to_xarray()`, `to_netcdf()`, `from_netcdf()`, `to_dict()`
- `cgrid_tools.gridgen` subpackage containing all generation code
- `cgrid_tools.geo` — optional pyproj-gated geographic helpers (Coriolis, great-circle distances, projections)
- `cgrid_tools.io` — netCDF and dict serialization helpers including ROMS-convention dataset builder
- MIT license
- Read the Docs configuration (`.readthedocs.yaml`)
- GitHub Actions CI workflow (Python 3.10, 3.11, 3.12)
- `gridgen-c` example parity tests (examples 1–4)
- Sphinx + furo documentation site

### Changed
- Distribution name is `cgrid-tools`; Python import name is `cgrid_tools`.
- `GridGenerator.generate()` now returns `Grid` instead of `GridResult`
- Generation classes (`GridGenerator`, `BoundaryDefinition`, `Focus`, etc.) moved to `cgrid_tools.gridgen` subpackage

### Removed
- `GridResult` class (replaced by `Grid`)
- `CGrid` class (functionality now on `Grid` via `@cached_property` delegates to `cgrid_tools.metrics`)
- `quality.py` backward-compatibility shim

[keepachangelog.com]: https://keepachangelog.com/en/1.1.0/
[github-repo]: https://github.com/hetland/cgrid-tools
