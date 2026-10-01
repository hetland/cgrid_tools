# API Reference

## Core

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   cgrid_tools.Grid
   cgrid_tools.GeoNotConfiguredError
```

## Metrics

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   cgrid_tools.metrics.orthogonality_error
   cgrid_tools.metrics.orthogonality_summary
   cgrid_tools.metrics.metric_coefficients
   cgrid_tools.metrics.cell_widths
   cgrid_tools.metrics.grid_angle
   cgrid_tools.metrics.cell_jacobians
   cgrid_tools.metrics.cell_areas
```

## Generation

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   cgrid_tools.gridgen.GridGenerationError
   cgrid_tools.gridgen.GridGenerator
   cgrid_tools.gridgen.BoundaryDefinition
   cgrid_tools.gridgen.Focus
   cgrid_tools.gridgen.SolverMode
   cgrid_tools.gridgen.CRDTOptions
```

## Geographic

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   cgrid_tools.geo.geographic_rho_points
   cgrid_tools.geo.coriolis
   cgrid_tools.geo.great_circle_dx
   cgrid_tools.geo.great_circle_dy
   cgrid_tools.geo.project_lonlat_to_xy
```

## Serialization

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   cgrid_tools.io.dict_from_grid
   cgrid_tools.io.grid_from_dict
   cgrid_tools.io.roms_grid_from_grid
```
