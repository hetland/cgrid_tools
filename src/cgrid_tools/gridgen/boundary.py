from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from cgrid_tools._arrays import readonly

from .geometry import signed_area

Projection = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class LogicalSides:
    south: np.ndarray
    east: np.ndarray
    north: np.ndarray
    west: np.ndarray


@dataclass(frozen=True)
class BoundaryDefinition:
    """Polygon vertices and logical turns (quarter turns, summing to four).

    Four-side solvers require exactly four ``beta=1`` corners and optional
    ``beta=0`` vertices along each side. Use CRDT for reentrant beta walks.
    ``source_crs`` describes the input coordinates; ``target_crs`` describes
    projected x/y when the boundary was constructed from longitude/latitude.
    """

    x: np.ndarray
    y: np.ndarray
    beta: np.ndarray
    source_crs: str | None = None
    projected: bool = True
    original_lon: np.ndarray | None = None
    original_lat: np.ndarray | None = None

    target_crs: str | None = None
    inverse_projection: Projection | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        beta = np.asarray(self.beta, dtype=float)

        if x.ndim != 1 or y.ndim != 1 or beta.ndim != 1:
            raise ValueError("x, y, and beta must be one-dimensional arrays")
        if len(x) != len(y):
            raise ValueError("x and y must have matching lengths")
        if len(x) < 4:
            raise ValueError("boundary must include at least four vertices")

        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(beta))):
            raise ValueError("boundary coordinates and beta must be finite")
        tolerance = max(float(np.ptp(x)), float(np.ptp(y))) * 1.0e-12
        is_closed = np.hypot(x[0] - x[-1], y[0] - y[-1]) <= tolerance
        vertex_count = len(x) - 1 if is_closed else len(x)
        if vertex_count < 4:
            raise ValueError("boundary must include at least four vertices excluding closure")
        expected_beta_lengths = {vertex_count, vertex_count + 1}
        if len(beta) not in expected_beta_lengths:
            raise ValueError(
                "beta must have one entry per vertex, with an optional trailing closing entry"
            )
        if len(beta) == vertex_count + 1 and not np.isclose(beta[-1], 0.0):
            raise ValueError("the closing beta entry must be 0")
        if not is_closed:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
        if len(beta) == vertex_count:
            beta = np.append(beta, 0.0)
        if not np.isclose(beta.sum(), 4.0):
            raise ValueError("sum(beta) must equal 4")
        if not np.isclose(beta[-1], 0.0):
            raise ValueError("the closing beta entry must be 0")

        # Close exactly so downstream segment tests do not inherit a tiny gap.
        x, y = x.copy(), y.copy()
        x[-1], y[-1] = x[0], y[0]
        object.__setattr__(self, "x", readonly(x))
        object.__setattr__(self, "y", readonly(y))
        object.__setattr__(self, "beta", readonly(beta))
        if (self.original_lon is None) != (self.original_lat is None):
            raise ValueError("original_lon and original_lat must be provided together")
        for name in ("original_lon", "original_lat"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=float)
                if value.ndim != 1 or len(value) not in (vertex_count, vertex_count + 1):
                    raise ValueError(f"{name} must have one entry per boundary vertex")
                object.__setattr__(self, name, readonly(value[:vertex_count]))

    @classmethod
    def from_lonlat(
        cls,
        lon: np.ndarray,
        lat: np.ndarray,
        beta: np.ndarray,
        projection: Projection | None = None,
        source_crs: str = "EPSG:4326",
        *,
        target_crs: str | None = None,
        inverse_projection: Projection | None = None,
    ) -> BoundaryDefinition:
        """Project a geographic boundary and retain the inverse mapping.

        Supply ``target_crs`` (requires pyproj), a ``pyproj.Proj`` instance,
        or a custom forward callable with ``inverse_projection``. The latter
        is required when generating geographic vertices with a custom map.
        ``source_crs`` identifies the input longitude/latitude CRS.
        """
        lon_array = np.asarray(lon, dtype=float)
        lat_array = np.asarray(lat, dtype=float)
        if projection is None:
            if target_crs is None:
                raise ValueError("provide projection or target_crs")
            from cgrid_tools.geo import _require_pyproj

            pyproj = _require_pyproj()
            projection = pyproj.Transformer.from_crs(
                source_crs, target_crs, always_xy=True
            ).transform
            inverse_projection = pyproj.Transformer.from_crs(
                target_crs, source_crs, always_xy=True
            ).transform
        elif hasattr(projection, "crs"):
            from cgrid_tools.geo import _require_pyproj

            pyproj = _require_pyproj()
            if target_crs is not None and pyproj.CRS(target_crs) != projection.crs:
                raise ValueError("target_crs must match the supplied projection")
            target_crs = projection.crs.to_string()
            # Proj alone does not transform from source_crs's datum. Use a
            # paired Transformer so a source CRS override is honored both ways.
            projection = pyproj.Transformer.from_crs(
                source_crs, target_crs, always_xy=True
            ).transform
            if inverse_projection is None:
                inverse_projection = pyproj.Transformer.from_crs(
                    target_crs, source_crs, always_xy=True
                ).transform
        x, y = projection(lon_array, lat_array)
        return cls(
            x=x,
            y=y,
            beta=beta,
            source_crs=source_crs,
            projected=False,
            original_lon=lon_array,
            original_lat=lat_array,
            target_crs=target_crs,
            inverse_projection=inverse_projection,
        )

    @property
    def cartesian_crs(self) -> str | None:
        """CRS of x/y, distinct from the geographic input CRS."""
        return self.target_crs if self.original_lon is not None else self.source_crs

    def geographic_vertices(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Inverse-project generated vertices, preserving excluded NaN nodes."""
        if self.original_lon is None:
            return None, None
        if self.inverse_projection is None:
            raise ValueError(
                "geographic generation requires inverse_projection for a custom projection"
            )
        finite = np.isfinite(x) & np.isfinite(y)
        lon, lat = np.full(x.shape, np.nan), np.full(y.shape, np.nan)
        lon[finite], lat[finite] = self.inverse_projection(x[finite], y[finite])
        if not np.all(np.isfinite(lon[finite]) & np.isfinite(lat[finite])):
            raise ValueError("inverse projection returned non-finite geographic coordinates")
        return lon, lat

    @property
    def vertices(self) -> np.ndarray:
        return np.column_stack((self.x[:-1], self.y[:-1]))

    def normalized(self, start_index: int | None = None) -> BoundaryDefinition:
        """Rotate to a logical corner, retaining geographic metadata."""
        vertex_count = len(self.x) - 1
        if start_index is None:
            candidates = np.flatnonzero(np.isclose(self.beta[:-1], 1.0))
            if not candidates.size:
                raise ValueError("no beta=1 logical corner; use GRIDGEN_CRDT")
            order = np.lexsort((self.x[candidates], self.y[candidates]))
            start_index = int(candidates[order[0]])
        if not 0 <= start_index < vertex_count:
            raise IndexError("start_index is out of range")
        return replace(
            self,
            x=np.roll(self.x[:-1], -start_index),
            y=np.roll(self.y[:-1], -start_index),
            beta=np.roll(self.beta[:-1], -start_index),
            original_lon=(
                None if self.original_lon is None else np.roll(self.original_lon, -start_index)
            ),
            original_lat=(
                None if self.original_lat is None else np.roll(self.original_lat, -start_index)
            ),
        )

    def logical_sides(self, start_index: int | None = None) -> LogicalSides:
        """Four contiguous polylines, oriented counterclockwise.

        Reentrant/fractional beta walks describe a nonrectangular logical
        domain and cannot be passed to rectangular transfinite interpolation.
        """
        beta = self.beta[:-1]
        if (
            not np.all(np.isclose(beta, 0.0) | np.isclose(beta, 1.0))
            or np.count_nonzero(np.isclose(beta, 1.0)) != 4
        ):
            raise NotImplementedError(
                "four-side solvers require four beta=1 corners and beta=0 side vertices; "
                "use GRIDGEN_CRDT for reentrant or fractional beta walks"
            )
        boundary = self.normalized(start_index=start_index)
        points = boundary.vertices
        beta = boundary.beta[:-1]
        if not np.isclose(beta[0], 1.0):
            raise ValueError("start_index must identify a beta=1 logical corner")
        if signed_area(points) < 0.0:
            order = np.r_[0, np.arange(len(points) - 1, 0, -1)]
            points, beta = points[order], beta[order]
        corners = np.flatnonzero(np.isclose(beta, 1.0))
        closed = np.vstack((points, points[0]))
        ends = np.r_[corners[1:], len(points)]
        sides = [closed[start : stop + 1] for start, stop in zip(corners, ends, strict=True)]
        return LogicalSides(*sides)
