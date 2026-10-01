"""Immutable xarray-backed grid container for cgrid-tools."""

from __future__ import annotations

import copy
import json
from functools import cached_property
from os import PathLike
from typing import Any

import numpy as np
import xarray as xr

from . import metrics
from ._arrays import json_array, readonly
from .gridgen.geometry import points_in_polygon as _points_in_polygon


class GeoNotConfiguredError(RuntimeError):
    """Raised when a geographic property is accessed but lon/lat are not set."""


class Grid:
    """Immutable, xarray-backed C-grid core object.

    Parameters
    ----------
    x_vert, y_vert
        Vertex-point coordinates with shape ``(Mp + 1, Lp + 1)``.
    mask_rho
        Optional rho-point water mask with shape ``(Mp, Lp)``. The provided
        mask is intersected with the finite-vertex mask.
    lon_vert, lat_vert
        Optional geographic vertex coordinates matching ``x_vert`` shape.
    crs
        Optional coordinate reference system identifier.
    options
        JSON-safe generator/configuration options.
    """

    _VERT_DIMS = ("eta_vert", "xi_vert")
    _RHO_DIMS = ("eta_rho", "xi_rho")

    def __setattr__(self, name: str, value: Any) -> None:
        if "_dataset" in self.__dict__:
            raise AttributeError("Grid is immutable; use replace() or with_mask()")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Grid is immutable; use replace() or with_mask()")

    def __init__(
        self,
        x_vert: np.ndarray,
        y_vert: np.ndarray,
        *,
        mask_rho: np.ndarray | None = None,
        lon_vert: np.ndarray | None = None,
        lat_vert: np.ndarray | None = None,
        crs: str | None = None,
        options: dict | None = None,
    ) -> None:
        x_array, y_array = self._validate_vertex_arrays(x_vert, y_vert)
        lon_array, lat_array = self._validate_geo_arrays(lon_vert, lat_vert, x_array.shape)

        derived_mask = metrics.rho_mask_from_vertices(x_array, y_array)
        if mask_rho is not None:
            supplied_mask = np.asarray(mask_rho).astype(bool, copy=True)
            if supplied_mask.ndim != 2:
                raise ValueError("mask_rho must be a 2D array")
            if supplied_mask.shape != derived_mask.shape:
                raise ValueError(
                    f"mask_rho has shape {supplied_mask.shape}, expected {derived_mask.shape}"
                )
            derived_mask = derived_mask & supplied_mask

        options_dict = dict(options or {})
        self._dataset = xr.Dataset(
            data_vars={
                "x_vert": (self._VERT_DIMS, x_array.copy()),
                "y_vert": (self._VERT_DIMS, y_array.copy()),
                "mask_rho": (self._RHO_DIMS, derived_mask.astype(bool, copy=True)),
            },
            attrs={
                "crs": crs or "",
                "created_by": "cgrid_tools",
                "options_json": json.dumps(options_dict, allow_nan=False),
            },
        )
        if lon_array is not None:
            self._dataset["lon_vert"] = (self._VERT_DIMS, lon_array.copy())
        if lat_array is not None:
            self._dataset["lat_vert"] = (self._VERT_DIMS, lat_array.copy())
        for variable in self._dataset.data_vars.values():
            variable.data = readonly(variable.values)

    @staticmethod
    def _validate_vertex_arrays(
        x_vert: np.ndarray, y_vert: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x_array = np.array(x_vert, dtype=np.float64, copy=True)
        y_array = np.array(y_vert, dtype=np.float64, copy=True)
        if x_array.ndim != 2 or y_array.ndim != 2:
            raise ValueError("x_vert and y_vert must be 2D arrays")
        if x_array.shape != y_array.shape:
            raise ValueError("x_vert and y_vert must have identical shapes")
        if min(x_array.shape) < 2:
            raise ValueError("x_vert and y_vert must have at least 2 rows and 2 columns")
        if np.any(np.isinf(x_array)) or np.any(np.isinf(y_array)):
            raise ValueError("vertex coordinates must be finite or NaN")
        return x_array, y_array

    @staticmethod
    def _validate_geo_arrays(
        lon_vert: np.ndarray | None,
        lat_vert: np.ndarray | None,
        expected_shape: tuple[int, int],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if (lon_vert is None) != (lat_vert is None):
            raise ValueError("lon_vert and lat_vert must be provided together")
        if lon_vert is None or lat_vert is None:
            return None, None

        lon_array = np.array(lon_vert, dtype=np.float64, copy=True)
        lat_array = np.array(lat_vert, dtype=np.float64, copy=True)
        if lon_array.ndim != 2 or lat_array.ndim != 2:
            raise ValueError("lon_vert and lat_vert must be 2D arrays")
        if lon_array.shape != expected_shape or lat_array.shape != expected_shape:
            raise ValueError("lon_vert and lat_vert must have the same shape as x_vert and y_vert")
        if np.any(np.isinf(lon_array)) or np.any(np.isinf(lat_array)):
            raise ValueError("geographic coordinates must be finite or NaN")
        if np.any(np.abs(lat_array[np.isfinite(lat_array)]) > 90):
            raise ValueError("latitude must lie within [-90, 90] degrees")
        return lon_array, lat_array

    @classmethod
    def from_xarray(cls, ds: xr.Dataset) -> Grid:
        """Create a :class:`Grid` from an xarray dataset."""
        missing = {"x_vert", "y_vert", "mask_rho"}.difference(ds.data_vars)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"dataset is missing required variable(s): {names}")

        options_json = ds.attrs.get("options_json", "{}")
        try:
            options = json.loads(options_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("dataset attrs['options_json'] is not valid JSON") from exc
        if not isinstance(options, dict):
            raise ValueError("dataset attrs['options_json'] must decode to a dict")

        def values(name: str, dims: tuple[str, str]) -> np.ndarray:
            if len(ds[name].dims) != 2 or set(ds[name].dims) != set(dims):
                raise ValueError(f"{name} must have dimensions {dims}")
            return ds[name].transpose(*dims).values

        lon_vert = values("lon_vert", cls._VERT_DIMS) if "lon_vert" in ds.data_vars else None
        lat_vert = values("lat_vert", cls._VERT_DIMS) if "lat_vert" in ds.data_vars else None
        crs_attr = ds.attrs.get("crs", "")
        crs = str(crs_attr) if crs_attr else None

        return cls(
            x_vert=values("x_vert", cls._VERT_DIMS),
            y_vert=values("y_vert", cls._VERT_DIMS),
            mask_rho=values("mask_rho", cls._RHO_DIMS),
            lon_vert=lon_vert,
            lat_vert=lat_vert,
            crs=crs,
            options=options,
        )

    @classmethod
    def from_netcdf(cls, path: str | PathLike[str]) -> Grid:
        """Read a grid from a netCDF file."""
        with xr.open_dataset(path) as ds:
            return cls.from_xarray(ds.load())

    def to_xarray(self) -> xr.Dataset:
        """Return a deep copy of the internal dataset."""
        return self._dataset.copy(deep=True)

    def to_netcdf(self, path: str | PathLike[str]) -> None:
        """Write the grid dataset to netCDF."""
        self._dataset.to_netcdf(path)

    def to_dict(self) -> dict:
        """Return a JSON-safe nested-dict representation."""
        result: dict[str, Any] = {
            "x_vert": json_array(self.x),
            "y_vert": json_array(self.y),
            "mask_rho": self.mask_rho.tolist(),
            "crs": self.crs,
            "options": self.options,
        }
        if self.lon is not None:
            result["lon_vert"] = json_array(self.lon)
        if self.lat is not None:
            result["lat_vert"] = json_array(self.lat)
        return result

    def with_mask(self, polygon: np.ndarray, *, replace: bool = False) -> Grid:
        """Return a new grid with rho cells inside ``polygon`` masked out."""
        polygon_array = np.asarray(polygon, dtype=float)
        if polygon_array.ndim != 2 or polygon_array.shape[1] != 2:
            raise ValueError("polygon must have shape (N, 2)")
        if polygon_array.shape[0] < 3:
            raise ValueError("polygon must contain at least three vertices")

        x_center, y_center = metrics.rho_points(self.x, self.y)
        inside = _points_in_polygon(x_center, y_center, polygon_array)
        updated_mask = ~inside if replace else self.mask_rho & ~inside
        return self.replace(mask_rho=updated_mask)

    def replace(self, **kwargs: Any) -> Grid:
        """Return a new grid with selected constructor fields replaced."""
        allowed = {"x_vert", "y_vert", "mask_rho", "lon_vert", "lat_vert", "crs", "options"}
        unknown = set(kwargs).difference(allowed)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"unexpected replacement field(s): {names}")

        values: dict[str, Any] = {
            "x_vert": self.x.copy(),
            "y_vert": self.y.copy(),
            "mask_rho": self.mask_rho.copy(),
            "lon_vert": None if self.lon is None else self.lon.copy(),
            "lat_vert": None if self.lat is None else self.lat.copy(),
            "crs": self.crs,
            "options": self.options,
        }
        values.update(kwargs)
        return type(self)(**values)

    @property
    def x(self) -> np.ndarray:
        return self._dataset["x_vert"].values

    @property
    def y(self) -> np.ndarray:
        return self._dataset["y_vert"].values

    @property
    def mask_rho(self) -> np.ndarray:
        return self._dataset["mask_rho"].values

    @property
    def lon(self) -> np.ndarray | None:
        if "lon_vert" not in self._dataset:
            return None
        return self._dataset["lon_vert"].values

    @property
    def lat(self) -> np.ndarray | None:
        if "lat_vert" not in self._dataset:
            return None
        return self._dataset["lat_vert"].values

    @property
    def crs(self) -> str | None:
        crs = self._dataset.attrs.get("crs", "")
        return str(crs) if crs else None

    @property
    def options(self) -> dict:
        return copy.deepcopy(json.loads(self._dataset.attrs.get("options_json", "{}") or "{}"))

    @property
    def diagnostics(self) -> dict:
        """Copied solver diagnostics, when this grid was generated by a solver."""
        return self.options.get("diagnostics", {})

    @property
    def shape(self) -> tuple[int, int]:
        return self.x.shape

    @cached_property
    def mask_u(self) -> np.ndarray:
        return readonly(metrics.uvp_masks(self.mask_rho)[0])

    @cached_property
    def mask_v(self) -> np.ndarray:
        return readonly(metrics.uvp_masks(self.mask_rho)[1])

    @cached_property
    def mask_psi(self) -> np.ndarray:
        return readonly(metrics.uvp_masks(self.mask_rho)[2])

    @cached_property
    def dx(self) -> np.ndarray:
        return readonly(metrics.cell_widths(self.x, self.y)[0])

    @cached_property
    def dy(self) -> np.ndarray:
        return readonly(metrics.cell_widths(self.x, self.y)[1])

    @cached_property
    def pm(self) -> np.ndarray:
        return readonly(metrics.metric_coefficients(self.x, self.y)[0])

    @cached_property
    def pn(self) -> np.ndarray:
        return readonly(metrics.metric_coefficients(self.x, self.y)[1])

    @cached_property
    def angle(self) -> np.ndarray:
        return readonly(metrics.grid_angle(self.x, self.y, unwrap=True))

    @cached_property
    def area(self) -> np.ndarray:
        return readonly(metrics.cell_areas(self.x, self.y))

    @cached_property
    def f(self) -> np.ndarray:
        if self.lat is None:
            raise GeoNotConfiguredError("Coriolis requires lat_vert; construct Grid with lon/lat.")
        try:
            from cgrid_tools.geo import coriolis
        except ImportError as exc:
            raise GeoNotConfiguredError(
                "cgrid_tools.geo is not importable. This should not happen in a normal install."
            ) from exc
        _, lat_rho = metrics.rho_points(self.lon, self.lat)
        return readonly(coriolis(lat_rho))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        if self.crs != other.crs or self.options != other.options:
            return False
        for name in ("x_vert", "y_vert", "mask_rho", "lon_vert", "lat_vert"):
            if (name in self._dataset) != (name in other._dataset):
                return False
            if name in self._dataset and not np.array_equal(
                self._dataset[name].values, other._dataset[name].values, equal_nan=True
            ):
                return False
        return True

    def __repr__(self) -> str:
        total = int(self.mask_rho.size)
        masked = int(total - np.count_nonzero(self.mask_rho))
        geo = "yes" if self.lon is not None and self.lat is not None else "no"
        mode = self.options.get("mode", "")
        return f"Grid(shape={self.shape}, masked={masked}/{total}, geo={geo}, mode={mode!r})"
