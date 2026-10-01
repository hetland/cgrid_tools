from __future__ import annotations

import dataclasses
import warnings
from enum import Enum

import numpy as np
from numba import njit

from ._validation import GridGenerationError, grid_shape, validate_cells
from .boundary import BoundaryDefinition
from .crdt import CRDTOptions, generate_crdt_grid
from .focus import Focus
from .geometry import points_in_polygon as _points_in_polygon


class SolverMode(str, Enum):
    TRANSFINITE = "transfinite"
    LAPLACIAN = "laplacian"
    ELLIPTIC = "elliptic"
    GRIDGEN_CRDT = "gridgen_crdt"


class GridGenerator:
    """Generate a grid from a polygon and a vertex shape ``(ny, nx)``.

    TRANSFINITE, LAPLACIAN and ELLIPTIC require four contiguous logical
    sides. GRIDGEN_CRDT also supports reentrant/fractional beta walks.
    Folded or degenerate active cells raise ``GridGenerationError``.

    Parameters
    ----------
    mode
        SolverMode value or its string value (default: elliptic).
    focus
        Optional computational-coordinate clustering.
    smoothing_iterations
        Maximum smoothing iterations; zero returns the initializer.
    smoothing_strength
        Laplacian relaxation factor in (0, 1].
    elliptic_relaxation
        Successive over-relaxation factor in (0, 2).
    elliptic_tolerance
        Absolute stopping tolerance in coordinate units; zero disables stopping.
    crdt_options
        Quadrature and convergence settings for the conformal solver.
    """

    def __init__(
        self,
        mode: SolverMode = SolverMode.ELLIPTIC,
        focus: Focus | None = None,
        smoothing_iterations: int = 300,
        smoothing_strength: float = 0.9,
        elliptic_relaxation: float = 1.4,
        elliptic_tolerance: float = 1.0e-8,
        crdt_options: CRDTOptions | None = None,
    ) -> None:
        if not isinstance(smoothing_iterations, (int, np.integer)) or smoothing_iterations < 0:
            raise ValueError("smoothing_iterations must be a nonnegative integer")
        if not np.isfinite(smoothing_strength) or not 0 < smoothing_strength <= 1:
            raise ValueError("smoothing_strength must lie within (0, 1]")
        if not np.isfinite(elliptic_relaxation) or not 0 < elliptic_relaxation < 2:
            raise ValueError("elliptic_relaxation must lie within (0, 2)")
        if not np.isfinite(elliptic_tolerance) or elliptic_tolerance < 0:
            raise ValueError("elliptic_tolerance must be finite and nonnegative")
        self.mode = SolverMode(mode)
        self.focus = focus
        self.smoothing_iterations = smoothing_iterations
        self.smoothing_strength = smoothing_strength
        self.elliptic_relaxation = elliptic_relaxation
        self.elliptic_tolerance = elliptic_tolerance
        self.crdt_options = crdt_options

    def generate(self, boundary: BoundaryDefinition, shape: tuple[int, int]):
        from cgrid_tools.grid import Grid

        shape = grid_shape(shape)
        if self.mode is SolverMode.GRIDGEN_CRDT:
            grid = generate_crdt_grid(
                boundary,
                shape=shape,
                focus=self.focus,
                options=self.crdt_options,
            )
            return grid

        from .triangulation import _has_self_intersections

        if _has_self_intersections(boundary.vertices):
            raise ValueError("boundary polygon is self-intersecting")
        computational_x, computational_y = _computational_coordinates(
            shape=shape,
            focus=self.focus,
        )
        edges = _extract_edges(
            boundary,
            computational_x=computational_x,
            computational_y=computational_y,
        )
        x, y = _transfinite_interpolation(
            *edges,
            computational_x=computational_x,
            computational_y=computational_y,
        )
        cell_mask = _cell_mask_from_boundary(x, y, boundary)
        update_mask = _node_update_mask(cell_mask)

        if self.mode is SolverMode.LAPLACIAN:
            x, y = _laplacian_smooth(
                x,
                y,
                computational_x=computational_x,
                computational_y=computational_y,
                update_mask=update_mask,
                iterations=self.smoothing_iterations,
                strength=self.smoothing_strength,
            )
            iterations = self.smoothing_iterations
        elif self.mode is SolverMode.ELLIPTIC:
            x, y, iterations, residual = _elliptic_smooth(
                x,
                y,
                computational_x=computational_x,
                computational_y=computational_y,
                update_mask=update_mask,
                max_iterations=self.smoothing_iterations,
                relaxation=self.elliptic_relaxation,
                tolerance=self.elliptic_tolerance,
                return_residual=True,
            )
        else:
            iterations = 0

        # A numerical failure must not turn into land when the center mask is
        # recomputed. Four-side grids have no excluded image-plane vertices.
        if not np.all(np.isfinite(x) & np.isfinite(y)):
            raise GridGenerationError("four-side solver produced non-finite vertices")
        # Smoothing moves centers; recompute the geometry mask on the final grid.
        cell_mask = _cell_mask_from_boundary(x, y, boundary)
        validate_cells(x, y, cell_mask)
        converged = None
        if self.mode is SolverMode.ELLIPTIC and self.elliptic_tolerance > 0:
            converged = residual < self.elliptic_tolerance if iterations else None
            if not converged and self.smoothing_iterations > 0:
                warnings.warn(
                    "elliptic smoothing reached its iteration limit; inspect grid quality",
                    RuntimeWarning,
                    stacklevel=2,
                )
        options = _generation_options(
            boundary=boundary,
            mode=self.mode,
            shape=shape,
            focus=self.focus,
            smoothing_iterations=self.smoothing_iterations,
            smoothing_strength=self.smoothing_strength,
            elliptic_relaxation=self.elliptic_relaxation,
            elliptic_tolerance=self.elliptic_tolerance,
            crdt_options=self.crdt_options,
            iterations=iterations,
        )
        options["diagnostics"] = {
            "iterations": int(iterations),
            "converged": converged,
            "max_update": (
                float(residual) if self.mode is SolverMode.ELLIPTIC and iterations else None
            ),
        }
        lon_vert, lat_vert = boundary.geographic_vertices(x, y)
        return Grid(
            x_vert=x,
            y_vert=y,
            mask_rho=cell_mask,
            lon_vert=lon_vert,
            lat_vert=lat_vert,
            crs=boundary.cartesian_crs,
            options=options,
        )


def _generation_options(
    *,
    boundary: BoundaryDefinition,
    mode: SolverMode,
    shape: tuple[int, int],
    focus: Focus | None,
    smoothing_iterations: int,
    smoothing_strength: float,
    elliptic_relaxation: float,
    elliptic_tolerance: float,
    crdt_options: CRDTOptions | None,
    iterations: int,
) -> dict:
    import cgrid_tools as _pkg

    return {
        "mode": mode.value,
        "shape": list(shape),
        "focus": _focus_to_dict(focus),
        "smoothing_iterations": smoothing_iterations,
        "smoothing_strength": smoothing_strength,
        "elliptic_relaxation": elliptic_relaxation,
        "elliptic_tolerance": elliptic_tolerance,
        "crdt_options": _crdt_options_to_dict(crdt_options),
        "boundary": {
            "x": boundary.x[:-1].tolist(),
            "y": boundary.y[:-1].tolist(),
            "beta": boundary.beta[:-1].tolist(),
            "source_crs": boundary.source_crs,
            "target_crs": boundary.target_crs,
        },
        "iterations": int(iterations),
        "cgrid_tools_version": _pkg.__version__,
    }


def _focus_to_dict(focus: Focus | None) -> dict | None:
    if focus is None:
        return None
    return {
        "points": [
            {
                "pos": point.pos,
                "axis": point.axis,
                "factor": point.factor,
                "extent": point.extent,
            }
            for point in focus.points
        ]
    }


def _crdt_options_to_dict(opts: CRDTOptions | None) -> dict | None:
    if opts is None:
        return None
    d = dataclasses.asdict(opts)
    if d.get("image_beta") is not None:
        d["image_beta"] = np.asarray(d["image_beta"], dtype=float).tolist()
    return d


def _resample_polyline(points: np.ndarray, samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    if np.any(np.diff(samples) < 0.0):
        raise ValueError("samples must be monotonically increasing")
    if samples[0] < 0.0 or samples[-1] > 1.0:
        raise ValueError("samples must lie within [0, 1]")

    deltas = np.diff(points, axis=0)
    segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    distance = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = distance[-1]
    if np.isclose(total, 0.0):
        raise ValueError("edge length must be positive")
    sample_distance = samples * total
    x = np.interp(sample_distance, distance, points[:, 0])
    y = np.interp(sample_distance, distance, points[:, 1])
    return np.column_stack((x, y))


def _extract_edges(
    boundary: BoundaryDefinition,
    computational_x: np.ndarray,
    computational_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sides = boundary.logical_sides()
    south = _resample_polyline(sides.south, computational_x[0, :])
    east = _resample_polyline(sides.east, computational_y[:, -1])
    north = _resample_polyline(sides.north[::-1], computational_x[-1, :])
    west = _resample_polyline(sides.west[::-1], computational_y[:, 0])
    return south, east, north, west


def _transfinite_interpolation(
    south: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    west: np.ndarray,
    computational_x: np.ndarray,
    computational_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    XI = computational_x
    ETA = computational_y

    sw = south[0]
    se = south[-1]
    nw = north[0]
    ne = north[-1]

    xb = (
        (1.0 - ETA) * south[:, 0][None, :]
        + ETA * north[:, 0][None, :]
        + (1.0 - XI) * west[:, 0][:, None]
        + XI * east[:, 0][:, None]
        - (
            (1.0 - XI) * (1.0 - ETA) * sw[0]
            + XI * (1.0 - ETA) * se[0]
            + (1.0 - XI) * ETA * nw[0]
            + XI * ETA * ne[0]
        )
    )
    yb = (
        (1.0 - ETA) * south[:, 1][None, :]
        + ETA * north[:, 1][None, :]
        + (1.0 - XI) * west[:, 1][:, None]
        + XI * east[:, 1][:, None]
        - (
            (1.0 - XI) * (1.0 - ETA) * sw[1]
            + XI * (1.0 - ETA) * se[1]
            + (1.0 - XI) * ETA * nw[1]
            + XI * ETA * ne[1]
        )
    )
    return xb, yb


def _computational_coordinates(
    shape: tuple[int, int], focus: Focus | None
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    xi = np.linspace(0.0, 1.0, nx)
    eta = np.linspace(0.0, 1.0, ny)
    XI, ETA = np.meshgrid(xi, eta)
    if focus is None:
        return XI, ETA
    return focus(XI, ETA)


def _cell_mask_from_boundary(
    x: np.ndarray,
    y: np.ndarray,
    boundary: BoundaryDefinition,
) -> np.ndarray:
    x_center = 0.25 * (x[:-1, :-1] + x[1:, :-1] + x[1:, 1:] + x[:-1, 1:])
    y_center = 0.25 * (y[:-1, :-1] + y[1:, :-1] + y[1:, 1:] + y[:-1, 1:])
    return _points_in_polygon(x_center, y_center, boundary.vertices)


def _node_update_mask(cell_mask: np.ndarray) -> np.ndarray:
    ny = cell_mask.shape[0] + 1
    nx = cell_mask.shape[1] + 1
    update_mask = np.zeros((ny, nx), dtype=bool)
    interior = cell_mask[:-1, :-1] & cell_mask[1:, :-1] & cell_mask[:-1, 1:] & cell_mask[1:, 1:]
    update_mask[1:-1, 1:-1] = interior
    return update_mask


def _axis_coordinates(coordinates: np.ndarray, axis: int) -> np.ndarray:
    if axis == 1:
        return np.mean(coordinates, axis=0)
    if axis == 0:
        return np.mean(coordinates, axis=1)
    raise ValueError("axis must be 0 or 1")


def _second_derivative_weights(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_left = coordinates[1:-1] - coordinates[:-2]
    h_right = coordinates[2:] - coordinates[1:-1]
    if np.any(h_left <= 0.0) or np.any(h_right <= 0.0):
        raise ValueError("computational coordinates must be strictly increasing")

    left = 2.0 / (h_left * (h_left + h_right))
    center = -2.0 / (h_left * h_right)
    right = 2.0 / (h_right * (h_left + h_right))
    return left, center, right


def _laplacian_smooth(
    x: np.ndarray,
    y: np.ndarray,
    computational_x: np.ndarray,
    computational_y: np.ndarray,
    update_mask: np.ndarray,
    iterations: int,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_new = x.copy()
    y_new = y.copy()
    xi = _axis_coordinates(computational_x, axis=1)
    eta = _axis_coordinates(computational_y, axis=0)
    wx_left, wx_center, wx_right = _second_derivative_weights(xi)
    wy_down, wy_center, wy_up = _second_derivative_weights(eta)

    left = wx_left[None, :]
    right = wx_right[None, :]
    down = wy_down[:, None]
    up = wy_up[:, None]
    center = -(wx_center[None, :] + wy_center[:, None])
    interior_update_mask = update_mask[1:-1, 1:-1]

    for _ in range(iterations):
        x_avg = (
            left * x_new[1:-1, :-2]
            + right * x_new[1:-1, 2:]
            + down * x_new[:-2, 1:-1]
            + up * x_new[2:, 1:-1]
        ) / center
        y_avg = (
            left * y_new[1:-1, :-2]
            + right * y_new[1:-1, 2:]
            + down * y_new[:-2, 1:-1]
            + up * y_new[2:, 1:-1]
        ) / center
        x_interior = x_new[1:-1, 1:-1]
        y_interior = y_new[1:-1, 1:-1]
        x_interior[interior_update_mask] = (1.0 - strength) * x_interior[
            interior_update_mask
        ] + strength * x_avg[interior_update_mask]
        y_interior[interior_update_mask] = (1.0 - strength) * y_interior[
            interior_update_mask
        ] + strength * y_avg[interior_update_mask]

    return x_new, y_new


def _elliptic_smooth(
    x: np.ndarray,
    y: np.ndarray,
    computational_x: np.ndarray,
    computational_y: np.ndarray,
    update_mask: np.ndarray,
    max_iterations: int,
    relaxation: float,
    tolerance: float,
    *,
    return_residual: bool = False,
):
    x_new = x.copy()
    y_new = y.copy()
    xi = _axis_coordinates(computational_x, axis=1)
    eta = _axis_coordinates(computational_y, axis=0)
    xi_left = xi[1:-1] - xi[:-2]
    xi_right = xi[2:] - xi[1:-1]
    eta_down = eta[1:-1] - eta[:-2]
    eta_up = eta[2:] - eta[1:-1]
    if (
        np.any(xi_left <= 0.0)
        or np.any(xi_right <= 0.0)
        or np.any(eta_down <= 0.0)
        or np.any(eta_up <= 0.0)
    ):
        raise ValueError("computational coordinates must be strictly increasing")

    iterations, residual = _elliptic_smooth_kernel(
        x_new,
        y_new,
        update_mask,
        xi_left,
        xi_right,
        eta_down,
        eta_up,
        max_iterations,
        relaxation,
        tolerance,
    )
    if return_residual:
        return x_new, y_new, iterations, residual
    return x_new, y_new, iterations


@njit(cache=True)
def _elliptic_smooth_kernel(
    x_new: np.ndarray,
    y_new: np.ndarray,
    update_mask: np.ndarray,
    xi_left: np.ndarray,
    xi_right: np.ndarray,
    eta_down: np.ndarray,
    eta_up: np.ndarray,
    max_iterations: int,
    relaxation: float,
    tolerance: float,
) -> tuple[int, float]:
    ny, nx = x_new.shape
    max_delta = np.inf
    for iteration in range(1, max_iterations + 1):
        max_delta = 0.0
        for j in range(1, ny - 1):
            eta_down_j = eta_down[j - 1]
            eta_up_j = eta_up[j - 1]
            deta = eta_down_j + eta_up_j
            for i in range(1, nx - 1):
                if not update_mask[j, i]:
                    continue

                xi_left_i = xi_left[i - 1]
                xi_right_i = xi_right[i - 1]
                dxi = xi_left_i + xi_right_i

                x_xi = (x_new[j, i + 1] - x_new[j, i - 1]) / dxi
                y_xi = (y_new[j, i + 1] - y_new[j, i - 1]) / dxi
                x_eta = (x_new[j + 1, i] - x_new[j - 1, i]) / deta
                y_eta = (y_new[j + 1, i] - y_new[j - 1, i]) / deta

                alpha = x_eta * x_eta + y_eta * y_eta
                beta = x_xi * x_eta + y_xi * y_eta
                gamma = x_xi * x_xi + y_xi * y_xi

                c_left = alpha * 2.0 / (xi_left_i * dxi)
                c_right = alpha * 2.0 / (xi_right_i * dxi)
                c_down = gamma * 2.0 / (eta_down_j * deta)
                c_up = gamma * 2.0 / (eta_up_j * deta)
                denom = c_left + c_right + c_down + c_up
                if denom <= 1.0e-14:
                    continue

                cross_x = (
                    x_new[j + 1, i + 1]
                    - x_new[j + 1, i - 1]
                    - x_new[j - 1, i + 1]
                    + x_new[j - 1, i - 1]
                ) / (dxi * deta)
                cross_y = (
                    y_new[j + 1, i + 1]
                    - y_new[j + 1, i - 1]
                    - y_new[j - 1, i + 1]
                    + y_new[j - 1, i - 1]
                ) / (dxi * deta)

                x_star = (
                    c_right * x_new[j, i + 1]
                    + c_left * x_new[j, i - 1]
                    + c_up * x_new[j + 1, i]
                    + c_down * x_new[j - 1, i]
                    - 2.0 * beta * cross_x
                ) / denom
                y_star = (
                    c_right * y_new[j, i + 1]
                    + c_left * y_new[j, i - 1]
                    + c_up * y_new[j + 1, i]
                    + c_down * y_new[j - 1, i]
                    - 2.0 * beta * cross_y
                ) / denom

                dx = relaxation * (x_star - x_new[j, i])
                dy = relaxation * (y_star - y_new[j, i])
                x_new[j, i] += dx
                y_new[j, i] += dy
                abs_dx = dx if dx >= 0.0 else -dx
                abs_dy = dy if dy >= 0.0 else -dy
                if abs_dx > max_delta:
                    max_delta = abs_dx
                if abs_dy > max_delta:
                    max_delta = abs_dy

        if max_delta < tolerance:
            return iteration, max_delta

    return max_iterations, max_delta


def _elliptic_smooth_python(
    x: np.ndarray,
    y: np.ndarray,
    computational_x: np.ndarray,
    computational_y: np.ndarray,
    update_mask: np.ndarray,
    max_iterations: int,
    relaxation: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    x_new = x.copy()
    y_new = y.copy()
    xi = _axis_coordinates(computational_x, axis=1)
    eta = _axis_coordinates(computational_y, axis=0)

    for iteration in range(1, max_iterations + 1):
        max_delta = 0.0
        for j in range(1, x_new.shape[0] - 1):
            for i in range(1, x_new.shape[1] - 1):
                if not update_mask[j, i]:
                    continue

                dxi = xi[i + 1] - xi[i - 1]
                deta = eta[j + 1] - eta[j - 1]
                if dxi <= 0.0 or deta <= 0.0:
                    raise ValueError("computational coordinates must be strictly increasing")

                x_xi = (x_new[j, i + 1] - x_new[j, i - 1]) / dxi
                y_xi = (y_new[j, i + 1] - y_new[j, i - 1]) / dxi
                x_eta = (x_new[j + 1, i] - x_new[j - 1, i]) / deta
                y_eta = (y_new[j + 1, i] - y_new[j - 1, i]) / deta

                alpha = x_eta * x_eta + y_eta * y_eta
                beta = x_xi * x_eta + y_xi * y_eta
                gamma = x_xi * x_xi + y_xi * y_xi

                xi_left = xi[i] - xi[i - 1]
                xi_right = xi[i + 1] - xi[i]
                eta_down = eta[j] - eta[j - 1]
                eta_up = eta[j + 1] - eta[j]
                xi_sum = xi_left + xi_right
                eta_sum = eta_down + eta_up

                c_left = alpha * 2.0 / (xi_left * xi_sum)
                c_right = alpha * 2.0 / (xi_right * xi_sum)
                c_down = gamma * 2.0 / (eta_down * eta_sum)
                c_up = gamma * 2.0 / (eta_up * eta_sum)
                denom = c_left + c_right + c_down + c_up
                if denom <= 1.0e-14:
                    continue

                cross_x = (
                    x_new[j + 1, i + 1]
                    - x_new[j + 1, i - 1]
                    - x_new[j - 1, i + 1]
                    + x_new[j - 1, i - 1]
                ) / (dxi * deta)
                cross_y = (
                    y_new[j + 1, i + 1]
                    - y_new[j + 1, i - 1]
                    - y_new[j - 1, i + 1]
                    + y_new[j - 1, i - 1]
                ) / (dxi * deta)

                x_star = (
                    c_right * x_new[j, i + 1]
                    + c_left * x_new[j, i - 1]
                    + c_up * x_new[j + 1, i]
                    + c_down * x_new[j - 1, i]
                    - 2.0 * beta * cross_x
                ) / denom
                y_star = (
                    c_right * y_new[j, i + 1]
                    + c_left * y_new[j, i - 1]
                    + c_up * y_new[j + 1, i]
                    + c_down * y_new[j - 1, i]
                    - 2.0 * beta * cross_y
                ) / denom

                dx = relaxation * (x_star - x_new[j, i])
                dy = relaxation * (y_star - y_new[j, i])
                x_new[j, i] += dx
                y_new[j, i] += dy
                max_delta = max(max_delta, abs(dx), abs(dy))

        if max_delta < tolerance:
            return x_new, y_new, iteration

    return x_new, y_new, max_iterations
