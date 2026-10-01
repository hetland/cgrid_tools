"""Curvilinear grid metrics (Arakawa C-grid, ROMS convention).

All public functions in this module accept *vertex* coordinate arrays
``x_vert`` and ``y_vert`` of shape ``(Mp + 1, Lp + 1)``. This is the outer
corner grid in ROMS convention, where ``Mp`` and ``Lp`` are the full-grid
dimensions (including the two boundary rho rows/columns) and the interior
rho grid lives at indices ``(1..Mm, 1..Lm)`` with ``Mm = Mp - 2``,
``Lm = Lp - 2``.

Shape summary (ROMS convention)
-------------------------------

- vertex corners : ``(Mp + 1, Lp + 1)``
- rho points     : ``(Mp, Lp)``           (cell centers)
- psi points     : ``(Mm + 1, Lm + 1)``   (interior vertex corners)
- u points       : ``(Mp, Lm + 1)``
- v points       : ``(Mm + 1, Lp)``

NaN handling
------------

A vertex entry of ``NaN`` is treated as "masked" / absent. Derived arrays
propagate NaN through arithmetic and also expose a boolean mask via
:func:`rho_mask_from_vertices` and :func:`uvp_masks`.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Arakawa subgrid points
# ---------------------------------------------------------------------------


def rho_points(x_vert: np.ndarray, y_vert: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cell-center (rho) coordinates.

    Returns
    -------
    x_rho, y_rho : np.ndarray, shape ``(Mp, Lp)``
        Average of the four vertex corners bounding each cell. The full rho
        grid is returned; the ROMS *interior* rho grid of shape
        ``(Mm, Lm)`` is ``x_rho[1:-1, 1:-1]``.
    """
    x_vert = np.asarray(x_vert, dtype=float)
    y_vert = np.asarray(y_vert, dtype=float)
    x_rho = 0.25 * (x_vert[1:, 1:] + x_vert[1:, :-1] + x_vert[:-1, 1:] + x_vert[:-1, :-1])
    y_rho = 0.25 * (y_vert[1:, 1:] + y_vert[1:, :-1] + y_vert[:-1, 1:] + y_vert[:-1, :-1])
    return x_rho, y_rho


def u_points(x_vert: np.ndarray, y_vert: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """u-velocity point coordinates (midpoints of eta-aligned interior edges).

    Returns
    -------
    x_u, y_u : np.ndarray, shape ``(Mp, Lm + 1)``
    """
    x_vert = np.asarray(x_vert, dtype=float)
    y_vert = np.asarray(y_vert, dtype=float)
    x_u = 0.5 * (x_vert[:-1, 1:-1] + x_vert[1:, 1:-1])
    y_u = 0.5 * (y_vert[:-1, 1:-1] + y_vert[1:, 1:-1])
    return x_u, y_u


def v_points(x_vert: np.ndarray, y_vert: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v-velocity point coordinates (midpoints of xi-aligned interior edges).

    Returns
    -------
    x_v, y_v : np.ndarray, shape ``(Mm + 1, Lp)``
    """
    x_vert = np.asarray(x_vert, dtype=float)
    y_vert = np.asarray(y_vert, dtype=float)
    x_v = 0.5 * (x_vert[1:-1, :-1] + x_vert[1:-1, 1:])
    y_v = 0.5 * (y_vert[1:-1, :-1] + y_vert[1:-1, 1:])
    return x_v, y_v


def psi_points(x_vert: np.ndarray, y_vert: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """psi-point coordinates (interior vertex corners).

    Returns
    -------
    x_psi, y_psi : np.ndarray, shape ``(Mm + 1, Lm + 1)``
    """
    x_vert = np.asarray(x_vert, dtype=float)
    y_vert = np.asarray(y_vert, dtype=float)
    return x_vert[1:-1, 1:-1].copy(), y_vert[1:-1, 1:-1].copy()


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def rho_mask_from_vertices(x_vert: np.ndarray, y_vert: np.ndarray) -> np.ndarray:
    """Boolean rho-cell mask derived from finiteness of the vertex corners.

    A rho cell is ``True`` (water) iff all four of its vertex corners are
    finite.

    Returns
    -------
    mask_rho : np.ndarray[bool], shape ``(Mp, Lp)``
    """
    x_vert = np.asarray(x_vert, dtype=float)
    y_vert = np.asarray(y_vert, dtype=float)
    finite = np.isfinite(x_vert) & np.isfinite(y_vert)
    return finite[:-1, :-1] & finite[1:, :-1] & finite[:-1, 1:] & finite[1:, 1:]


def uvp_masks(mask_rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive u-, v-, and psi-point masks from a rho-point mask.

    Port of ``octant.grid.uvp_masks`` returning boolean arrays.

    Parameters
    ----------
    mask_rho : np.ndarray, shape ``(Mp, Lp)``
        Boolean or ``{0, 1}`` mask at rho points.

    Returns
    -------
    mask_u : np.ndarray[bool], shape ``(Mp, Lm + 1)``
    mask_v : np.ndarray[bool], shape ``(Mm + 1, Lp)``
    mask_psi : np.ndarray[bool], shape ``(Mm + 1, Lm + 1)``
    """
    rmask = np.asarray(mask_rho).astype(bool)
    if rmask.ndim != 2:
        raise ValueError("mask_rho must be a 2D array")
    u_mask = rmask[:, :-1] & rmask[:, 1:]
    v_mask = rmask[:-1, :] & rmask[1:, :]
    psi_mask = rmask[:-1, :-1] & rmask[:-1, 1:] & rmask[1:, :-1] & rmask[1:, 1:]
    return u_mask, v_mask, psi_mask


# ---------------------------------------------------------------------------
# Cell widths & metric coefficients
# ---------------------------------------------------------------------------


def cell_widths(x_vert: np.ndarray, y_vert: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cell widths along the xi (``dx``) and eta (``dy``) directions.

    Follows ``octant.grid.CGrid._calculate_metrics``: ``dx`` is computed from
    the eta-average of adjacent vertices differenced along xi, and vice versa.

    Returns
    -------
    dx, dy : np.ndarray, shape ``(Mp, Lp)``
        Cell widths along xi and eta at rho points.
    """
    x_vert = np.asarray(x_vert, dtype=float)
    y_vert = np.asarray(y_vert, dtype=float)

    x_temp = 0.5 * (x_vert[1:, :] + x_vert[:-1, :])
    y_temp = 0.5 * (y_vert[1:, :] + y_vert[:-1, :])
    dx = np.hypot(np.diff(x_temp, axis=1), np.diff(y_temp, axis=1))

    x_temp = 0.5 * (x_vert[:, 1:] + x_vert[:, :-1])
    y_temp = 0.5 * (y_vert[:, 1:] + y_vert[:, :-1])
    dy = np.hypot(np.diff(x_temp, axis=0), np.diff(y_temp, axis=0))

    return dx, dy


def metric_coefficients(x_vert: np.ndarray, y_vert: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ROMS metric coefficients ``pm = 1/dx`` and ``pn = 1/dy``.

    Where the corresponding width is zero or non-finite the coefficient is
    returned as NaN rather than inf to keep downstream arithmetic well-behaved.

    Returns
    -------
    pm, pn : np.ndarray, shape ``(Mp, Lp)``
    """
    dx, dy = cell_widths(x_vert, y_vert)
    with np.errstate(divide="ignore", invalid="ignore"):
        pm = np.where(np.isfinite(dx) & (dx > 0.0), 1.0 / dx, np.nan)
        pn = np.where(np.isfinite(dy) & (dy > 0.0), 1.0 / dy, np.nan)
    return pm, pn


def metric_derivatives(x_vert: np.ndarray, y_vert: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centered derivatives of the inverse metric coefficients.

    Specifically:

    - ``dndx = d(1/pn)/dxi = d(dy)/dxi`` via centered differences in xi.
    - ``dmde = d(1/pm)/deta = d(dx)/deta`` via centered differences in eta.

    Boundary rows/columns are filled with zero, matching octant's convention.

    Returns
    -------
    dndx, dmde : np.ndarray, shape ``(Mp, Lp)``
    """
    dx, dy = cell_widths(x_vert, y_vert)
    dndx = np.zeros_like(dx)
    dmde = np.zeros_like(dx)
    dndx[1:-1, 1:-1] = 0.5 * (dy[1:-1, 2:] - dy[1:-1, :-2])
    dmde[1:-1, 1:-1] = 0.5 * (dx[2:, 1:-1] - dx[:-2, 1:-1])
    return dndx, dmde


# ---------------------------------------------------------------------------
# Grid angle (orientation)
# ---------------------------------------------------------------------------


def _unwrap_along_axis_with_mask(arr: np.ndarray, axis: int) -> np.ndarray:
    """Unwrap ``arr`` along ``axis`` within each contiguous finite run.

    NaN entries are preserved; they act as fences that split the unwrap
    operation into independent segments. Behavior on a fully-finite array is
    identical to ``np.unwrap(arr, axis=axis)``.
    """
    arr = np.asarray(arr, dtype=float).copy()
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    # Iterate along the opposite axis.
    if axis == 0:
        for j in range(arr.shape[1]):
            arr[:, j] = _unwrap_1d_with_mask(arr[:, j])
    else:
        for i in range(arr.shape[0]):
            arr[i, :] = _unwrap_1d_with_mask(arr[i, :])
    return arr


def _unwrap_1d_with_mask(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    finite = np.isfinite(values)
    if not np.any(finite):
        return result
    n = len(values)
    idx = 0
    while idx < n:
        if not finite[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and finite[idx]:
            idx += 1
        stop = idx  # exclusive
        if stop - start >= 2:
            result[start:stop] = np.unwrap(values[start:stop])
    return result


def grid_angle(x_vert: np.ndarray, y_vert: np.ndarray, unwrap: bool = True) -> np.ndarray:
    """Angle (radians) of grid xi-lines relative to the x-axis.

    Follows ``octant.grid.CGrid._calculate_metrics``: uses the eta-average of
    adjacent vertices, then ``arctan2`` of the xi-diff. The result is padded
    back to vertex shape ``(Mp + 1, Lp + 1)`` by repeating the last row and column.

    Parameters
    ----------
    x_vert, y_vert : np.ndarray, shape ``(Mp + 1, Lp + 1)``
    unwrap : bool, default True
        If True, ``np.unwrap`` is applied along each axis within each
        contiguous finite run (NaN entries act as fences). If False, raw
        ``arctan2`` output is returned.

    Returns
    -------
    angle : np.ndarray, shape ``(Mp + 1, Lp + 1)``
    """
    x_vert = np.asarray(x_vert, dtype=float)
    y_vert = np.asarray(y_vert, dtype=float)

    # Reproduce octant's formula exactly:
    #     arctan2(diff(0.5*(y_vert[1:,:]+y_vert[:-1,:])),
    #             diff(0.5*(x_vert[1:,:]+x_vert[:-1,:])))
    # The eta-average collapses the Mp+1 vertex rows to Mp; the xi-diff
    # collapses Lp+1 columns to Lp. So the inner angle array has shape
    # (Mp, Lp).
    inner = np.arctan2(
        np.diff(0.5 * (y_vert[1:, :] + y_vert[:-1, :]), axis=1),
        np.diff(0.5 * (x_vert[1:, :] + x_vert[:-1, :]), axis=1),
    )

    # Pad to vertex shape (Mp + 1, Lp + 1) by repeating the last row and
    # last column, preserving the historical vertex-shaped API.
    padded = np.full((x_vert.shape[0], x_vert.shape[1]), np.nan, dtype=float)
    padded[:-1, :-1] = inner
    padded[-1, :-1] = inner[-1, :]
    padded[:-1, -1] = inner[:, -1]
    padded[-1, -1] = inner[-1, -1]

    if unwrap:
        padded = _unwrap_along_axis_with_mask(padded, axis=0)
        padded = _unwrap_along_axis_with_mask(padded, axis=1)
    return padded


# ---------------------------------------------------------------------------
# Orthogonality
# ---------------------------------------------------------------------------


def orthogonality_corner_angles(x_vert: np.ndarray, y_vert: np.ndarray) -> np.ndarray:
    """Four signed deviations from pi/2 at each rho cell's four corners.

    Each of the four returned layers contains ``ang_k - pi/2`` radians, where
    ``ang_k`` is the interior angle at one of the four vertex corners of
    each rho cell. A perfectly orthogonal grid yields zeros everywhere.

    Returns
    -------
    angles : np.ndarray, shape ``(4, Mp, Lp)``
        Stacked signed corner-angle deviations at rho-cell corners.
    """
    x_vert = np.asarray(x_vert, dtype=float)
    y_vert = np.asarray(y_vert, dtype=float)
    z = x_vert + 1j * y_vert

    # Edge vectors (normalized) along xi and eta.
    with np.errstate(divide="ignore", invalid="ignore"):
        du_full = np.diff(z, axis=1)
        du_full = du_full / np.abs(du_full)
        dv_full = np.diff(z, axis=0)
        dv_full = dv_full / np.abs(dv_full)

    # Octant's 4 corner stencils per rho cell (ang1..ang4):
    # ang1 -> (du[:-1,:], dv[:,:-1])   south-west
    # ang2 -> (du[1:,:],  dv[:,:-1])   north-west
    # ang3 -> (du[:-1,:], dv[:,1:])    south-east
    # ang4 -> (du[1:,:],  dv[:,1:])    north-east
    configs = (
        (du_full[:-1, :], dv_full[:, :-1]),
        (du_full[1:, :], dv_full[:, :-1]),
        (du_full[:-1, :], dv_full[:, 1:]),
        (du_full[1:, :], dv_full[:, 1:]),
    )
    out = np.empty((4, z.shape[0] - 1, z.shape[1] - 1), dtype=float)
    for k, (du, dv) in enumerate(configs):
        dot = du.real * dv.real + du.imag * dv.imag
        out[k] = np.arccos(np.clip(dot, -1.0, 1.0)) - np.pi / 2.0
    # out has shape (4, Mp, Lp) for a vertex grid of shape (Mp+1, Lp+1).
    return out


def orthogonality_signed(x_vert: np.ndarray, y_vert: np.ndarray) -> np.ndarray:
    """Signed orthogonality error at rho cells (radians).

    Direct port of ``octant.grid.CGrid.calculate_orthogonality``: returns the
    mean of the four ``|corner angle|`` values minus pi/2 at each rho cell.
    The sign follows the angle between increasing xi and eta directions,
    not geometric interior angles. A perfectly orthogonal grid yields zeros.

    Returns
    -------
    error : np.ndarray, shape ``(Mp, Lp)``
    """
    corners = orthogonality_corner_angles(x_vert, y_vert)
    abs_angles = np.abs(corners + np.pi / 2.0)
    return np.mean(abs_angles, axis=0) - np.pi / 2.0


def orthogonality_error(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unsigned orthogonality error at rho cells (radians).

    Returns the mean of ``|corner_angle - pi/2|`` over the four rho-cell
    vertex corners. NaN cells are propagated.

    Returns
    -------
    error : np.ndarray, shape ``(Mp, Lp)``
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dux = x[:, 1:] - x[:, :-1]
    duy = y[:, 1:] - y[:, :-1]
    dvx = x[1:, :] - x[:-1, :]
    dvy = y[1:, :] - y[:-1, :]

    dots = (
        dux[:-1, :] * dvx[:, :-1] + duy[:-1, :] * dvy[:, :-1],
        dux[1:, :] * dvx[:, :-1] + duy[1:, :] * dvy[:, :-1],
        dux[:-1, :] * dvx[:, 1:] + duy[:-1, :] * dvy[:, 1:],
        dux[1:, :] * dvx[:, 1:] + duy[1:, :] * dvy[:, 1:],
    )
    nu = (
        np.hypot(dux[:-1, :], duy[:-1, :]) * np.hypot(dvx[:, :-1], dvy[:, :-1]),
        np.hypot(dux[1:, :], duy[1:, :]) * np.hypot(dvx[:, :-1], dvy[:, :-1]),
        np.hypot(dux[:-1, :], duy[:-1, :]) * np.hypot(dvx[:, 1:], dvy[:, 1:]),
        np.hypot(dux[1:, :], duy[1:, :]) * np.hypot(dvx[:, 1:], dvy[:, 1:]),
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        angles = [np.arccos(np.clip(d / n, -1.0, 1.0)) for d, n in zip(dots, nu)]
    return np.mean(np.abs(np.stack(angles) - (np.pi / 2.0)), axis=0)


def orthogonality_summary(x_vert: np.ndarray, y_vert: np.ndarray) -> dict:
    """Scalar summary of orthogonality error (radians).

    Uses :func:`orthogonality_error` (unsigned) for ``mean_abs``/``max_abs``/
    ``rms`` and :func:`orthogonality_signed` for ``signed_mean``. NaN entries
    arising from masked corners are ignored via ``np.nan*`` reductions.

    Returns
    -------
    summary : dict with keys ``mean_abs``, ``max_abs``, ``rms``, ``signed_mean``
    """
    err = orthogonality_error(x_vert, y_vert)
    signed = orthogonality_signed(x_vert, y_vert)
    with np.errstate(invalid="ignore"):
        return {
            "mean_abs": float(np.nanmean(err)),
            "max_abs": float(np.nanmax(err)),
            "rms": float(np.sqrt(np.nanmean(err * err))),
            "signed_mean": float(np.nanmean(signed)),
        }


# ---------------------------------------------------------------------------
# Cell areas (kept here so quality.py is a thin re-export)
# ---------------------------------------------------------------------------


def cell_areas(x_vert: np.ndarray, y_vert: np.ndarray) -> np.ndarray:
    """Cell areas computed as the sum of two triangle areas per quad.

    Returns
    -------
    area : np.ndarray, shape ``(Mp, Lp)``
    """
    x = np.asarray(x_vert, dtype=float)
    y = np.asarray(y_vert, dtype=float)
    x00 = x[:-1, :-1]
    y00 = y[:-1, :-1]
    x10 = x[1:, :-1]
    y10 = y[1:, :-1]
    x11 = x[1:, 1:]
    y11 = y[1:, 1:]
    x01 = x[:-1, 1:]
    y01 = y[:-1, 1:]
    area1 = 0.5 * np.abs((x10 - x00) * (y11 - y00) - (x11 - x00) * (y10 - y00))
    area2 = 0.5 * np.abs((x11 - x00) * (y01 - y00) - (x01 - x00) * (y11 - y00))
    return area1 + area2


def cell_jacobians(x_vert: np.ndarray, y_vert: np.ndarray) -> np.ndarray:
    """Signed bilinear-map Jacobian at each cell corner, shape ``(4, Mp, Lp)``.

    Positive at every corner means a nondegenerate, counterclockwise cell.
    Mixed signs identify folded/concave cells; all negative means clockwise.
    Unlike absolute areas, these values retain orientation information.
    """
    x = np.asarray(x_vert, dtype=float)
    y = np.asarray(y_vert, dtype=float)
    dux, duy = np.diff(x, axis=1), np.diff(y, axis=1)
    dvx, dvy = np.diff(x, axis=0), np.diff(y, axis=0)
    return np.stack(
        (
            dux[:-1] * dvy[:, :-1] - duy[:-1] * dvx[:, :-1],
            dux[:-1] * dvy[:, 1:] - duy[:-1] * dvx[:, 1:],
            dux[1:] * dvy[:, :-1] - duy[1:] * dvx[:, :-1],
            dux[1:] * dvy[:, 1:] - duy[1:] * dvx[:, 1:],
        )
    )
