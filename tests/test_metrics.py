"""Analytic-fixture tests for :mod:`cgrid_tools.metrics` and :class:`Grid`.

All fixtures use vertex-point arrays of shape ``(Mp + 1, Lp + 1)`` in ROMS
convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from cgrid_tools import (
    Grid,
    cell_areas,
    cell_widths,
    grid_angle,
    metric_coefficients,
    metric_derivatives,
    orthogonality_corner_angles,
    orthogonality_error,
    orthogonality_signed,
    orthogonality_summary,
    psi_points,
    rho_mask_from_vertices,
    rho_points,
    u_points,
    uvp_masks,
    v_points,
)
from cgrid_tools.gridgen import BoundaryDefinition, Focus, GridGenerator, SolverMode

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _uniform_grid(Mp: int = 6, Lp: int = 8, dx0: float = 2.0, dy0: float = 1.5):
    """Uniform Cartesian grid with vertex shape ``(Mp + 1, Lp + 1)``."""
    xi = np.arange(Lp + 1, dtype=float) * dx0
    eta = np.arange(Mp + 1, dtype=float) * dy0
    X, Y = np.meshgrid(xi, eta)
    return X, Y


def _rotated_grid(Mp: int = 6, Lp: int = 8, angle_rad: float = np.pi / 6):
    X, Y = _uniform_grid(Mp, Lp, dx0=2.0, dy0=1.5)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return c * X - s * Y, s * X + c * Y


def _sheared_grid(Mp: int = 6, Lp: int = 8, shear: float = 0.3):
    X, Y = _uniform_grid(Mp, Lp, dx0=2.0, dy0=1.5)
    return X + shear * Y, Y


def _polar_patch(
    Mp: int = 10,
    Lp: int = 12,
    r0: float = 1.0,
    r1: float = 2.5,
    theta0: float = 0.2,
    theta1: float = 1.1,
):
    """Annular patch, uniform in computational coordinates."""
    r = np.linspace(r0, r1, Mp + 1)
    theta = np.linspace(theta0, theta1, Lp + 1)
    R, T = np.meshgrid(r, theta, indexing="ij")
    return R * np.cos(T), R * np.sin(T)


# ---------------------------------------------------------------------------
# 1. Uniform Cartesian grid
# ---------------------------------------------------------------------------


def test_uniform_cartesian_metrics_match_analytic_values() -> None:
    Mp, Lp = 6, 8
    dx0, dy0 = 2.0, 1.5
    x, y = _uniform_grid(Mp, Lp, dx0=dx0, dy0=dy0)

    assert orthogonality_error(x, y).shape == (Mp, Lp)
    assert np.allclose(orthogonality_error(x, y), 0.0, atol=1.0e-12)
    assert np.allclose(orthogonality_signed(x, y), 0.0, atol=1.0e-12)

    dx, dy = cell_widths(x, y)
    assert dx.shape == (Mp, Lp)
    assert dy.shape == (Mp, Lp)
    assert np.allclose(dx, dx0)
    assert np.allclose(dy, dy0)

    pm, pn = metric_coefficients(x, y)
    assert np.allclose(pm, 1.0 / dx0)
    assert np.allclose(pn, 1.0 / dy0)

    dndx, dmde = metric_derivatives(x, y)
    assert np.allclose(dndx, 0.0, atol=1.0e-12)
    assert np.allclose(dmde, 0.0, atol=1.0e-12)

    ang = grid_angle(x, y)
    assert ang.shape == x.shape
    assert np.allclose(ang, 0.0, atol=1.0e-12)

    area = cell_areas(x, y)
    assert area.shape == (Mp, Lp)
    assert np.allclose(area, dx0 * dy0)


# ---------------------------------------------------------------------------
# 2. Rotated uniform grid
# ---------------------------------------------------------------------------


def test_rotated_uniform_grid_preserves_widths_and_reports_rotation_angle() -> None:
    rot = np.pi / 6
    x, y = _rotated_grid(angle_rad=rot)

    assert np.allclose(orthogonality_error(x, y), 0.0, atol=1.0e-12)
    dx, dy = cell_widths(x, y)
    assert np.allclose(dx, 2.0)
    assert np.allclose(dy, 1.5)

    ang = grid_angle(x, y)
    assert np.allclose(ang, rot, atol=1.0e-10)


# ---------------------------------------------------------------------------
# 3. Sheared grid (non-orthogonal by construction)
# ---------------------------------------------------------------------------


def test_sheared_grid_reports_consistent_signed_and_unsigned_orthogonality() -> None:
    shear = 0.3
    x, y = _sheared_grid(shear=shear)

    err = orthogonality_error(x, y)
    signed = orthogonality_signed(x, y)
    assert err.shape == signed.shape

    expected = abs(np.arctan2(shear, 1.0))
    assert np.allclose(err, expected)
    assert np.allclose(np.abs(signed), expected)
    assert np.all(signed < 0.0)

    # pm unaffected by shear along x (dx of horizontal rows is unchanged
    # because each cell is displaced by the same delta in x).
    pm, _pn = metric_coefficients(x, y)
    dx, _ = cell_widths(x, y)
    assert np.allclose(dx, 2.0)
    assert np.allclose(pm, 0.5)


# ---------------------------------------------------------------------------
# 4. Annular polar patch
# ---------------------------------------------------------------------------


def test_polar_patch_is_orthogonal_and_has_radially_varying_pm() -> None:
    x, y = _polar_patch(Mp=40, Lp=48, r0=1.0, r1=2.5)

    # Discretized polar edges are chords, not tangents, so finite-resolution
    # grids exhibit a small uniform orthogonality bias proportional to
    # d(theta)/2. For a 40x48 patch this is well below 1 degree.
    err = orthogonality_error(x, y)
    assert np.max(err) < np.deg2rad(1.0)

    # The bias should decrease with resolution.
    x_coarse, y_coarse = _polar_patch(Mp=10, Lp=12, r0=1.0, r1=2.5)
    err_coarse = orthogonality_error(x_coarse, y_coarse)
    assert np.max(err) < np.max(err_coarse)

    pm, pn = metric_coefficients(x, y)
    # With indexing='ij' on (r, theta), theta varies along axis=1 (xi).
    # So dx = r*d(theta) grows with radius -> pm = 1/dx decreases with r.
    # dy = dr is constant -> pn constant.
    pm_rows = np.mean(pm, axis=1)
    assert np.all(np.diff(pm_rows) < 0.0)
    assert np.std(pn) < 0.05

    # dmde captures the radial variation of dx = r*d(theta).
    _dndx, dmde = metric_derivatives(x, y)
    assert np.any(np.abs(dmde[1:-1, 1:-1]) > 1.0e-6)


# ---------------------------------------------------------------------------
# 5. Focus applied to uniform channel
# ---------------------------------------------------------------------------


def test_focus_produces_monotonic_pm_toward_target() -> None:
    focus = Focus()
    focus.add_focus(pos=0.3, axis="x", factor=3.0, extent=0.15)
    boundary = BoundaryDefinition(
        x=np.array([0.0, 10.0, 10.0, 0.0]),
        y=np.array([0.0, 0.0, 4.0, 4.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    result = GridGenerator(mode=SolverMode.TRANSFINITE, focus=focus).generate(
        boundary, shape=(6, 41)
    )
    # Along a row (constant eta), dx should be smallest near pos=0.3 and
    # larger away from it.
    dx = result.dx
    row = dx[dx.shape[0] // 2]
    target_i = int(round(0.3 * (dx.shape[1] - 1)))
    assert row[target_i] < row[0]
    assert row[target_i] < row[-1]


# ---------------------------------------------------------------------------
# 6. Grid metric-property contract
# ---------------------------------------------------------------------------


def test_grid_exposes_expected_metric_shapes_for_uniform_grid() -> None:
    Mp, Lp = 5, 7
    x, y = _uniform_grid(Mp, Lp)
    grid = Grid(x, y)

    x_rho, _ = rho_points(grid.x, grid.y)
    x_psi, _ = psi_points(grid.x, grid.y)
    x_u, _ = u_points(grid.x, grid.y)
    x_v, _ = v_points(grid.x, grid.y)
    assert grid.x.shape == (Mp + 1, Lp + 1)
    assert x_rho.shape == (Mp, Lp)
    assert x_psi.shape == (Mp - 1, Lp - 1)
    assert x_u.shape == (Mp, Lp - 1)
    assert x_v.shape == (Mp - 1, Lp)

    assert grid.dx.shape == (Mp, Lp)
    assert grid.dy.shape == (Mp, Lp)
    assert grid.pm.shape == (Mp, Lp)
    assert grid.pn.shape == (Mp, Lp)
    dndx, dmde = metric_derivatives(grid.x, grid.y)
    assert dndx.shape == (Mp, Lp)
    assert dmde.shape == (Mp, Lp)

    assert grid.angle.shape == (Mp + 1, Lp + 1)

    assert orthogonality_error(grid.x, grid.y).shape == (Mp, Lp)
    assert orthogonality_signed(grid.x, grid.y).shape == (Mp, Lp)
    assert orthogonality_corner_angles(grid.x, grid.y).shape == (4, Mp, Lp)

    assert grid.mask_rho.shape == (Mp, Lp)
    assert grid.mask_u.shape == (Mp, Lp - 1)
    assert grid.mask_v.shape == (Mp - 1, Lp)
    assert grid.mask_psi.shape == (Mp - 1, Lp - 1)

    assert np.all(grid.mask_rho)
    assert np.all(grid.mask_u)
    assert np.all(grid.mask_v)
    assert np.all(grid.mask_psi)


# ---------------------------------------------------------------------------
# 7. NaN vertex propagates through rho and uvp masks
# ---------------------------------------------------------------------------


def test_nan_vertex_propagates_to_rho_and_uvp_masks() -> None:
    Mp, Lp = 5, 6
    x, y = _uniform_grid(Mp, Lp)
    x = x.copy()
    y = y.copy()
    x[2, 3] = np.nan
    y[2, 3] = np.nan

    mask = rho_mask_from_vertices(x, y)
    assert mask.shape == (Mp, Lp)
    # The four rho cells sharing the corrupted vertex should be False.
    expected_false_cells = {(1, 2), (1, 3), (2, 2), (2, 3)}
    for j in range(Mp):
        for i in range(Lp):
            if (j, i) in expected_false_cells:
                assert not mask[j, i], f"expected mask[{j},{i}] False"
            else:
                assert mask[j, i], f"expected mask[{j},{i}] True"

    u_mask, v_mask, psi_mask = uvp_masks(mask)
    # Spot-check that at least the u/v edges sharing the bad rho cells are
    # also masked.
    assert not u_mask[2, 2]
    assert not v_mask[1, 3]
    assert not psi_mask[1, 2]


# ---------------------------------------------------------------------------
# 8. grid_angle unwrapping tolerates NaN holes
# ---------------------------------------------------------------------------


def test_grid_angle_unwrap_survives_nan_column() -> None:
    x, y = _uniform_grid(Mp=6, Lp=10)
    x = x.copy()
    y = y.copy()
    # Blank out one interior column of vertices.
    x[:, 5] = np.nan
    y[:, 5] = np.nan

    ang = grid_angle(x, y, unwrap=True)
    assert ang.shape == x.shape
    finite = np.isfinite(ang)
    # The angles at finite locations should still be near zero (uniform
    # horizontal grid), despite the NaN band.
    assert np.allclose(ang[finite], 0.0, atol=1.0e-12)
    # NaN band is preserved somewhere inside the array.
    assert not np.all(finite)


# ---------------------------------------------------------------------------
# 9. quality.py backward-compat shim
# ---------------------------------------------------------------------------


def test_metrics_module_exposes_grid_metric_symbols() -> None:
    from cgrid_tools.metrics import cell_areas as metrics_area
    from cgrid_tools.metrics import orthogonality_error as metrics_orth

    x, y = _uniform_grid()
    assert np.allclose(metrics_area(x, y), cell_areas(x, y))
    assert np.allclose(metrics_orth(x, y), orthogonality_error(x, y))


# ---------------------------------------------------------------------------
# 10. GridGenerator/Grid integration
# ---------------------------------------------------------------------------


def test_grid_generator_returns_grid_with_metric_properties() -> None:
    boundary = BoundaryDefinition(
        x=np.array([0.0, 4.0, 4.0, 0.0]),
        y=np.array([0.0, 0.0, 2.0, 2.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    grid = GridGenerator(mode=SolverMode.TRANSFINITE).generate(boundary, shape=(5, 9))

    assert isinstance(grid, Grid)
    x_rho, _ = rho_points(grid.x, grid.y)
    assert x_rho.shape == (grid.x.shape[0] - 1, grid.x.shape[1] - 1)

    summary = orthogonality_summary(grid.x, grid.y)
    assert set(summary) == {"mean_abs", "max_abs", "rms", "signed_mean"}
    assert np.isfinite(summary["mean_abs"])
    assert np.isfinite(summary["max_abs"])
    assert summary["max_abs"] >= summary["mean_abs"]

    # dx/dy/pm/pn/angle all work and have consistent shapes.
    assert grid.dx.shape == (grid.x.shape[0] - 1, grid.x.shape[1] - 1)
    assert grid.dy.shape == grid.dx.shape
    assert grid.pm.shape == grid.dx.shape
    assert grid.pn.shape == grid.dx.shape
    assert grid.angle.shape == grid.x.shape
    # Orthogonality is rho-cell shaped.
    assert orthogonality_error(grid.x, grid.y).shape == grid.dx.shape
    assert orthogonality_signed(grid.x, grid.y).shape == grid.dx.shape


# ---------------------------------------------------------------------------
# Subgrid-point helpers
# ---------------------------------------------------------------------------


def test_rho_u_v_psi_points_return_expected_shapes_and_values() -> None:
    Mp, Lp = 4, 5
    x, y = _uniform_grid(Mp, Lp, dx0=1.0, dy0=1.0)

    xr, yr = rho_points(x, y)
    assert xr.shape == (Mp, Lp)
    # Rho point (0,0) centers the (0,0) cell at (0.5, 0.5).
    assert np.isclose(xr[0, 0], 0.5)
    assert np.isclose(yr[0, 0], 0.5)

    xu, _yu = u_points(x, y)
    assert xu.shape == (Mp, Lp - 1)

    xv, _yv = v_points(x, y)
    assert xv.shape == (Mp - 1, Lp)

    xp, _yp = psi_points(x, y)
    assert xp.shape == (Mp - 1, Lp - 1)


# ---------------------------------------------------------------------------
# Grid preserves explicit mask
# ---------------------------------------------------------------------------


def test_grid_honors_explicit_mask() -> None:
    Mp, Lp = 4, 6
    x, y = _uniform_grid(Mp, Lp)
    rho_mask = np.ones((Mp, Lp), dtype=bool)
    rho_mask[1, 2] = False
    grid = Grid(x, y, mask_rho=rho_mask)
    assert not grid.mask_rho[1, 2]
    assert grid.mask_rho.sum() == rho_mask.sum()

    with pytest.raises(ValueError):
        Grid(x, y, mask_rho=np.zeros((Mp + 1, Lp), dtype=bool))


# ---------------------------------------------------------------------------
# Corner-angle accessor matches signed mean
# ---------------------------------------------------------------------------


def test_orthogonality_corner_angles_average_matches_signed_metric() -> None:
    x, y = _sheared_grid(shear=0.2)
    corners = orthogonality_corner_angles(x, y)
    signed = orthogonality_signed(x, y)
    # The signed metric uses mean(|abs(ang_k + pi/2) - pi/2|); directly
    # reconstruct here to ensure the helpers are consistent.
    abs_angles = np.abs(corners + np.pi / 2.0)
    reconstructed = np.mean(abs_angles, axis=0) - np.pi / 2.0
    assert np.allclose(reconstructed, signed)
