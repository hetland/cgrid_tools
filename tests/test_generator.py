import numpy as np
import pytest

import cgrid_tools.gridgen.generator as generator
from cgrid_tools import orthogonality_error
from cgrid_tools.gridgen import BoundaryDefinition, Focus, GridGenerator, SolverMode


def test_boundary_definition_closes_open_polygon_inputs() -> None:
    boundary = BoundaryDefinition(
        x=np.array([0.0, 2.0, 2.0, 0.0]),
        y=np.array([0.0, 0.0, 1.0, 1.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0]),
    )

    assert np.allclose(boundary.x, [0.0, 2.0, 2.0, 0.0, 0.0])
    assert np.allclose(boundary.y, [0.0, 0.0, 1.0, 1.0, 0.0])
    assert np.allclose(boundary.beta, [1.0, 1.0, 1.0, 1.0, 0.0])


def test_boundary_definition_accepts_open_polygon_with_closing_beta() -> None:
    boundary = BoundaryDefinition(
        x=np.array([0.0, 2.0, 2.0, 0.0]),
        y=np.array([0.0, 0.0, 1.0, 1.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
    )

    assert np.allclose(boundary.x, [0.0, 2.0, 2.0, 0.0, 0.0])
    assert np.allclose(boundary.y, [0.0, 0.0, 1.0, 1.0, 0.0])
    assert np.allclose(boundary.beta, [1.0, 1.0, 1.0, 1.0, 0.0])


def test_transfinite_generation_for_simple_trapezoid() -> None:
    boundary = BoundaryDefinition(
        x=np.array([0.5, 2.0, 2.5, 1.5, 0.5]),
        y=np.array([0.5, 0.5, 2.0, 2.0, 0.5]),
        beta=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
    )
    result = GridGenerator(mode=SolverMode.TRANSFINITE).generate(boundary, shape=(10, 5))

    assert result.x.shape == (10, 5)
    assert result.y.shape == (10, 5)
    assert np.all(np.isfinite(result.x))
    assert np.all(np.isfinite(result.y))


def test_elliptic_mode_preserves_boundary_nodes() -> None:
    boundary = BoundaryDefinition(
        x=np.array([0.0, 2.0, 2.0, 0.0, 0.0]),
        y=np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
    )
    result = GridGenerator(mode=SolverMode.ELLIPTIC, smoothing_iterations=25).generate(
        boundary,
        shape=(6, 8),
    )

    assert np.allclose(result.y[0, :], 0.0)
    assert np.allclose(result.y[-1, :], 1.0)
    assert np.allclose(result.x[:, 0], 0.0)
    assert np.allclose(result.x[:, -1], 2.0)


def test_laplacian_mode_preserves_boundary_nodes() -> None:
    boundary = BoundaryDefinition(
        x=np.array([0.0, 2.0, 2.0, 0.0, 0.0]),
        y=np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
    )
    result = GridGenerator(mode=SolverMode.LAPLACIAN, smoothing_iterations=25).generate(
        boundary,
        shape=(6, 8),
    )

    assert np.allclose(result.y[0, :], 0.0)
    assert np.allclose(result.y[-1, :], 1.0)
    assert np.allclose(result.x[:, 0], 0.0)
    assert np.allclose(result.x[:, -1], 2.0)


def test_logical_side_extraction_handles_side_channel_boundary() -> None:
    boundary = BoundaryDefinition(
        x=np.array([0.5, 2.0, 2.0, 3.5, 3.5, 2.0, 2.0, 0.5, 0.5]),
        y=np.array([0.5, 0.5, 1.75, 1.75, 2.25, 2.25, 3.5, 3.5, 0.5]),
        beta=np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 0.0]),
    )

    with pytest.raises(NotImplementedError, match="GRIDGEN_CRDT"):
        boundary.logical_sides()


@pytest.mark.parametrize(
    "mode", [SolverMode.TRANSFINITE, SolverMode.LAPLACIAN, SolverMode.ELLIPTIC]
)
def test_four_side_solvers_reject_reentrant_topology(mode) -> None:
    boundary = BoundaryDefinition(
        x=np.array([0.5, 2.0, 2.0, 3.5, 3.5, 2.0, 2.0, 0.5, 0.5]),
        y=np.array([0.5, 0.5, 1.75, 1.75, 2.25, 2.25, 3.5, 3.5, 0.5]),
        beta=np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 0.0]),
    )

    with pytest.raises(NotImplementedError, match="GRIDGEN_CRDT"):
        GridGenerator(mode=mode).generate(boundary, shape=(21, 21))


def test_elliptic_solver_changes_interior_for_curved_boundary() -> None:
    boundary = BoundaryDefinition(
        x=np.array([0.0, 1.0, 2.2, 2.0, 0.0]),
        y=np.array([0.0, -0.1, 1.2, 2.0, 0.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
    )

    transfinite = GridGenerator(mode=SolverMode.TRANSFINITE).generate(boundary, shape=(12, 10))
    elliptic = GridGenerator(mode=SolverMode.ELLIPTIC, smoothing_iterations=100).generate(
        boundary,
        shape=(12, 10),
    )

    assert np.allclose(elliptic.x[0, :], transfinite.x[0, :])
    assert np.allclose(elliptic.y[:, 0], transfinite.y[:, 0])
    assert not np.allclose(elliptic.x[1:-1, 1:-1], transfinite.x[1:-1, 1:-1])

    # Smoke bound: the elliptic solver should not produce a catastrophically
    # non-orthogonal grid on a mildly curved boundary. The bound is loose;
    # its purpose is to catch regressions, not enforce solver-quality targets.
    max_err = float(np.nanmax(orthogonality_error(elliptic.x, elliptic.y)))
    assert max_err < np.pi / 4.0, (
        f"elliptic orthogonality max {np.degrees(max_err):.2f} deg exceeds " f"45 deg smoke bound"
    )


def test_focus_clusters_x_coordinates_near_target_position() -> None:
    focus = Focus()
    focus.add_focus(0.5, axis="x", factor=3.0, extent=0.2)

    boundary = BoundaryDefinition(
        x=np.array([0.0, 2.0, 2.6, 0.2, 0.0]),
        y=np.array([0.0, 0.0, 1.2, 1.0, 0.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
    )

    uniform = GridGenerator(mode=SolverMode.TRANSFINITE).generate(boundary, shape=(8, 21))
    focused = GridGenerator(mode=SolverMode.TRANSFINITE, focus=focus).generate(
        boundary, shape=(8, 21)
    )

    uniform_dx = np.diff(uniform.x[4, :])
    focused_dx = np.diff(focused.x[4, :])

    assert focused_dx[9] < uniform_dx[9]
    assert focused_dx[0] > 0.0
    assert np.allclose(focused.x[:, 0], uniform.x[:, 0])
    assert np.isclose(focused.x[0, 0], uniform.x[0, 0])
    assert np.isclose(focused.x[0, -1], uniform.x[0, -1])
    assert np.all(np.diff(focused.x[0, :]) > 0.0)


def test_focus_can_cluster_y_for_side_channel_case() -> None:
    focus = Focus()
    focus.add_focus(0.5, axis="y", factor=5.0, extent=0.25)

    boundary = BoundaryDefinition(
        x=np.array([0.5, 2.0, 2.0, 3.5, 3.5, 2.0, 2.0, 0.5, 0.5]),
        y=np.array([0.5, 0.5, 1.75, 1.75, 2.25, 2.25, 3.5, 3.5, 0.5]),
        beta=np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 0.0]),
    )

    result = GridGenerator(mode=SolverMode.GRIDGEN_CRDT, focus=focus).generate(
        boundary, shape=(15, 15)
    )

    assert result.x.shape == (15, 15)
    assert result.y.shape == (15, 15)
    assert np.any(result.mask_rho)
    assert np.any(~result.mask_rho)


def test_laplacian_smoothing_preserves_focused_interior_stripe() -> None:
    focus = Focus()
    focus.add_focus(0.5, axis="x", factor=4.0, extent=0.2)
    boundary = BoundaryDefinition(
        x=np.array([0.0, 1.0, 1.0, 0.0]),
        y=np.array([0.0, 0.0, 1.0, 1.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0]),
    )

    focused = GridGenerator(
        mode=SolverMode.LAPLACIAN,
        focus=focus,
        smoothing_iterations=75,
    ).generate(boundary, shape=(11, 31))
    uniform = GridGenerator(
        mode=SolverMode.LAPLACIAN,
        smoothing_iterations=75,
    ).generate(boundary, shape=(11, 31))

    focused_dx = np.diff(focused.x[5, :])
    uniform_dx = np.diff(uniform.x[5, :])

    assert focused_dx[14] < uniform_dx[14]
    assert focused_dx[0] > uniform_dx[0]
    assert np.all(np.diff(focused.x[5, :]) > 0.0)


def test_elliptic_smoothing_preserves_focused_interior_stripe() -> None:
    focus = Focus()
    focus.add_focus(0.5, axis="x", factor=4.0, extent=0.2)
    boundary = BoundaryDefinition(
        x=np.array([0.0, 1.0, 1.0, 0.0]),
        y=np.array([0.0, 0.0, 1.0, 1.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0]),
    )

    focused = GridGenerator(
        mode=SolverMode.ELLIPTIC,
        focus=focus,
        smoothing_iterations=75,
    ).generate(boundary, shape=(11, 31))
    uniform = GridGenerator(
        mode=SolverMode.ELLIPTIC,
        smoothing_iterations=75,
    ).generate(boundary, shape=(11, 31))

    focused_dx = np.diff(focused.x[5, :])
    uniform_dx = np.diff(uniform.x[5, :])

    assert focused_dx[14] < uniform_dx[14]
    assert focused_dx[0] > uniform_dx[0]
    assert np.all(np.diff(focused.x[5, :]) > 0.0)


def test_elliptic_numba_kernel_matches_python_reference() -> None:
    focus = Focus()
    focus.add_focus(0.5, axis="x", factor=3.0, extent=0.2)
    boundary = BoundaryDefinition(
        x=np.array([0.0, 1.0, 1.2, 0.0]),
        y=np.array([0.0, -0.1, 1.0, 1.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    computational_x, computational_y = generator._computational_coordinates((14, 18), focus)
    edges = generator._extract_edges(boundary, computational_x, computational_y)
    x, y = generator._transfinite_interpolation(
        *edges,
        computational_x=computational_x,
        computational_y=computational_y,
    )
    cell_mask = generator._cell_mask_from_boundary(x, y, boundary)
    update_mask = generator._node_update_mask(cell_mask)

    accelerated_x, accelerated_y, accelerated_iterations = generator._elliptic_smooth(
        x,
        y,
        computational_x,
        computational_y,
        update_mask,
        max_iterations=35,
        relaxation=1.25,
        tolerance=0.0,
    )
    reference_x, reference_y, reference_iterations = generator._elliptic_smooth_python(
        x,
        y,
        computational_x,
        computational_y,
        update_mask,
        max_iterations=35,
        relaxation=1.25,
        tolerance=0.0,
    )

    assert accelerated_iterations == reference_iterations
    assert np.allclose(accelerated_x, reference_x)
    assert np.allclose(accelerated_y, reference_y)


def test_elliptic_numba_kernel_matches_python_reference_on_masked_side_channel() -> None:
    """Numba/python parity holds even when the update_mask excludes cells.

    The side-channel boundary produces a non-rectangular interior whose
    ``update_mask`` has interior False entries. This exercises the masked
    path of both kernels.
    """
    boundary = BoundaryDefinition(
        x=np.array([0.5, 2.0, 2.0, 3.5, 3.5, 2.0, 2.0, 0.5, 0.5]),
        y=np.array([0.5, 0.5, 1.75, 1.75, 2.25, 2.25, 3.5, 3.5, 0.5]),
        beta=np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 0.0]),
    )
    computational_x, computational_y = generator._computational_coordinates((24, 24), None)
    outer = BoundaryDefinition([0.5, 3.5, 3.5, 0.5], [0.5, 0.5, 3.5, 3.5], [1, 1, 1, 1])
    edges = generator._extract_edges(outer, computational_x, computational_y)
    x, y = generator._transfinite_interpolation(
        *edges,
        computational_x=computational_x,
        computational_y=computational_y,
    )
    cell_mask = generator._cell_mask_from_boundary(x, y, boundary)
    update_mask = generator._node_update_mask(cell_mask)
    # Sanity: the mask must have at least one excluded interior node for
    # this test to exercise the masked path meaningfully.
    assert np.any(~update_mask[1:-1, 1:-1])

    acc_x, acc_y, acc_iter = generator._elliptic_smooth(
        x.copy(),
        y.copy(),
        computational_x,
        computational_y,
        update_mask,
        max_iterations=20,
        relaxation=1.2,
        tolerance=0.0,
    )
    ref_x, ref_y, ref_iter = generator._elliptic_smooth_python(
        x.copy(),
        y.copy(),
        computational_x,
        computational_y,
        update_mask,
        max_iterations=20,
        relaxation=1.2,
        tolerance=0.0,
    )
    assert acc_iter == ref_iter
    assert np.allclose(acc_x, ref_x)
    assert np.allclose(acc_y, ref_y)


def test_negative_beta_side_channel_returns_excluded_cell_mask() -> None:
    boundary = BoundaryDefinition(
        x=np.array([0.5, 2.0, 2.0, 3.5, 3.5, 2.0, 2.0, 0.5, 0.5]),
        y=np.array([0.5, 0.5, 1.75, 1.75, 2.25, 2.25, 3.5, 3.5, 0.5]),
        beta=np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 0.0]),
    )

    result = GridGenerator(mode=SolverMode.GRIDGEN_CRDT).generate(
        boundary,
        shape=(30, 30),
    )

    assert result.mask_rho.shape == (29, 29)
    assert np.any(result.mask_rho)
    assert np.any(~result.mask_rho)


# ---------------------------------------------------------------------------
# BoundaryDefinition.from_lonlat
# ---------------------------------------------------------------------------


def test_boundary_definition_from_lonlat_applies_projection_callable() -> None:
    """``from_lonlat`` delegates to the supplied projection callable."""

    def simple_projection(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Trivial equirectangular-ish projection scaled so small inputs map
        # to Cartesian units without depending on pyproj.
        return 111_000.0 * lon, 111_000.0 * lat

    lon = np.array([-122.40, -122.35, -122.35, -122.40])
    lat = np.array([37.70, 37.70, 37.75, 37.75])
    beta = np.array([1.0, 1.0, 1.0, 1.0])

    boundary = BoundaryDefinition.from_lonlat(lon, lat, beta, simple_projection)

    expected_x, expected_y = simple_projection(lon, lat)
    assert np.allclose(boundary.x[:-1], expected_x)
    assert np.allclose(boundary.y[:-1], expected_y)
    assert np.isclose(boundary.x[-1], expected_x[0])
    assert np.isclose(boundary.y[-1], expected_y[0])
    assert boundary.source_crs == "EPSG:4326"
    assert boundary.projected is False
    # Closing beta appended automatically.
    assert np.allclose(boundary.beta, [1.0, 1.0, 1.0, 1.0, 0.0])


def test_boundary_definition_from_lonlat_accepts_custom_source_crs() -> None:
    def identity_projection(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(lon, dtype=float).copy(), np.asarray(lat, dtype=float).copy()

    lon = np.array([0.0, 1.0, 1.0, 0.0])
    lat = np.array([0.0, 0.0, 1.0, 1.0])
    beta = np.array([1.0, 1.0, 1.0, 1.0])

    boundary = BoundaryDefinition.from_lonlat(
        lon, lat, beta, identity_projection, source_crs="EPSG:3857"
    )
    assert boundary.source_crs == "EPSG:3857"


# ---------------------------------------------------------------------------
# Focus.focused_coordinates
# ---------------------------------------------------------------------------


def test_focus_focused_coordinates_returns_monotonic_grid_over_unit_square() -> None:
    focus = Focus()
    focus.add_focus(pos=0.5, axis="x", factor=2.0, extent=0.2)

    xi, eta = focus.focused_coordinates((4, 7))
    # Shape matches the requested (ny, nx).
    assert xi.shape == (4, 7)
    assert eta.shape == (4, 7)
    # Endpoints pinned at 0 and 1 on each row.
    assert np.allclose(xi[:, 0], 0.0)
    assert np.allclose(xi[:, -1], 1.0)
    # Monotonic increase along the focused axis.
    assert np.all(np.diff(xi, axis=1) > 0.0)
    # Untouched axis is a uniform linspace replicated per column.
    assert np.allclose(eta[0, 0], 0.0)
    assert np.allclose(eta[-1, 0], 1.0)
    # Spacing near the focus target should be smaller than spacing near the
    # edges (focus factor > 1 clusters nodes at pos).
    row = xi[0]
    target_i = int(round(0.5 * (row.size - 1)))
    assert np.diff(row)[target_i] < np.diff(row)[0]


def test_focus_focused_coordinates_passthrough_when_no_focus_points() -> None:
    focus = Focus()
    xi, eta = focus.focused_coordinates((3, 5))
    expected_xi = np.broadcast_to(np.linspace(0.0, 1.0, 5), (3, 5))
    expected_eta = np.broadcast_to(np.linspace(0.0, 1.0, 3)[:, None], (3, 5))
    assert np.allclose(xi, expected_xi)
    assert np.allclose(eta, expected_eta)


@pytest.mark.parametrize(
    "mode", [SolverMode.TRANSFINITE, SolverMode.LAPLACIAN, SolverMode.ELLIPTIC]
)
@pytest.mark.parametrize("axis", ["x", "y"])
def test_asymmetric_focus_keeps_rectangle_orthogonal(mode, axis):
    boundary = BoundaryDefinition([0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1])
    focus = Focus()
    focus.add_focus(0.2, axis, factor=5, extent=0.1)
    grid = GridGenerator(mode=mode, focus=focus).generate(boundary, (21, 41))
    assert np.allclose(grid.x[0], grid.x[-1])
    assert np.allclose(grid.y[:, 0], grid.y[:, -1])
    assert np.max(orthogonality_error(grid.x, grid.y)) < 1e-6
    values = grid.x[0] if axis == "x" else grid.y[:, 0]
    assert np.argmin(np.diff(values)) < len(values) // 2


@pytest.mark.parametrize("offset", [0.0, 1e6])
def test_projected_boundary_closure_is_translation_invariant(offset):
    boundary = BoundaryDefinition(
        np.array([0, 100, 100, 0]) + offset, np.array([0, 0, 5, 5]) + offset, [1, 1, 1, 1]
    )
    grid = GridGenerator(mode=SolverMode.TRANSFINITE).generate(boundary, (6, 11))
    assert len(boundary.vertices) == 4
    assert np.allclose(grid.x - offset, np.linspace(0, 100, 11)[None, :])
    assert np.allclose(grid.y - offset, np.linspace(0, 5, 6)[:, None])


def test_lowest_mid_edge_vertex_does_not_become_logical_corner():
    boundary = BoundaryDefinition([0, 0.5, 1, 1, 0], [0, -0.1, 0, 1, 1], [1, 0, 1, 1, 1])
    sides = boundary.logical_sides()
    assert sides.south.shape == (3, 2)
    grid = GridGenerator(mode=SolverMode.TRANSFINITE).generate(boundary, (7, 11))
    assert np.isclose(grid.y[0, 5], -0.1)
    assert np.all(grid.mask_rho)


@pytest.mark.parametrize("mode", list(SolverMode))
def test_geographic_generation_inverse_projects_output(mode):
    pyproj = pytest.importorskip("pyproj")
    proj = pyproj.Proj("EPSG:32636")
    boundary = BoundaryDefinition.from_lonlat(
        [36, 37, 37, 36], [35, 35, 36, 36], [1, 1, 1, 1], proj
    )
    grid = GridGenerator(mode=mode).generate(boundary, (5, 7))
    assert grid.lon.shape == grid.shape
    assert grid.lat.shape == grid.shape
    assert grid.f.shape == (4, 6)
    assert grid.crs == "EPSG:32636"
    x, y = proj(grid.lon, grid.lat)
    assert np.allclose(grid.x, x)
    assert np.allclose(grid.y, y)
    assert boundary.normalized().original_lon is not None


def test_geographic_generation_with_target_crs_and_custom_inverse():
    pytest.importorskip("pyproj")
    boundary = BoundaryDefinition.from_lonlat(
        [0, 1, 1, 0], [0, 0, 1, 1], [1] * 4, target_crs="EPSG:3857"
    )
    grid = GridGenerator(mode=SolverMode.TRANSFINITE).generate(boundary, (4, 5))
    assert np.allclose(grid.lon[0], np.linspace(0, 1, 5))
    assert grid.crs == "EPSG:3857"
    boundary = BoundaryDefinition.from_lonlat(
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1] * 4,
        lambda lon, lat: (2 * lon, 3 * lat),
        inverse_projection=lambda x, y: (x / 2, y / 3),
    )
    grid = GridGenerator(mode=SolverMode.TRANSFINITE).generate(boundary, (4, 5))
    assert np.allclose(grid.lat[:, 0], np.linspace(0, 1, 4))


def test_geographic_generation_rejects_missing_custom_inverse():
    boundary = BoundaryDefinition.from_lonlat(
        [0, 1, 1, 0], [0, 0, 1, 1], [1] * 4, lambda lon, lat: (lon, lat)
    )
    with pytest.raises(ValueError, match="inverse_projection"):
        GridGenerator().generate(boundary, (4, 5))


@pytest.mark.parametrize("mode", list(SolverMode))
def test_self_intersecting_boundaries_are_rejected(mode):
    boundary = BoundaryDefinition([0, 1, 0, 1], [0, 1, 1, 0], [1] * 4)
    with pytest.raises(ValueError, match="self-intersecting"):
        GridGenerator(mode=mode).generate(boundary, (11, 11))


def test_clockwise_boundary_is_normalized_to_positive_cells():
    from cgrid_tools import cell_jacobians

    boundary = BoundaryDefinition([0, 0, 1, 1], [0, 1, 1, 0], [1] * 4)
    grid = GridGenerator().generate(boundary, (5, 7))
    assert np.all(cell_jacobians(grid.x, grid.y) > 0)


def test_generated_folded_cells_are_rejected(monkeypatch):
    from cgrid_tools.gridgen import GridGenerationError

    def folded(*args, **kwargs):
        return np.array([[0.0, 1.0], [0.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 0.0]])

    monkeypatch.setattr(generator, "_transfinite_interpolation", folded)
    boundary = BoundaryDefinition([0, 1, 1, 0], [0, 0, 1, 1], [1] * 4)
    with pytest.raises(GridGenerationError, match="folded or degenerate"):
        GridGenerator(mode=SolverMode.TRANSFINITE).generate(boundary, (2, 2))


def test_nonfinite_solver_output_is_not_silently_masked(monkeypatch):
    from cgrid_tools.gridgen import GridGenerationError

    original = generator._transfinite_interpolation

    def nonfinite(*args, **kwargs):
        x, y = original(*args, **kwargs)
        x[2, 2] = np.nan
        return x, y

    monkeypatch.setattr(generator, "_transfinite_interpolation", nonfinite)
    boundary = BoundaryDefinition([0, 1, 1, 0], [0, 0, 1, 1], [1] * 4)
    with pytest.raises(GridGenerationError, match="non-finite vertices"):
        GridGenerator(mode=SolverMode.TRANSFINITE).generate(boundary, (5, 5))


def test_projection_object_honors_source_datum():
    pyproj = pytest.importorskip("pyproj")
    lon, lat = [-101, -100, -100, -101], [39, 39, 40, 40]
    boundary = BoundaryDefinition.from_lonlat(
        lon, lat, [1] * 4, pyproj.Proj("EPSG:32614"), source_crs="EPSG:4267"
    )
    expected_x, expected_y = pyproj.Transformer.from_crs(
        "EPSG:4267", "EPSG:32614", always_xy=True
    ).transform(lon, lat)
    np.testing.assert_allclose(boundary.x[:-1], expected_x, rtol=0, atol=1e-7)
    np.testing.assert_allclose(boundary.y[:-1], expected_y, rtol=0, atol=1e-7)
    lon_back, lat_back = boundary.geographic_vertices(boundary.x, boundary.y)
    np.testing.assert_allclose(lon_back[:-1], lon, rtol=0, atol=1e-7)
    np.testing.assert_allclose(lat_back[:-1], lat, rtol=0, atol=1e-7)


def test_projection_object_rejects_mismatched_crs_metadata():
    pyproj = pytest.importorskip("pyproj")
    with pytest.raises(ValueError, match="target_crs must match"):
        BoundaryDefinition.from_lonlat(
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1] * 4,
            pyproj.Proj("EPSG:32631"),
            target_crs="EPSG:32632",
        )


def test_elliptic_last_iteration_convergence_is_reported():
    boundary = BoundaryDefinition([0, 1, 1, 0], [0, 0, 1, 1], [1] * 4)
    grid = GridGenerator(smoothing_iterations=1).generate(boundary, (5, 5))
    assert grid.diagnostics["converged"] is True
    assert grid.diagnostics["max_update"] < 1e-8
