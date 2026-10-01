import numpy as np
import pytest

import cgrid_tools.gridgen.crdt as crdt
from cgrid_tools.gridgen import (
    BoundaryDefinition,
    CRDTOptions,
    GridGenerator,
    SchwarzChristoffelTransform,
    SolverMode,
    prepare_crdt_topology,
    solve_crdt_state,
)
from cgrid_tools.metrics import orthogonality_error

ORTHOGONALITY_MEAN_TOLERANCE = np.deg2rad(4.5)
ORTHOGONALITY_P95_TOLERANCE = np.deg2rad(15.0)
EDGE_SPACING_CV_TOLERANCE = 0.30
EDGE_SPACING_P95_P5_TOLERANCE = 2.1
MONOTONIC_TOLERANCE = 1.0e-3


def _tilted_channel_boundary() -> BoundaryDefinition:
    return BoundaryDefinition(
        x=np.array([3.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0]),
        y=np.array([4.0, 4.0, 3.0, 4.0, 3.0, 2.0, 1.0, 1.0]),
        beta=np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]),
    )


def _angle_matched_tilted_channel_image_beta() -> np.ndarray:
    return np.array([1.0, 1.0, -1.5, 1.5, 0.5, -0.5, 1.0, 1.0])


def _finite_runs(values: np.ndarray, finite: np.ndarray) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    start: int | None = None
    for idx, is_finite in enumerate(finite):
        if is_finite and start is None:
            start = idx
        elif not is_finite and start is not None:
            runs.append(values[start:idx])
            start = None
    if start is not None:
        runs.append(values[start:])
    return [run for run in runs if len(run) >= 3]


def _is_monotonic(values: np.ndarray) -> bool:
    deltas = np.diff(values)
    return bool(np.all(deltas >= -MONOTONIC_TOLERANCE) or np.all(deltas <= MONOTONIC_TOLERANCE))


def _nonmonotonic_runs(x: np.ndarray, y: np.ndarray) -> list[tuple[str, str, int]]:
    finite = np.isfinite(x) & np.isfinite(y)
    failures: list[tuple[str, str, int]] = []
    for row in range(x.shape[0]):
        for name, values in (("x", x[row]), ("y", y[row])):
            for run in _finite_runs(values, finite[row]):
                if not _is_monotonic(run):
                    failures.append(("row", name, row))
                    break
    for column in range(x.shape[1]):
        for name, values in (("x", x[:, column]), ("y", y[:, column])):
            for run in _finite_runs(values, finite[:, column]):
                if not _is_monotonic(run):
                    failures.append(("column", name, column))
                    break
    return failures


def _valid_node_edge_lengths(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y)
    horizontal = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
    vertical = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
    horizontal_valid = finite[:, :-1] & finite[:, 1:]
    vertical_valid = finite[:-1, :] & finite[1:, :]
    return np.concatenate((horizontal[horizontal_valid], vertical[vertical_valid]))


def _local_mean_edge_lengths(
    x: np.ndarray,
    y: np.ndarray,
    targets: tuple[tuple[float, float], ...],
) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y)
    horizontal = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
    vertical = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
    means: list[float] = []
    for target_x, target_y in targets:
        distances = np.where(finite, (x - target_x) ** 2 + (y - target_y) ** 2, np.inf)
        j, i = (int(index) for index in np.unravel_index(np.argmin(distances), distances.shape))
        local_lengths = []
        for jj, ii, values in (
            (j, i, horizontal),
            (j, i - 1, horizontal),
            (j, i, vertical),
            (j - 1, i, vertical),
        ):
            if (
                0 <= jj < values.shape[0]
                and 0 <= ii < values.shape[1]
                and np.isfinite(values[jj, ii])
            ):
                local_lengths.append(float(values[jj, ii]))
        means.append(float(np.mean(local_lengths)))
    return np.asarray(means, dtype=float)


def _finite_cell_nodes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y)
    return finite[:-1, :-1] & finite[1:, :-1] & finite[:-1, 1:] & finite[1:, 1:]


def _edge_key(edge: tuple[int, int]) -> tuple[int, int]:
    return edge if edge[0] < edge[1] else (edge[1], edge[0])


def _image_grid_for_state(
    state,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = shape
    image_polygon = np.column_stack((state.image_polygon.real, state.image_polygon.imag))
    xmin, xmax, ymin, ymax = state.image_extents
    xi = np.linspace(0.0, 1.0, nx)
    eta = np.linspace(0.0, 1.0, ny)
    XI, ETA = np.meshgrid(xi, eta)
    image_x = xmin + XI * (xmax - xmin)
    image_y = ymin + ETA * (ymax - ymin)
    valid_nodes = crdt._points_in_polygon(image_x, image_y, image_polygon)
    return image_x, image_y, valid_nodes


def test_schwarz_christoffel_forward_transform_returns_finite_value() -> None:
    transform = SchwarzChristoffelTransform(
        betas=np.array([-0.5, -0.5, -0.5, -0.5]),
        nnodes=8,
    )
    prevertices = np.exp(1j * np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False))

    value = transform.w2z(0.2 + 0.1j, prevertices)

    assert np.isfinite(value.real)
    assert np.isfinite(value.imag)


def test_schwarz_christoffel_inverse_round_trips_interior_point() -> None:
    transform = SchwarzChristoffelTransform(
        betas=np.array([-0.5, -0.5, -0.5, -0.5]),
        nnodes=8,
    )
    prevertices = np.exp(1j * np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False))
    w = 0.15 + 0.2j
    z = transform.w2z(w, prevertices)
    z2 = transform.w2z(0.25 + 0.0j, prevertices)

    round_tripped = transform.z2w(
        z,
        prevertices,
        z1=0.0 + 0.0j,
        w1=0.0 + 0.0j,
        z2=z2,
        w2=0.25 + 0.0j,
        tolerance=1.0e-7,
    )

    assert abs(round_tripped - w) < 1.0e-6


def test_crdt_quadrilaterals_match_gridgen_c_topology_contract() -> None:
    topology = prepare_crdt_topology(_tilted_channel_boundary())
    triangles = topology.triangulation.triangles
    internal_edges = {
        _edge_key((int(edge[0]), int(edge[1]))) for edge in topology.triangulation.internal_edges
    }

    assert len(topology.quadrilaterals) == len(topology.triangulation.points) - 3
    for quad in topology.quadrilaterals:
        v0, v1, v2, v3 = quad.vids
        tri0 = set(int(vertex) for vertex in triangles[quad.tids[0]])
        tri1 = set(int(vertex) for vertex in triangles[quad.tids[1]])
        assert _edge_key((v0, v2)) in internal_edges
        assert {v0, v1, v2} == tri0
        assert {v0, v2, v3} == tri1
        assert v1 not in tri1
        assert v3 not in tri0

        ordered_tri0 = [int(vertex) for vertex in triangles[quad.tids[0]]]
        start = ordered_tri0.index(v0)
        assert ordered_tri0[(start + 1) % 3] != v2


def test_gridgen_crdt_square_returns_finite_uniform_grid() -> None:
    boundary = BoundaryDefinition(
        x=np.array([0.0, 1.0, 1.0, 0.0]),
        y=np.array([0.0, 0.0, 1.0, 1.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0]),
    )

    result = GridGenerator(mode=SolverMode.GRIDGEN_CRDT).generate(boundary, shape=(9, 9))

    assert result.x.shape == (9, 9)
    assert result.y.shape == (9, 9)
    assert np.all(np.isfinite(result.x))
    assert np.all(np.isfinite(result.y))
    assert np.all(result.mask_rho)
    assert np.all(np.diff(result.x[4, :]) > 0.0)
    assert np.all(np.diff(result.y[:, 4]) > 0.0)


def test_gridgen_crdt_negative_beta_domain_returns_nan_exclusions() -> None:
    boundary = BoundaryDefinition(
        x=np.array([3.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0]),
        y=np.array([4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0]),
        beta=np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]),
    )

    result = GridGenerator(mode=SolverMode.GRIDGEN_CRDT).generate(boundary, shape=(17, 17))

    assert result.x.shape == (17, 17)
    assert result.y.shape == (17, 17)
    assert np.isnan(result.x[-1, 0])
    assert np.isnan(result.x[0, 0])
    assert np.isfinite(result.x[8, 0])
    assert np.isfinite(result.y[8, 0])
    assert np.all(np.isfinite(result.x[:, -1]))
    assert np.any(result.mask_rho)
    assert np.any(~result.mask_rho)


def test_gridgen_crdt_tilted_channel_has_rectangular_image_exclusions() -> None:
    beta = np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
    tilted = BoundaryDefinition(
        x=np.array([3.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0]),
        y=np.array([4.0, 4.0, 3.0, 4.0, 3.0, 2.0, 1.0, 1.0]),
        beta=beta,
    )

    tilted_result = GridGenerator(mode=SolverMode.GRIDGEN_CRDT).generate(tilted, shape=(17, 17))
    tilted_nodes = np.isfinite(tilted_result.x) & np.isfinite(tilted_result.y)

    assert np.isnan(tilted_result.x[0, 0])
    assert np.isnan(tilted_result.x[-1, 0])
    assert np.isfinite(tilted_result.x[8, 0])
    assert np.all(tilted_nodes[4:9, :])
    assert np.all(~tilted_nodes[:4, :12])
    assert np.all(tilted_nodes[:4, -1])
    assert np.any(tilted_result.mask_rho)
    assert np.any(~tilted_result.mask_rho)
    assert tilted_result.y[8, 0] > tilted_result.y[8, 4]


def test_gridgen_crdt_tilted_channel_is_monotonic_by_row_and_column() -> None:
    result = GridGenerator(mode=SolverMode.GRIDGEN_CRDT).generate(
        _tilted_channel_boundary(),
        shape=(17, 17),
    )

    failures = _nonmonotonic_runs(result.x, result.y)

    assert failures == [], f"nonmonotonic finite grid runs: {failures}"


def test_gridgen_crdt_tilted_channel_has_acceptable_orthogonality() -> None:
    result = GridGenerator(mode=SolverMode.GRIDGEN_CRDT).generate(
        _tilted_channel_boundary(),
        shape=(17, 17),
    )
    finite_cells = _finite_cell_nodes(result.x, result.y)
    valid_cells = result.mask_rho & finite_cells
    errors = orthogonality_error(result.x, result.y)[valid_cells]
    mean_error = float(np.mean(errors))
    p95_error = float(np.percentile(errors, 95.0))

    assert mean_error <= ORTHOGONALITY_MEAN_TOLERANCE, (
        f"mean orthogonality error {np.degrees(mean_error):.2f} deg exceeds "
        f"{np.degrees(ORTHOGONALITY_MEAN_TOLERANCE):.2f} deg"
    )
    assert p95_error <= ORTHOGONALITY_P95_TOLERANCE, (
        f"p95 orthogonality error {np.degrees(p95_error):.2f} deg exceeds "
        f"{np.degrees(ORTHOGONALITY_P95_TOLERANCE):.2f} deg"
    )


def test_gridgen_crdt_tilted_channel_has_similar_spacing_without_focus() -> None:
    result = GridGenerator(mode=SolverMode.GRIDGEN_CRDT).generate(
        _tilted_channel_boundary(),
        shape=(17, 17),
    )

    lengths = _valid_node_edge_lengths(result.x, result.y)
    coefficient_of_variation = np.std(lengths) / np.mean(lengths)
    percentile_ratio = np.percentile(lengths, 95.0) / np.percentile(lengths, 5.0)

    assert coefficient_of_variation <= EDGE_SPACING_CV_TOLERANCE, (
        f"edge-spacing coefficient of variation {coefficient_of_variation:.3f} "
        f"exceeds {EDGE_SPACING_CV_TOLERANCE:.3f}"
    )
    assert percentile_ratio <= EDGE_SPACING_P95_P5_TOLERANCE, (
        f"edge-spacing p95/p5 ratio {percentile_ratio:.3f} exceeds "
        f"{EDGE_SPACING_P95_P5_TOLERANCE:.3f}"
    )


def test_crdt_state_has_cross_ratios_and_orthogonal_image_sides() -> None:
    boundary = BoundaryDefinition(
        x=np.array([3.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0]),
        y=np.array([4.0, 4.0, 3.0, 4.0, 3.0, 2.0, 1.0, 1.0]),
        beta=np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]),
    )

    state = solve_crdt_state(boundary)
    image_points = np.column_stack((state.image_vertices.real, state.image_vertices.imag))
    deltas = np.roll(image_points, -1, axis=0) - image_points

    assert len(state.cis) == len(state.topology.quadrilaterals)
    assert np.all(np.isfinite(state.cis))
    assert state.residual_norm <= 1.0e-10
    assert state.ws is not None
    assert state.physical_a is not None
    assert state.physical_b is not None
    assert state.image_a is not None
    assert state.image_b is not None
    assert state.ws.shape[0] == len(state.topology.quadrilaterals)
    assert state.diagnostics is not None
    assert state.diagnostics.pre_alignment_vertex_mismatch <= 1.0e-8
    assert np.all((np.isclose(deltas[:, 0], 0.0)) | (np.isclose(deltas[:, 1], 0.0)))


def test_crdt_state_default_corner_angle_diagnostics_explain_tilted_channel_spacing() -> None:
    state = solve_crdt_state(_tilted_channel_boundary())
    assert state.diagnostics is not None

    expected_physical_degrees = np.array([90.0, 90.0, 315.0, 45.0, 135.0, 225.0, 90.0, 90.0])
    expected_image_degrees = np.array([90.0, 90.0, 270.0, 90.0, 90.0, 270.0, 90.0, 90.0])
    expected_ratios = expected_physical_degrees / expected_image_degrees

    assert np.allclose(
        np.degrees(state.diagnostics.physical_corner_angles), expected_physical_degrees
    )
    assert np.allclose(np.degrees(state.diagnostics.image_corner_angles), expected_image_degrees)
    assert np.allclose(state.diagnostics.corner_angle_ratios, expected_ratios)
    assert np.array_equal(
        state.diagnostics.corner_angle_warning,
        np.array([False, False, False, True, True, False, False, False]),
    )
    assert state.diagnostics.sigma_residual_norm <= 1.0e-10
    assert state.diagnostics.image_vertex_mismatch <= 1.0e-8


def test_crdt_options_image_beta_separates_image_angles_from_topology_beta() -> None:
    image_beta = _angle_matched_tilted_channel_image_beta()

    state = solve_crdt_state(
        _tilted_channel_boundary(),
        options=CRDTOptions(image_beta=image_beta),
    )
    assert state.diagnostics is not None

    assert np.array_equal(state.user_betas, _tilted_channel_boundary().beta[:-1])
    assert np.array_equal(state.image_marker_betas, image_beta)
    assert np.allclose(state.diagnostics.corner_angle_ratios, 1.0)
    assert not np.any(state.diagnostics.corner_angle_warning)
    assert state.diagnostics.image_beta_source == "explicit"


def test_crdt_options_image_beta_preprocesses_closed_reversed_boundaries() -> None:
    boundary = BoundaryDefinition(
        x=np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
        y=np.array([0.0, 1.0, 1.0, 0.0, 0.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
    )
    image_beta = np.array([0.5, 1.5, 1.0, 1.0, 0.0])

    state = solve_crdt_state(boundary, options=CRDTOptions(image_beta=image_beta))

    assert np.array_equal(state.image_marker_betas, np.array([1.0, 1.0, 1.5, 0.5]))


@pytest.mark.parametrize(
    ("image_beta", "message"),
    [
        (np.array([1.0, 1.0, 1.0]), "one entry per vertex"),
        (np.array([1.0, 1.0, 1.0, 1.0, 1.0]), "closing image_beta entry must be 0"),
        (np.array([2.0, 0.0, 2.0, 0.0]), "at least three"),
        (np.array([1.0, 1.0, 1.0, 0.5]), "sum\\(image_beta\\) must equal 4"),
    ],
)
def test_crdt_options_image_beta_validation_errors(image_beta: np.ndarray, message: str) -> None:
    boundary = BoundaryDefinition(
        x=np.array([0.0, 1.0, 1.0, 0.0]),
        y=np.array([0.0, 0.0, 1.0, 1.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0]),
    )

    with pytest.raises(ValueError, match=message):
        solve_crdt_state(boundary, options=CRDTOptions(image_beta=image_beta))


def test_crdt_options_angle_matched_image_beta_reduces_reported_spacing_bias() -> None:
    default_state = solve_crdt_state(_tilted_channel_boundary())
    matched_state = solve_crdt_state(
        _tilted_channel_boundary(),
        options=CRDTOptions(image_beta=_angle_matched_tilted_channel_image_beta()),
    )
    assert default_state.diagnostics is not None
    assert matched_state.diagnostics is not None

    default_bias = np.max(np.abs(default_state.diagnostics.corner_angle_ratios - 1.0))
    matched_bias = np.max(np.abs(matched_state.diagnostics.corner_angle_ratios - 1.0))

    assert matched_bias < default_bias
    assert matched_bias <= 1.0e-12


def test_crdt_options_angle_matched_image_beta_reduces_local_spacing_spread() -> None:
    targets = ((1.0, 4.0), (2.0, 2.0), (1.0, 3.0), (2.0, 3.0))
    default_result = GridGenerator(mode=SolverMode.GRIDGEN_CRDT).generate(
        _tilted_channel_boundary(),
        shape=(41, 41),
    )
    matched_result = GridGenerator(
        mode=SolverMode.GRIDGEN_CRDT,
        crdt_options=CRDTOptions(image_beta=_angle_matched_tilted_channel_image_beta()),
    ).generate(
        _tilted_channel_boundary(),
        shape=(41, 41),
    )

    default_lengths = _local_mean_edge_lengths(default_result.x, default_result.y, targets)
    matched_lengths = _local_mean_edge_lengths(matched_result.x, matched_result.y, targets)
    default_spread = float(np.max(default_lengths) / np.min(default_lengths))
    matched_spread = float(np.max(matched_lengths) / np.min(matched_lengths))

    assert matched_spread < default_spread
    assert matched_spread <= 1.5


def test_gridgen_crdt_angle_matched_image_beta_maps_high_resolution_nodes() -> None:
    result_51 = GridGenerator(
        mode=SolverMode.GRIDGEN_CRDT,
        crdt_options=CRDTOptions(image_beta=_angle_matched_tilted_channel_image_beta()),
    ).generate(
        _tilted_channel_boundary(),
        shape=(51, 51),
    )
    result_101 = GridGenerator(
        mode=SolverMode.GRIDGEN_CRDT,
        crdt_options=CRDTOptions(image_beta=_angle_matched_tilted_channel_image_beta()),
    ).generate(
        _tilted_channel_boundary(),
        shape=(101, 101),
    )

    assert np.any(np.isfinite(result_51.x))
    assert np.any(np.isfinite(result_101.x))


def test_crdt_inverse_least_squares_maps_known_acute_corner_node() -> None:
    state = solve_crdt_state(
        _tilted_channel_boundary(),
        options=CRDTOptions(image_beta=_angle_matched_tilted_channel_image_beta()),
    )
    stats = crdt._MappingStats()
    point = 0.04 + 2.94j

    mapped = crdt._map_image_point_with_details(point, state, stats)

    assert mapped is not None
    assert mapped.method == "least_squares"
    assert abs(mapped.w) <= 1.0
    assert np.isfinite(mapped.z.real)
    assert np.isfinite(mapped.z.imag)
    assert mapped.inverse_residual <= 1.0e-6
    assert stats.least_squares_solves == 1


def test_crdt_options_image_beta_blend_resolves_expected_sources_and_values() -> None:
    boundary = _tilted_channel_boundary()
    default_state = solve_crdt_state(boundary)
    zero_blend_state = solve_crdt_state(boundary, options=CRDTOptions(image_beta_blend=0.0))
    full_blend_state = solve_crdt_state(boundary, options=CRDTOptions(image_beta_blend=1.0))
    explicit_state = solve_crdt_state(
        boundary,
        options=CRDTOptions(image_beta=_angle_matched_tilted_channel_image_beta()),
    )

    assert default_state.diagnostics is not None
    assert zero_blend_state.diagnostics is not None
    assert full_blend_state.diagnostics is not None
    assert explicit_state.diagnostics is not None
    assert default_state.diagnostics.image_beta_source == "topology"
    assert zero_blend_state.diagnostics.image_beta_source == "blend"
    assert full_blend_state.diagnostics.image_beta_source == "blend"
    assert np.allclose(zero_blend_state.image_marker_betas, default_state.image_marker_betas)
    assert np.allclose(full_blend_state.image_marker_betas, explicit_state.image_marker_betas)
    assert np.allclose(full_blend_state.image_vertices, explicit_state.image_vertices)


def test_crdt_options_image_beta_blend_reduces_spacing_with_intermediate_tradeoff() -> None:
    targets = ((1.0, 4.0), (2.0, 2.0), (1.0, 3.0), (2.0, 3.0))
    default_result = GridGenerator(mode=SolverMode.GRIDGEN_CRDT).generate(
        _tilted_channel_boundary(),
        shape=(41, 41),
    )
    blend_result = GridGenerator(
        mode=SolverMode.GRIDGEN_CRDT,
        crdt_options=CRDTOptions(image_beta_blend=0.75),
    ).generate(
        _tilted_channel_boundary(),
        shape=(41, 41),
    )
    matched_result = GridGenerator(
        mode=SolverMode.GRIDGEN_CRDT,
        crdt_options=CRDTOptions(image_beta_blend=1.0),
    ).generate(
        _tilted_channel_boundary(),
        shape=(41, 41),
    )

    default_lengths = _local_mean_edge_lengths(default_result.x, default_result.y, targets)
    blend_lengths = _local_mean_edge_lengths(blend_result.x, blend_result.y, targets)
    matched_lengths = _local_mean_edge_lengths(matched_result.x, matched_result.y, targets)
    default_spread = float(np.max(default_lengths) / np.min(default_lengths))
    blend_spread = float(np.max(blend_lengths) / np.min(blend_lengths))
    matched_spread = float(np.max(matched_lengths) / np.min(matched_lengths))

    assert matched_spread < blend_spread < default_spread


@pytest.mark.parametrize("blend", [-0.1, 1.1])
def test_crdt_options_image_beta_blend_validation_errors(blend: float) -> None:
    with pytest.raises(ValueError, match="image_beta_blend must lie within \\[0, 1\\]"):
        solve_crdt_state(_tilted_channel_boundary(), options=CRDTOptions(image_beta_blend=blend))


def test_crdt_state_maps_interior_points_with_sc_embedding() -> None:
    boundary = BoundaryDefinition(
        x=np.array([3.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0]),
        y=np.array([4.0, 4.0, 3.0, 4.0, 3.0, 2.0, 1.0, 1.0]),
        beta=np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]),
    )

    state = solve_crdt_state(boundary)
    mapped = None
    stats = crdt._MappingStats()
    assert state.image_quadrilateral_polygons is not None
    for polygon in state.image_quadrilateral_polygons:
        candidate = complex(np.mean(polygon.real), np.mean(polygon.imag))
        mapped = crdt._map_image_point_with_sc(candidate, state, stats)
        if mapped is not None:
            break

    assert len(state.image_quadrilateral_polygons) == len(state.topology.quadrilaterals)
    assert mapped is not None
    assert np.isfinite(mapped.real)
    assert np.isfinite(mapped.imag)
    assert 1.0 <= mapped.real <= 3.0
    assert 1.0 <= mapped.imag <= 4.0


def test_gridgen_crdt_tilted_channel_uses_only_sc_mapping() -> None:
    state = solve_crdt_state(_tilted_channel_boundary())
    image_x, image_y, valid_nodes = _image_grid_for_state(state, (17, 17))
    stats = crdt._MappingStats()

    crdt._map_image_grid_to_physical(image_x, image_y, state, valid_nodes, stats)

    assert stats.sc_map_failures == 0


def test_gridgen_crdt_tilted_channel_tracks_inverse_integrity() -> None:
    state = solve_crdt_state(_tilted_channel_boundary())
    image_x, image_y, valid_nodes = _image_grid_for_state(state, (17, 17))
    stats = crdt._MappingStats()

    x, y = crdt._map_image_grid_to_physical(image_x, image_y, state, valid_nodes, stats)

    assert np.isfinite(x[12, 15])
    assert np.isfinite(y[12, 15])
    assert np.isfinite(x[13, 15])
    assert np.isfinite(y[13, 15])
    assert stats.continuation_uses > 0
    assert stats.inverse_evaluations > 0
    assert stats.max_inverse_residual <= 1.0e-6
    assert stats.worst_node_index is not None


def test_gridgen_crdt_candidate_short_circuit_tracks_attempts() -> None:
    state = solve_crdt_state(_tilted_channel_boundary())
    image_x, image_y, valid_nodes = _image_grid_for_state(state, (17, 17))
    stats = crdt._MappingStats()

    crdt._map_image_grid_to_physical(image_x, image_y, state, valid_nodes, stats)

    assert stats.sc_map_failures == 0
    assert stats.candidate_nodes == int(np.count_nonzero(valid_nodes))
    assert stats.candidate_attempts >= stats.candidate_nodes
    assert stats.candidate_attempts <= int(np.ceil(stats.candidate_nodes * 1.5))
    assert stats.root_solves <= 3
    assert stats.least_squares_solves == 0


# ---------------------------------------------------------------------------
# _align_image_vertices: end-to-end property that image polygon corners land
# on the integer axis lattice when image_beta is half-integer-valued.
# ---------------------------------------------------------------------------


def test_align_image_vertices_produces_integer_lattice_corners_for_side_channel() -> None:
    """Angle-matched image_beta yields integer-lattice corners on the image polygon.

    Every entry in the angle-matched image_beta sequence is a multiple of
    0.5, so the axis-aligned walk executes at every segment. The resulting
    image polygon corners should lie on the integer axis lattice. This is
    the user-observable correctness property; the internal walk algorithm
    is also covered by synthetic corner-preservation tests below.
    """
    state = solve_crdt_state(
        _tilted_channel_boundary(),
        options=CRDTOptions(image_beta=_angle_matched_tilted_channel_image_beta()),
    )
    corners = state.image_vertices[state.corner_vertex_ids]
    for z in corners:
        assert np.isclose(z.real, round(z.real), atol=1.0e-6)
        assert np.isclose(z.imag, round(z.imag), atol=1.0e-6)


def test_align_image_vertices_produces_integer_lattice_corners_for_default_square() -> None:
    """Default (topology-driven) image_beta on a simple square gives a unit-square image."""
    boundary = BoundaryDefinition(
        x=np.array([0.0, 1.0, 1.0, 0.0]),
        y=np.array([0.0, 0.0, 1.0, 1.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    state = solve_crdt_state(boundary)
    corners = state.image_vertices[state.corner_vertex_ids]
    # Four corners should land on an axis-aligned rectangle, each coordinate
    # snapping to 0 or 1 in the normalized image plane.
    for z in corners:
        assert np.isclose(abs(z.real * (1 - z.real)), 0.0, atol=1.0e-6)
        assert np.isclose(abs(z.imag * (1 - z.imag)), 0.0, atol=1.0e-6)


def test_sigma_solver_success_flag_does_not_override_bad_residual(monkeypatch):
    from types import SimpleNamespace

    def false_success(fun, initial, **kwargs):
        return SimpleNamespace(success=True, x=np.full_like(initial, 2.0), message="success")

    monkeypatch.setattr(crdt, "root", false_success)
    boundary = BoundaryDefinition([0, 50, 50, 0], [0, 0, 1, 1], [1] * 4)
    with pytest.raises(crdt.CRDTError, match="sigma solve failed.*residual="):
        solve_crdt_state(boundary)


def test_crdt_mapping_diagnostics_survive_public_generation():
    boundary = BoundaryDefinition([0, 1, 1, 0], [0, 0, 1, 1], [1] * 4)
    grid = GridGenerator(mode=SolverMode.GRIDGEN_CRDT).generate(boundary, (7, 9))
    assert grid.diagnostics["converged"]
    mapping = grid.diagnostics["mapping"]
    assert mapping["sc_map_failures"] == 0
    assert mapping["candidate_nodes"] == grid.x.size
    assert grid.options["crdt_options"]["precision"] == 1e-10


def test_alignment_walk_preserves_downstream_corners():
    # Slightly off-axis corners must not be overwritten by the interior walk.
    z = np.array(
        [0j, 0.1 + 0.5j, 0.02 + 1j, 0.5 + 1.1j, 1 + 1.03j, 1.1 + 0.5j, 1.04 + 0j, 0.5 - 0.1j]
    )
    corners = np.array([0, 2, 4, 6])
    betas = np.array([-0.5, 0, -0.5, 0, -0.5, 0, -0.5, 0])
    aligned = crdt._align_image_vertices(z, betas, corners, 1e-10)
    assert np.array_equal(aligned[corners], z[corners])
    assert aligned[1].real == 0
    assert aligned[3].imag == z[2].imag


def test_fractional_alignment_walk_preserves_noncorner_vertices():
    z = np.array([0j, 0.1 + 0.5j, 1j, 0.5 + 1.1j, 1 + 1j, 1.1 + 0.5j, 1 + 0j, 0.5 - 0.1j])
    corners = np.array([0, 2, 4, 6])
    betas = np.array([-0.7, 0, -0.3, 0, -0.5, 0, -0.5, 0])
    aligned = crdt._align_image_vertices(z, betas, corners, 1e-10)
    assert np.array_equal(aligned[[3, 5, 7]], z[[3, 5, 7]])
    assert np.array_equal(aligned[corners], z[corners])


@pytest.mark.parametrize("second_triangle", [True, False])
def test_crowded_prevertices_remain_distinct(second_triangle):
    from types import SimpleNamespace

    quad = SimpleNamespace(vids=(0, 1, 2, 3), neighbours=())
    topology = SimpleNamespace(quadrilaterals=(quad,))
    pa, pb, pc = -1 + 0j, 1 + 0j, np.exp(1e-6j)
    ws = np.array([pa, pb, pc, 0j]) if second_triangle else np.array([pc, 0j, pa, pb])
    crdt._process_quadrilateral(
        topology, np.array([0.0]), ws, np.array([False]), 0, second_triangle
    )
    pd = ws[3 if second_triangle else 1]
    cross_ratio = (pd - pa) * (pb - pc) / ((pc - pd) * (pa - pb))
    assert abs(pd - pc) > 1e-8
    assert abs(abs(pd) - 1) < 1e-12
    assert abs(cross_ratio + 1) < 1e-8
