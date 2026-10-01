import os
import time

import numpy as np
import pytest

import cgrid_tools.gridgen.crdt as crdt
import cgrid_tools.gridgen.generator as generator
from cgrid_tools.gridgen import (
    BoundaryDefinition,
    CRDTOptions,
    GridGenerator,
    SolverMode,
    solve_crdt_state,
)

pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(
        os.environ.get("RUN_PERFORMANCE") != "1",
        reason="set RUN_PERFORMANCE=1 to run opt-in performance checks",
    ),
]


def _rectangle_boundary() -> BoundaryDefinition:
    return BoundaryDefinition(
        x=np.array([0.0, 3.0, 3.0, 0.0]),
        y=np.array([0.0, 0.0, 2.0, 2.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0]),
    )


def _tilted_channel_boundary() -> BoundaryDefinition:
    return BoundaryDefinition(
        x=np.array([3.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0]),
        y=np.array([4.0, 4.0, 3.0, 4.0, 3.0, 2.0, 1.0, 1.0]),
        beta=np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]),
    )


def _elliptic_inputs(shape: tuple[int, int]):
    boundary = _rectangle_boundary()
    computational_x, computational_y = generator._computational_coordinates(shape, None)
    edges = generator._extract_edges(boundary, computational_x, computational_y)
    x, y = generator._transfinite_interpolation(
        *edges,
        computational_x=computational_x,
        computational_y=computational_y,
    )
    cell_mask = generator._cell_mask_from_boundary(x, y, boundary)
    update_mask = generator._node_update_mask(cell_mask)
    return x, y, computational_x, computational_y, update_mask


def _image_grid_for_state(
    state, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = shape
    xmin, xmax, ymin, ymax = state.image_extents
    xi = np.linspace(0.0, 1.0, nx)
    eta = np.linspace(0.0, 1.0, ny)
    XI, ETA = np.meshgrid(xi, eta)
    image_x = xmin + XI * (xmax - xmin)
    image_y = ymin + ETA * (ymax - ymin)
    image_polygon = np.column_stack((state.image_polygon.real, state.image_polygon.imag))
    valid_nodes = crdt._points_in_polygon(image_x, image_y, image_polygon)
    return image_x, image_y, valid_nodes


def test_elliptic_numba_kernel_is_faster_than_python_reference() -> None:
    inputs = _elliptic_inputs((101, 151))

    generator._elliptic_smooth(
        *inputs,
        max_iterations=1,
        relaxation=1.4,
        tolerance=0.0,
    )

    start = time.perf_counter()
    accelerated_x, accelerated_y, accelerated_iterations = generator._elliptic_smooth(
        *inputs,
        max_iterations=30,
        relaxation=1.4,
        tolerance=0.0,
    )
    accelerated_seconds = time.perf_counter() - start

    start = time.perf_counter()
    reference_x, reference_y, reference_iterations = generator._elliptic_smooth_python(
        *inputs,
        max_iterations=30,
        relaxation=1.4,
        tolerance=0.0,
    )
    reference_seconds = time.perf_counter() - start

    assert accelerated_iterations == reference_iterations
    assert np.allclose(accelerated_x, reference_x)
    assert np.allclose(accelerated_y, reference_y)
    assert accelerated_seconds < 0.5 * reference_seconds


def test_crdt_sloped_channel_mapping_performance_smoke() -> None:
    state = solve_crdt_state(_tilted_channel_boundary(), CRDTOptions(image_beta_blend=0.5))
    image_x, image_y, valid_nodes = _image_grid_for_state(state, (101, 151))
    stats = crdt._MappingStats()

    start = time.perf_counter()
    x, y = crdt._map_image_grid_to_physical(image_x, image_y, state, valid_nodes, stats)
    elapsed_seconds = time.perf_counter() - start

    assert np.any(np.isfinite(x))
    assert np.any(np.isfinite(y))
    assert stats.sc_map_failures == 0
    assert stats.candidate_nodes == int(np.count_nonzero(valid_nodes))
    assert stats.candidate_attempts <= int(np.ceil(stats.candidate_nodes * 1.75))
    assert elapsed_seconds < 30.0


def test_large_solver_entry_points_complete() -> None:
    elliptic = GridGenerator(
        mode=SolverMode.ELLIPTIC,
        smoothing_iterations=50,
    ).generate(_rectangle_boundary(), shape=(201, 301))
    crdt_grid = GridGenerator(
        mode=SolverMode.GRIDGEN_CRDT,
        crdt_options=CRDTOptions(image_beta_blend=0.5),
    ).generate(_tilted_channel_boundary(), shape=(101, 151))

    assert elliptic.x.shape == (201, 301)
    assert crdt_grid.x.shape == (101, 151)
    assert np.all(np.isfinite(elliptic.x))
    assert np.any(np.isfinite(crdt_grid.x))
