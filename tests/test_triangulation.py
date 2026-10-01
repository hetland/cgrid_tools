import numpy as np
import pytest

from cgrid_tools.gridgen import (
    BoundaryDefinition,
    TriangulationError,
    build_constrained_triangulation,
    extract_quadrilaterals,
    prepare_crdt_topology,
)


def _boundary(x, y, beta=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if beta is None:
        beta = np.zeros(len(x), dtype=float)
        beta[[0, len(x) // 4, len(x) // 2, 3 * len(x) // 4]] = 1.0
    return BoundaryDefinition(x=x, y=y, beta=np.asarray(beta, dtype=float))


def _edge_set(edges):
    return {tuple(sorted(edge)) for edge in np.asarray(edges, dtype=int)}


def _triangle_incidence(triangles, point_count):
    incidence = [[] for _ in range(point_count)]
    for tid, triangle in enumerate(triangles):
        for vertex in triangle:
            incidence[int(vertex)].append(tid)
    return tuple(tuple(item) for item in incidence)


def test_constrained_triangulation_preserves_square_boundary_edges() -> None:
    boundary = _boundary(
        x=[0.0, 1.0, 1.0, 0.0],
        y=[0.0, 0.0, 1.0, 1.0],
        beta=[1.0, 1.0, 1.0, 1.0],
    )

    triangulation = build_constrained_triangulation(boundary)

    assert triangulation.points.shape == (4, 2)
    assert triangulation.triangles.shape == (2, 3)
    assert _edge_set(triangulation.constrained_edges).issubset(_edge_set(triangulation.edges))
    assert len(triangulation.internal_edges) == len(triangulation.points) - 3


def test_constrained_triangulation_handles_concave_l_shape() -> None:
    boundary = _boundary(
        x=[0.0, 2.0, 2.0, 1.0, 1.0, 0.0],
        y=[0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        beta=[1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    )

    triangulation = build_constrained_triangulation(boundary)

    assert _edge_set(triangulation.constrained_edges).issubset(_edge_set(triangulation.edges))
    assert len(triangulation.triangles) == len(triangulation.points) - 2
    assert len(triangulation.internal_edges) == len(triangulation.points) - 3


def test_constrained_triangulation_handles_negative_beta_side_channel() -> None:
    boundary = BoundaryDefinition(
        x=np.array([0.5, 2.0, 2.0, 3.5, 3.5, 2.0, 2.0, 0.5, 0.5]),
        y=np.array([0.5, 0.5, 1.75, 1.75, 2.25, 2.25, 3.5, 3.5, 0.5]),
        beta=np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 0.0]),
    )

    triangulation = build_constrained_triangulation(boundary)

    assert _edge_set(triangulation.constrained_edges).issubset(_edge_set(triangulation.edges))
    assert len(triangulation.triangles) == len(triangulation.points) - 2
    assert len(triangulation.internal_edges) == len(triangulation.points) - 3


def test_constrained_triangulation_removes_near_duplicate_vertices() -> None:
    boundary = _boundary(
        x=[0.0, 1.0e-14, 1.0, 1.0, 0.0],
        y=[0.0, 0.0, 0.0, 1.0, 1.0],
        beta=[0.5, 0.5, 1.0, 1.0, 1.0],
    )

    triangulation = build_constrained_triangulation(boundary)

    assert triangulation.points.shape == (4, 2)
    assert np.isclose(triangulation.beta.sum(), 4.0)
    assert _edge_set(triangulation.constrained_edges).issubset(_edge_set(triangulation.edges))


def test_constrained_triangulation_rejects_self_intersection() -> None:
    boundary = BoundaryDefinition(
        x=np.array([0.0, 1.0, 0.0, 1.0]),
        y=np.array([0.0, 1.0, 1.0, 0.0]),
        beta=np.array([1.0, 1.0, 1.0, 1.0]),
    )

    with pytest.raises(TriangulationError, match="self-intersecting"):
        build_constrained_triangulation(boundary)


def test_point_triangle_incidence_matches_triangles() -> None:
    boundary = _boundary(
        x=[0.0, 2.0, 2.0, 1.0, 1.0, 0.0],
        y=[0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        beta=[1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    )

    triangulation = build_constrained_triangulation(boundary)

    assert triangulation.point_triangles == _triangle_incidence(
        triangulation.triangles,
        len(triangulation.points),
    )


def test_quadrilateral_extraction_matches_gridgen_counts() -> None:
    boundary = _boundary(
        x=[0.0, 2.0, 2.0, 1.0, 1.0, 0.0],
        y=[0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        beta=[1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    )

    triangulation = build_constrained_triangulation(boundary)
    quadrilaterals = extract_quadrilaterals(triangulation)

    assert len(quadrilaterals) == len(triangulation.points) - 3
    assert all(len(quad.vids) == 4 for quad in quadrilaterals)
    assert all(len(quad.tids) == 2 for quad in quadrilaterals)
    assert any(quad.neighbours for quad in quadrilaterals)


def test_prepare_crdt_topology_returns_triangulation_and_quadrilaterals() -> None:
    boundary = _boundary(
        x=[0.0, 1.0, 1.0, 0.0],
        y=[0.0, 0.0, 1.0, 1.0],
        beta=[1.0, 1.0, 1.0, 1.0],
    )

    topology = prepare_crdt_topology(boundary)

    assert len(topology.triangulation.triangles) == 2
    assert len(topology.quadrilaterals) == 1


def test_repeated_triangulation_does_not_mutate_boundary():
    boundary = _boundary([0, 1e-14, 1, 1, 0], [0, 0, 0, 1, 1], [0.5, 0.5, 1, 1, 1])
    before = boundary.beta.copy()
    first = build_constrained_triangulation(boundary)
    second = build_constrained_triangulation(boundary)
    assert np.array_equal(boundary.beta, before)
    assert np.array_equal(first.beta, second.beta)
    assert np.isclose(second.beta.sum(), 4)
    assert np.array_equal(first.triangles, second.triangles)
