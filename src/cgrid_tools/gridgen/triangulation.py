from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, QhullError

from .boundary import BoundaryDefinition
from .geometry import edge_key as _edge_key
from .geometry import signed_area as _signed_area


class TriangulationError(ValueError):
    """Raised when a polygon cannot be turned into a valid constrained mesh."""


@dataclass(frozen=True)
class ConstrainedTriangulation:
    points: np.ndarray
    triangles: np.ndarray
    point_triangles: tuple[tuple[int, ...], ...]
    edges: np.ndarray
    constrained_edges: np.ndarray
    beta: np.ndarray

    @property
    def internal_edges(self) -> np.ndarray:
        constrained = {_edge_key(edge) for edge in self.constrained_edges}
        return np.asarray(
            [edge for edge in self.edges if _edge_key(edge) not in constrained],
            dtype=int,
        )


@dataclass(frozen=True)
class Quadrilateral:
    id: int
    vids: tuple[int, int, int, int]
    tids: tuple[int, int]
    neighbours: tuple[int, ...]


@dataclass(frozen=True)
class CRDTTopology:
    triangulation: ConstrainedTriangulation
    quadrilaterals: tuple[Quadrilateral, ...]


def prepare_crdt_topology(boundary: BoundaryDefinition) -> CRDTTopology:
    triangulation = build_constrained_triangulation(boundary)
    quadrilaterals = extract_quadrilaterals(triangulation)
    return CRDTTopology(triangulation=triangulation, quadrilaterals=quadrilaterals)


def build_constrained_triangulation(boundary: BoundaryDefinition) -> ConstrainedTriangulation:
    points, beta = _preprocess_boundary(boundary.vertices, boundary.beta[:-1])
    if len(points) < 3:
        raise TriangulationError("boundary must contain at least three distinct vertices")
    if _has_self_intersections(points):
        raise TriangulationError("boundary polygon is self-intersecting")

    constrained_edges = _boundary_edges(len(points))
    triangles = _candidate_delaunay_triangles(points)
    triangles = _recover_constraints_with_flips(points, triangles, constrained_edges)
    if not _contains_edges(triangles, constrained_edges):
        triangles = _ear_clip_triangulation(points)

    triangles = _filter_inside_triangles(points, triangles)
    if not _contains_edges(triangles, constrained_edges):
        triangles = _ear_clip_triangulation(points)
    triangles = _improve_constrained_delaunay(points, triangles, constrained_edges)
    triangles = _orient_triangles_ccw(points, triangles)
    triangulation = _make_triangulation(points, beta, triangles, constrained_edges)
    _validate_triangulation(triangulation)
    return triangulation


def extract_quadrilaterals(triangulation: ConstrainedTriangulation) -> tuple[Quadrilateral, ...]:
    constrained = {_edge_key(edge) for edge in triangulation.constrained_edges}
    diagonal_stack: list[int] = []
    for edge in triangulation.edges:
        key = _edge_key(edge)
        if key in constrained:
            continue
        diagonal_stack.extend((int(edge[0]), int(edge[1])))

    quadrilaterals: list[Quadrilateral] = []

    while diagonal_stack:
        vid1 = diagonal_stack.pop()
        vid2 = diagonal_stack.pop()
        tids = _common_triangles_in_point_order(
            triangulation.point_triangles,
            vid1,
            vid2,
        )
        if len(tids) != 2:
            continue
        vids, tids = _quadrilateral_vertices(triangulation.triangles, tids, vid1, vid2)
        quadrilaterals.append(
            Quadrilateral(
                id=len(quadrilaterals),
                vids=vids,
                tids=tids,
                neighbours=(),
            )
        )

    neighbours = _quadrilateral_neighbours(quadrilaterals)
    return tuple(
        Quadrilateral(
            id=quad.id,
            vids=quad.vids,
            tids=quad.tids,
            neighbours=neighbours[quad.id],
        )
        for quad in quadrilaterals
    )


def _preprocess_boundary(points: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    beta = np.array(beta, dtype=float, copy=True)
    if points.ndim != 2 or points.shape[1] != 2:
        raise TriangulationError("boundary vertices must have shape (n, 2)")
    if len(points) != len(beta):
        raise TriangulationError("boundary vertices and beta must have matching lengths")

    span = np.ptp(points, axis=0)
    tolerance = max(float(np.max(span)) / 1.0e6, 1.0e-12)
    keep = [0]
    for idx in range(1, len(points)):
        if np.linalg.norm(points[idx] - points[keep[-1]]) > tolerance:
            keep.append(idx)
        else:
            beta[keep[-1]] += beta[idx]
    if len(keep) > 1 and np.linalg.norm(points[keep[0]] - points[keep[-1]]) <= tolerance:
        beta[keep[0]] += beta[keep[-1]]
        keep.pop()

    points = points[keep]
    beta = beta[keep]
    if _signed_area(points) < 0.0:
        points = points[::-1]
        beta = beta[::-1]
    return points, beta


def _candidate_delaunay_triangles(points: np.ndarray) -> np.ndarray:
    try:
        triangles = Delaunay(points).simplices
    except QhullError as error:
        raise TriangulationError("boundary vertices could not be triangulated") from error
    return np.asarray(triangles, dtype=int)


def _recover_constraints_with_flips(
    points: np.ndarray,
    triangles: np.ndarray,
    constrained_edges: np.ndarray,
) -> np.ndarray:
    triangles = _orient_triangles_ccw(points, triangles)
    max_iterations = max(10, len(points) * len(points) * 4)

    for constrained_edge in constrained_edges:
        target = _edge_key(constrained_edge)
        for _ in range(max_iterations):
            if _contains_edge(triangles, target):
                break
            crossing = _find_crossing_edge(points, triangles, target)
            if crossing is None:
                break
            flipped = _flip_edge(points, triangles, crossing)
            if flipped is None:
                break
            triangles = flipped
        if not _contains_edge(triangles, target):
            return triangles
    return triangles


def _find_crossing_edge(
    points: np.ndarray,
    triangles: np.ndarray,
    target: tuple[int, int],
) -> tuple[int, int] | None:
    target_segment = (points[target[0]], points[target[1]])
    for edge in _edges_from_triangles(triangles):
        key = _edge_key(edge)
        if target[0] in key or target[1] in key:
            continue
        if _segments_cross_strict(
            target_segment[0], target_segment[1], points[key[0]], points[key[1]]
        ):
            return key
    return None


def _flip_edge(
    points: np.ndarray,
    triangles: np.ndarray,
    edge: tuple[int, int],
) -> np.ndarray | None:
    incident = [
        idx for idx, triangle in enumerate(triangles) if edge[0] in triangle and edge[1] in triangle
    ]
    if len(incident) != 2:
        return None

    tri_a = triangles[incident[0]]
    tri_b = triangles[incident[1]]
    opposite_a = [int(v) for v in tri_a if v not in edge]
    opposite_b = [int(v) for v in tri_b if v not in edge]
    if len(opposite_a) != 1 or len(opposite_b) != 1:
        return None

    quad_points = points[[opposite_a[0], edge[0], opposite_b[0], edge[1]]]
    if not _is_convex_quad(quad_points):
        return None

    flipped = triangles.copy()
    flipped[incident[0]] = [opposite_a[0], edge[0], opposite_b[0]]
    flipped[incident[1]] = [opposite_a[0], opposite_b[0], edge[1]]
    return _orient_triangles_ccw(points, flipped)


def _ear_clip_triangulation(points: np.ndarray) -> np.ndarray:
    remaining = list(range(len(points)))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    max_iterations = max(1, len(points) * len(points))

    while len(remaining) > 3 and guard < max_iterations:
        guard += 1
        clipped = False
        for local_idx, vertex in enumerate(remaining):
            prev_vertex = remaining[local_idx - 1]
            next_vertex = remaining[(local_idx + 1) % len(remaining)]
            if not _is_ear(points, remaining, prev_vertex, vertex, next_vertex):
                continue
            triangles.append((prev_vertex, vertex, next_vertex))
            del remaining[local_idx]
            clipped = True
            break
        if not clipped:
            raise TriangulationError(
                "could not recover polygon constraints; polygon may be degenerate"
            )

    if len(remaining) == 3:
        triangles.append(tuple(remaining))
    return np.asarray(triangles, dtype=int)


def _improve_constrained_delaunay(
    points: np.ndarray,
    triangles: np.ndarray,
    constrained_edges: np.ndarray,
) -> np.ndarray:
    constrained = {_edge_key(edge) for edge in constrained_edges}
    triangles = _orient_triangles_ccw(points, triangles)
    max_iterations = max(1, len(points) * len(points))

    for _ in range(max_iterations):
        flipped_any = False
        edge_to_triangles = _triangles_by_edge(triangles)
        for edge, incident in edge_to_triangles.items():
            if edge in constrained or len(incident) != 2:
                continue
            flipped = _flip_if_illegal(points, triangles, edge, incident)
            if flipped is not None:
                triangles = flipped
                flipped_any = True
                break
        if not flipped_any:
            return triangles
    return triangles


def _flip_if_illegal(
    points: np.ndarray,
    triangles: np.ndarray,
    edge: tuple[int, int],
    incident: tuple[int, int],
) -> np.ndarray | None:
    tri_a = triangles[incident[0]]
    tri_b = triangles[incident[1]]
    opposite_a = [int(v) for v in tri_a if int(v) not in edge]
    opposite_b = [int(v) for v in tri_b if int(v) not in edge]
    if len(opposite_a) != 1 or len(opposite_b) != 1:
        return None

    a = opposite_a[0]
    b = edge[0]
    c = opposite_b[0]
    d = edge[1]
    if not _is_convex_quad(points[[a, b, c, d]]):
        return None
    if not _point_in_circumcircle(points[c], points[a], points[b], points[d]):
        return None

    flipped = triangles.copy()
    flipped[incident[0]] = [a, b, c]
    flipped[incident[1]] = [a, c, d]
    return _orient_triangles_ccw(points, flipped)


def _is_ear(
    points: np.ndarray,
    polygon: list[int],
    a: int,
    b: int,
    c: int,
) -> bool:
    if _cross(points[a], points[b], points[c]) <= 1.0e-14:
        return False
    triangle = points[[a, b, c]]
    for vertex in polygon:
        if vertex in {a, b, c}:
            continue
        if _point_in_triangle(points[vertex], triangle):
            return False
    diagonal = (points[a], points[c])
    for idx, start in enumerate(polygon):
        stop = polygon[(idx + 1) % len(polygon)]
        if start in {a, c} or stop in {a, c}:
            continue
        if _segments_cross_strict(diagonal[0], diagonal[1], points[start], points[stop]):
            return False
    return True


def _filter_inside_triangles(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    if len(triangles) == 0:
        raise TriangulationError("triangulation produced no triangles")
    centroids = points[triangles].mean(axis=1)
    inside = np.asarray([_point_in_polygon(centroid, points) for centroid in centroids], dtype=bool)
    return np.asarray(triangles[inside], dtype=int)


def _make_triangulation(
    points: np.ndarray,
    beta: np.ndarray,
    triangles: np.ndarray,
    constrained_edges: np.ndarray,
) -> ConstrainedTriangulation:
    edges = _edges_from_triangles(triangles)
    point_triangles: list[list[int]] = [[] for _ in range(len(points))]
    for tid, triangle in enumerate(triangles):
        for vertex in triangle:
            point_triangles[int(vertex)].append(tid)
    return ConstrainedTriangulation(
        points=points,
        triangles=triangles,
        point_triangles=tuple(tuple(item) for item in point_triangles),
        edges=edges,
        constrained_edges=constrained_edges,
        beta=beta,
    )


def _validate_triangulation(triangulation: ConstrainedTriangulation) -> None:
    if len(triangulation.triangles) == 0:
        raise TriangulationError("triangulation produced no interior triangles")
    if not _contains_edges(triangulation.triangles, triangulation.constrained_edges):
        raise TriangulationError("triangulation does not preserve every boundary segment")

    edge_to_triangles = _triangles_by_edge(triangulation.triangles)
    constrained = {_edge_key(edge) for edge in triangulation.constrained_edges}
    for edge, incident in edge_to_triangles.items():
        if edge in constrained:
            if len(incident) != 1:
                raise TriangulationError("boundary edge has invalid triangle incidence")
        elif len(incident) != 2:
            raise TriangulationError("interior edge has invalid triangle incidence")

    for edge in triangulation.edges:
        key = _edge_key(edge)
        if key in constrained:
            continue
        p0, p1 = triangulation.points[key[0]], triangulation.points[key[1]]
        for constrained_edge in triangulation.constrained_edges:
            ckey = _edge_key(constrained_edge)
            if key[0] in ckey or key[1] in ckey:
                continue
            q0, q1 = triangulation.points[ckey[0]], triangulation.points[ckey[1]]
            if _segments_cross_strict(p0, p1, q0, q1):
                raise TriangulationError("interior edge crosses a boundary segment")

    _validate_connected(triangulation.triangles)


def _validate_connected(triangles: np.ndarray) -> None:
    edge_to_triangles = _triangles_by_edge(triangles)
    neighbours = [set() for _ in range(len(triangles))]
    for incident in edge_to_triangles.values():
        if len(incident) == 2:
            neighbours[incident[0]].add(incident[1])
            neighbours[incident[1]].add(incident[0])

    seen = {0}
    stack = [0]
    while stack:
        tid = stack.pop()
        for neighbour in neighbours[tid]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    if len(seen) != len(triangles):
        raise TriangulationError("triangulation is not connected")


def _common_triangles_in_point_order(
    point_triangles: tuple[tuple[int, ...], ...],
    vid1: int,
    vid2: int,
) -> tuple[int, ...]:
    tids: list[int] = []
    for tid1 in point_triangles[vid1]:
        for tid2 in point_triangles[vid2]:
            if tid1 == tid2:
                tids.append(tid1)
    return tuple(tids)


def _quadrilateral_vertices(
    triangles: np.ndarray,
    tids: tuple[int, int],
    vid1: int,
    vid2: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    tri0 = triangles[tids[0]]
    tri1 = triangles[tids[1]]
    diagonal = (vid1, vid2)
    v1 = next(int(vertex) for vertex in tri0 if int(vertex) not in diagonal)
    v3 = next(int(vertex) for vertex in tri1 if int(vertex) not in diagonal)
    vids = [vid1, v1, vid2, v3]
    ordered_tids = [int(tids[0]), int(tids[1])]

    start = next(idx for idx, vertex in enumerate(tri0) if int(vertex) == vid1)
    if int(tri0[(start + 1) % 3]) == vid2:
        vids[1], vids[3] = vids[3], vids[1]
        ordered_tids[0], ordered_tids[1] = ordered_tids[1], ordered_tids[0]
    return (tuple(vids), tuple(ordered_tids))


def _quadrilateral_neighbours(quadrilaterals: list[Quadrilateral]) -> tuple[tuple[int, ...], ...]:
    neighbours: list[list[int]] = [[] for _ in quadrilaterals]
    for quad in quadrilaterals:
        for candidate in quadrilaterals:
            if candidate.id == quad.id:
                continue
            if set(quad.tids) & set(candidate.tids):
                neighbours[quad.id].append(candidate.id)
    return tuple(tuple(item) for item in neighbours)


def _edges_from_triangles(triangles: np.ndarray) -> np.ndarray:
    edges = {_edge_key((triangle[0], triangle[1])) for triangle in triangles}
    edges.update({_edge_key((triangle[1], triangle[2])) for triangle in triangles})
    edges.update({_edge_key((triangle[2], triangle[0])) for triangle in triangles})
    return np.asarray(sorted(edges), dtype=int)


def _triangles_by_edge(triangles: np.ndarray) -> dict[tuple[int, int], tuple[int, ...]]:
    edge_map: dict[tuple[int, int], list[int]] = {}
    for tid, triangle in enumerate(triangles):
        for edge in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge_map.setdefault(_edge_key(edge), []).append(tid)
    return {edge: tuple(tids) for edge, tids in edge_map.items()}


def _boundary_edges(npoints: int) -> np.ndarray:
    return np.asarray([(idx, (idx + 1) % npoints) for idx in range(npoints)], dtype=int)


def _contains_edges(triangles: np.ndarray, edges: np.ndarray) -> bool:
    return all(_contains_edge(triangles, _edge_key(edge)) for edge in edges)


def _contains_edge(triangles: np.ndarray, edge: tuple[int, int]) -> bool:
    for triangle in triangles:
        if edge[0] in triangle and edge[1] in triangle:
            return True
    return False


def _orient_triangles_ccw(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=int).copy()
    for idx, triangle in enumerate(triangles):
        if _cross(points[triangle[0]], points[triangle[1]], points[triangle[2]]) < 0.0:
            triangles[idx, 1], triangles[idx, 2] = triangles[idx, 2], triangles[idx, 1]
    return triangles


def _cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _has_self_intersections(points: np.ndarray) -> bool:
    npoints = len(points)
    for i in range(npoints):
        a0 = points[i]
        a1 = points[(i + 1) % npoints]
        for j in range(i + 1, npoints):
            if abs(i - j) <= 1 or {i, j} == {0, npoints - 1}:
                continue
            b0 = points[j]
            b1 = points[(j + 1) % npoints]
            if _segments_cross_strict(a0, a1, b0, b1):
                return True
    return False


def _segments_cross_strict(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> bool:
    o1 = _cross(a0, a1, b0)
    o2 = _cross(a0, a1, b1)
    o3 = _cross(b0, b1, a0)
    o4 = _cross(b0, b1, a1)
    return (o1 * o2 < -1.0e-14) and (o3 * o4 < -1.0e-14)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x = point[0]
    y = point[1]
    inside = False
    for start, stop in zip(polygon, np.roll(polygon, -1, axis=0)):
        if _point_on_segment(point, start, stop):
            return True
        crosses = (start[1] > y) != (stop[1] > y)
        if crosses:
            x_intersection = (stop[0] - start[0]) * (y - start[1]) / (stop[1] - start[1]) + start[0]
            if x < x_intersection:
                inside = not inside
    return inside


def _point_on_segment(point: np.ndarray, start: np.ndarray, stop: np.ndarray) -> bool:
    length = np.linalg.norm(stop - start)
    if length <= 1.0e-14:
        return bool(np.linalg.norm(point - start) <= 1.0e-14)
    cross = abs(_cross(start, stop, point)) / length
    dot = float(np.dot(point - start, stop - start))
    return cross <= 1.0e-12 and -1.0e-12 <= dot <= length * length + 1.0e-12


def _point_in_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    a, b, c = triangle
    area = _cross(a, b, c)
    if area <= 1.0e-14:
        return False
    w0 = _cross(point, b, c) / area
    w1 = _cross(point, c, a) / area
    w2 = _cross(point, a, b) / area
    return w0 >= -1.0e-14 and w1 >= -1.0e-14 and w2 >= -1.0e-14


def _point_in_circumcircle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    ax = a[0] - point[0]
    ay = a[1] - point[1]
    bx = b[0] - point[0]
    by = b[1] - point[1]
    cx = c[0] - point[0]
    cy = c[1] - point[1]
    determinant = (
        (ax * ax + ay * ay) * (bx * cy - by * cx)
        - (bx * bx + by * by) * (ax * cy - ay * cx)
        + (cx * cx + cy * cy) * (ax * by - ay * bx)
    )
    return bool(determinant > 1.0e-14)


def _is_convex_quad(points: np.ndarray) -> bool:
    signs = []
    for idx in range(4):
        signs.append(_cross(points[idx], points[(idx + 1) % 4], points[(idx + 2) % 4]))
    return all(sign > 1.0e-14 for sign in signs) or all(sign < -1.0e-14 for sign in signs)
