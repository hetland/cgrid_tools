"""Shared low-level geometry helpers.

These helpers are used across ``generator``, ``crdt``, and ``triangulation``.
Consolidating them here keeps the numerical definitions in a single place so
that tolerance tweaks and edge-case fixes propagate everywhere consistently.
"""

from __future__ import annotations

import numpy as np


def edge_key(edge: tuple[int, int] | np.ndarray) -> tuple[int, int]:
    """Return an undirected-edge key ``(min, max)`` suitable for dict/set use."""
    a = int(edge[0])
    b = int(edge[1])
    return (a, b) if a < b else (b, a)


def signed_area(points: np.ndarray) -> float:
    """Signed polygon area via the shoelace formula (positive = CCW).

    ``points`` must be shape ``(N, 2)``.
    """
    # Translation avoids cancellation for small polygons in projected coordinates.
    x = points[:, 0] - points[0, 0]
    y = points[:, 1] - points[0, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def points_on_segment(
    x: np.ndarray,
    y: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    tol_perp: float = 1.0e-10,
    tol_parallel: float = 1.0e-12,
) -> np.ndarray:
    """Boolean mask of points lying on the closed segment ``start -> stop``.

    ``x``/``y`` are arrays of identical shape holding point coordinates.
    ``start``/``stop`` are 2-vectors (x, y).
    """
    dx = float(stop[0]) - float(start[0])
    dy = float(stop[1]) - float(start[1])
    length = float(np.hypot(dx, dy))
    if length <= 1.0e-14:
        return np.hypot(x - float(start[0]), y - float(start[1])) <= tol_parallel
    cross = np.abs((x - float(start[0])) * dy - (y - float(start[1])) * dx) / length
    dot = (x - float(start[0])) * dx + (y - float(start[1])) * dy
    return (cross <= tol_perp) & (dot >= -tol_parallel) & (dot <= length * length + tol_parallel)


def points_on_polygon_boundary(x: np.ndarray, y: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Boolean mask of points lying on any edge of the closed polygon."""
    on_boundary = np.zeros(x.shape, dtype=bool)
    start_points = polygon
    stop_points = np.roll(polygon, -1, axis=0)
    for start, stop in zip(start_points, stop_points):
        on_boundary |= points_on_segment(x, y, start, stop)
    return on_boundary


def points_in_polygon(x: np.ndarray, y: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test, inclusive of boundary points.

    ``polygon`` is shape ``(N, 2)`` describing a closed polygon (the last
    vertex is implicitly connected back to the first).
    """
    inside = np.zeros(x.shape, dtype=bool)
    x0 = polygon[:, 0]
    y0 = polygon[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)

    for xa, ya, xb, yb in zip(x0, y0, x1, y1):
        crosses = (ya > y) != (yb > y)
        if yb == ya:
            continue
        x_intersection = (xb - xa) * (y - ya) / (yb - ya) + xa
        inside ^= crosses & (x < x_intersection)

    return inside | points_on_polygon_boundary(x, y, polygon)
