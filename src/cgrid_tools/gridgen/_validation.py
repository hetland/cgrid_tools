"""Shared solver input and output validation."""

from __future__ import annotations

import numpy as np

from cgrid_tools.metrics import cell_jacobians


class GridGenerationError(RuntimeError):
    """A solver could not produce a grid with valid active cells."""


def grid_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2 or any(
        isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2
        for n in shape
    ):
        raise ValueError("shape must contain two integer node counts, each at least two")
    return int(shape[0]), int(shape[1])


def validate_cells(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> None:
    """Reject invalid active cells; do not hide failed cells by changing masks."""
    if not np.any(mask):
        raise GridGenerationError(
            "grid contains no active cells; increase resolution or revise the boundary"
        )
    jacobians = cell_jacobians(x, y)
    invalid = mask & np.any(~np.isfinite(jacobians) | (jacobians <= 0.0), axis=0)
    if np.any(invalid):
        count = int(np.count_nonzero(invalid))
        raise GridGenerationError(
            f"grid contains {count} folded or degenerate active cells; "
            "revise the boundary, resolution or focus, or try GRIDGEN_CRDT"
        )
