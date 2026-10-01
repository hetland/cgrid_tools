"""Internal array ownership and strict JSON helpers."""

from __future__ import annotations

import numpy as np


def readonly(array: np.ndarray) -> np.ndarray:
    """Own an immutable buffer, including against ``setflags(write=True)``."""
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def json_array(array: np.ndarray) -> list:
    """Encode absent coordinates as JSON null (decoded as NaN by NumPy)."""
    return np.where(np.isfinite(array), array, None).tolist()
